"""
Re-vendor the forward-only flash-linear-attention kernels into
exllamav3/vendor/fla from an installed fla package.

Copies the kernel modules listed in FILES, keeping only the top-level definitions reachable from
the host-side forward functions in ROOTS (the ones __init__.py calls), so backward kernels, autograd
wrappers, context-parallel and in-kernel-gate code are dropped. `from fla...` imports are rewritten
to relative imports, `@dispatch(...)` decorators and fla's autotune-cache wrapper are replaced by
plain triton.autotune. utils.py, __init__.py and LICENSE are hand-maintained and left untouched;
after regenerating, run tests/test_fla_vendored_.py against the same fla version.

Written against fla 0.5.2. A newer fla may move functions between modules (update FILES /
REEXPORT) or add imports the rewrite doesn't know (they are listed as "unresolved").
"""
import ast, os, re, sys, importlib.util
_spec = importlib.util.find_spec("fla")
assert _spec and _spec.origin, "flash-linear-attention is not installed"
SRC = os.path.dirname(_spec.origin)
DST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exllamav3", "vendor", "fla")
FILES = {   # fla module -> vendored module name
    "fla.ops.gated_delta_rule.chunk_fwd": "gdn_chunk_fwd",
    "fla.ops.gated_delta_rule.wy_fast": "gdn_wy_fast",
    "fla.ops.kda.chunk_intra": "kda_chunk_intra",
    "fla.ops.kda.chunk_intra_token_parallel": "kda_chunk_intra_token_parallel",
    "fla.ops.kda.wy_fast": "kda_wy_fast",
    "fla.ops.common.chunk_h": "chunk_h",
    "fla.ops.common.chunk_delta_h": "chunk_delta_h",
    "fla.ops.common.chunk_o": "chunk_o",
    "fla.ops.common.chunk_scaled_dot_kkt": "chunk_scaled_dot_kkt",
    "fla.ops.utils.cumsum": "cumsum",
    "fla.ops.utils.solve_tril": "solve_tril",
    "fla.ops.utils.op": "op",
    "fla.ops.utils.constant": "constant",
    "fla.ops.utils.index": "index",
    "fla.modules.l2norm": "l2norm",
    "fla.ops.gla.chunk": "gla_chunk",
}
# names re-exported by package __init__s -> defining module
REEXPORT = {"fla.ops.utils": {"chunk_local_cumsum": "fla.ops.utils.cumsum", "chunk_global_cumsum": "fla.ops.utils.cumsum",
                              "solve_tril": "fla.ops.utils.solve_tril", "prepare_chunk_indices": "fla.ops.utils.index",
                              "prepare_chunk_offsets": "fla.ops.utils.index", "prepare_lens": "fla.ops.utils.index",
                              "softmax_fwd": None, "softmax_bwd": None, "prepare_block_csr": None, "chunk_global_reversed_cumsum": "fla.ops.utils.cumsum"}}
ROOTS = ["chunk_local_cumsum", "chunk_gated_delta_rule_fwd_intra", "chunk_gated_delta_rule_fwd_h", "chunk_fwd_o",
         "chunk_kda_fwd_intra", "chunk_gla_fwd_o_gk", "chunk_fwd_h", "l2norm_fwd"]

def is_bwd(name): return "bwd" in name or "backward" in name
def src_path(mod): return os.path.join(SRC, *mod.split(".")[1:]) + ".py"

mods = {}
for mod in FILES:
    text = open(src_path(mod)).read(); tree = ast.parse(text); lines = text.splitlines(keepends = True)
    defs, imports, others, roots_extra = {}, [], [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            is_autograd = isinstance(node, ast.ClassDef) and any("Function" in ast.unparse(b) for b in node.bases)
            if is_autograd:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "forward":
                        roots_extra.append(item)
                continue
            if is_bwd(node.name): continue
            defs[node.name] = (node, start, node.end_lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        else:
            others.append(node)
    mods[mod] = dict(text = text, lines = lines, defs = defs, imports = imports, others = others, roots_extra = roots_extra)

# name -> module resolution: local defs first, then imports from fla modules
needed_mods = set()
def resolve(mod, name):
    if name in mods[mod]["defs"]: return (mod, name)
    for imp in mods[mod]["imports"]:
        if isinstance(imp, ast.ImportFrom) and imp.module and imp.module.startswith("fla"):
            for a in imp.names:
                if (a.asname or a.name) == name:
                    target = imp.module
                    if target in REEXPORT: target = REEXPORT[target].get(a.name)
                    if target in mods:
                        needed_mods.add(target)
                        if a.name in mods[target]["defs"]: return (target, a.name)
    return None

def refs(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

reach = set()
work = []
for mod, m in mods.items():
    for r in ROOTS:
        if r in m["defs"]: work.append((mod, r))
while work:
    t = work.pop()
    if t in reach: continue
    reach.add(t)
    mod, name = t
    node = mods[mod]["defs"][name][0]
    for n in refs(node):
        t2 = resolve(mod, n)
        if t2 and t2 not in reach: work.append(t2)

# second pass: module-level statements of needed modules reference defs too
changed = True
while changed:
    changed = False
    for mod in list(needed_mods | {m for (m, n) in reach}):
        for o in mods[mod]["others"]:
            for n in refs(o):
                t = resolve(mod, n)
                if t and t not in reach:
                    work.append(t); changed = True
    while work:
        t = work.pop()
        if t in reach: continue
        reach.add(t)
        mod, name = t
        for n in refs(mods[mod]["defs"][name][0]):
            t2 = resolve(mod, n)
            if t2 and t2 not in reach: work.append(t2)

os.makedirs(DST, exist_ok = True)
unresolved = set()
for mod, m in mods.items():
    kept = sorted([m["defs"][n] for (mm, n) in reach if mm == mod], key = lambda d: d[1])
    if not kept and mod not in needed_mods:
        print(f"-- {mod}: nothing reachable, skipped"); continue
    out = [m["lines"][0:7]]  # license header
    body_text = "".join(m["lines"])
    # imports: keep non-fla imports verbatim, rewrite fla ones
    imp_lines = []
    kept_names = " ".join(ast.unparse(d[0]) for d in kept) + " " + " ".join(ast.unparse(o) for o in m["others"])
    for imp in m["imports"]:
        if isinstance(imp, ast.ImportFrom) and imp.module and imp.module.startswith("fla"):
            for a in imp.names:
                nm = a.asname or a.name
                if not re.search(r"\b" + re.escape(nm) + r"\b", kept_names): continue
                target = imp.module
                if target in REEXPORT: target = REEXPORT[target].get(a.name)
                if target in FILES:
                    imp_lines.append(f"from .{FILES[target]} import {a.name}" + (f" as {nm}" if a.asname else ""))
                elif target == "fla.utils":
                    imp_lines.append(f"from .utils import {a.name}" + (f" as {nm}" if a.asname else ""))
                elif a.name in ("dispatch", "fla_cache_autotune"):
                    pass   # stripped / replaced by plain triton.autotune in the post-processing below
                else:
                    unresolved.add((mod, imp.module, a.name)); imp_lines.append(f"# UNRESOLVED: from {imp.module} import {a.name}")
        else:
            imp_lines.append(ast.get_source_segment(m["text"], imp))
    kept_names = {d[0].name for d in kept}
    is_alias = lambda o: isinstance(o, ast.Assign) and isinstance(o.value, ast.Name) and o.value.id in kept_names
    other_lines = [ "".join(m["lines"][o.lineno - 1:o.end_lineno]) for o in m["others"] if not is_alias(o) ]
    alias_lines = [ "".join(m["lines"][o.lineno - 1:o.end_lineno]) for o in m["others"] if is_alias(o) ]
    text = "".join(out[0]) + "\n" + "\n".join(imp_lines) + "\n\n" + "".join(other_lines) + "\n"
    for node, start, end in kept:
        text += "\n" + "".join(m["lines"][start - 1:end]) + "\n"
    if alias_lines:
        text += "\n" + "".join(alias_lines)
    text = re.sub(r"^@dispatch\(.*\)\n", "", text, flags = re.M)
    text = text.replace("@fla_cache_autotune(", "@triton.autotune(")
    text = re.sub(r"^# UNRESOLVED:.*\n", "", text, flags = re.M)
    defined = set(d[0].name for d in kept) | set(re.findall(r"import (\w+)", text)) | set(re.findall(r"^\s*def (\w+)", text, flags = re.M))
    text = "\n".join(l for l in text.split("\n") if not (re.match(r"^(\w+) = (\w+)$", l) and re.match(r"^(\w+) = (\w+)$", l).group(2) not in defined))
    text = re.sub(r"\n{3,}", "\n\n\n", text)
    open(os.path.join(DST, FILES[mod] + ".py"), "w").write(text)
    print(f"{FILES[mod]:32s} kept {len(kept):2d} defs, {text.count(chr(10)):5d} lines: {', '.join(d[0].name for d in kept)}")
print("unresolved:", sorted(unresolved))
