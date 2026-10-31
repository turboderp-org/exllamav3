"""Stdlib-only source tests; run directly to avoid the Torch pytest conftest.

    python3 -B tests/kernels/attention/test_paged_fixed_partition_cpu.py -v
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


if "pytest" in sys.modules:
    pytestmark = sys.modules["pytest"].mark.nogpu

TREE = Path(__file__).resolve().parents[3]
SOURCE = TREE / "exllamav3/modules/attention_fn/triton_paged.py"
SMEM = SOURCE.with_name("smem.py")
ADAPTIVE = ast.parse("""
splits = num_splits
if splits is None:
    splits = max(1, min(target // programs, triton.cdiv(max_k_len, 1 * block_n), 128))
split_len = triton.cdiv(triton.cdiv(max_k_len, splits), block_n) * block_n
""").body


def function(name):
    return next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def selection(max_k_len, split_size=None, block_n=64, num_splits=None, target=160, programs=8):
    """Execute only the production prepare() arithmetic, never module initialization."""
    decode = function("paged_attn_triton_decode")
    prepare = next(n for n in decode.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
    end = next(i for i, n in enumerate(prepare.body)
               if isinstance(n, ast.If) and ast.unparse(n.test) == "splits > 1")
    ns = dict(max_k_len=max_k_len, split_size=split_size, block_n=block_n,
              num_splits=num_splits, target=target, programs=programs,
              triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b))
    body = ast.Module(body=copy.deepcopy(prepare.body[1:end]), type_ignores=[])
    exec(compile(ast.fix_missing_locations(body), str(SOURCE), "exec"), ns)
    return ns["split_len"], ns["splits"]


class FixedPartitionTests(unittest.TestCase):
    def test_optional_argument_is_trailing_and_defaults_to_none(self):
        args = function("paged_attn_triton_decode").args
        self.assertEqual(args.args[-1].arg, "split_size")
        self.assertIsNone(ast.literal_eval(args.defaults[-1]))

    def test_fixed_length_is_not_rebalanced(self):
        for capacity in (1, 255, 256, 257, 768, 3073):
            for size in (256, 512):
                with self.subTest(capacity=capacity, size=size):
                    self.assertEqual(selection(capacity, size),
                                     (size, max(1, (capacity + size - 1) // size)))

    def test_rejects_invalid_lengths_and_conflicting_count(self):
        with self.assertRaisesRegex(ValueError, "split_size"):
            selection(3073, 63)
        for size in (0, -64, 63, 257, True, 256.0, "256"):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "split_size"):
                selection(3073, size)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            selection(3073, 256, num_splits=3)

    def test_window_partitions_use_fixed_logical_boundaries(self):
        kernel = function("_paged_attn_decode_split_kernel")
        branch = next(n for n in kernel.body if isinstance(n, ast.If) and n.orelse
                      and any(isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "n_start"
                              for x in n.orelse))
        code = compile(ast.Module(body=[branch], type_ignores=[]), str(SOURCE), "exec")
        for split in range(5):
            ns = dict(WINDOW_LEFT=127, FIXED_SPLITS=True, total_k_len=770, q_len=1,
                      split=split, split_len=256, num_splits=5, BLOCK_N=64,
                      tl=SimpleNamespace(maximum=max, minimum=min, cdiv=lambda a, b: (a + b - 1) // b))
            exec(code, ns)
            self.assertEqual((ns["n_start"], ns["n_end"]), (split * 256, min((split + 1) * 256, 770)))

    def test_kernel_flag_is_bound_in_eager_and_graph_calls(self):
        decode = function("paged_attn_triton_decode")
        prepare = next(n for n in decode.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
        args = next(n.value for n in prepare.body if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == "args")
        self.assertIsInstance(args, ast.Tuple)
        self.assertEqual(ast.unparse(args.elts[-1]), "split_size is not None")
        bc = ast.parse(SOURCE.with_name("bc_attn.py").read_text())
        configure = next(n for n in ast.walk(bc) if isinstance(n, ast.FunctionDef) and n.name == "_configure")
        signature = next(n.value for n in configure.body if isinstance(n, ast.Assign)
                         and ast.unparse(n.targets[0]) == "sig")
        self.assertIn("FIXED_SPLITS", ast.unparse(signature))
        consts = next(n.value for n in configure.body if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == "consts")
        self.assertIsInstance(consts, ast.Call)
        values = {n.arg: n.value for n in consts.keywords}
        self.assertIn("FIXED_SPLITS", values)
        self.assertIs(ast.literal_eval(values["FIXED_SPLITS"]), False)

    def test_fixed_mode_has_separate_tile_pick_key(self):
        decode = function("paged_attn_triton_decode")
        device_scope = next(n for n in decode.body if isinstance(n, ast.With))
        key = next(n for n in device_scope.body if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == "pick_key")
        following = device_scope.body[device_scope.body.index(key) + 1]
        self.assertIsInstance(following, ast.If)
        self.assertEqual(ast.unparse(following.test), "split_size is not None")
        self.assertEqual(ast.unparse(following.body[0]), "pick_key += (True,)")

    def test_adaptive_statements_are_unchanged(self):
        decode = function("paged_attn_triton_decode")
        prepare = next(n for n in decode.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
        branch = next(n for n in prepare.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "split_size is not None")
        self.assertEqual(ast.dump(ast.Module(body=branch.orelse, type_ignores=[])),
                         ast.dump(ast.Module(body=ADAPTIVE, type_ignores=[])))

    def test_adaptive_results(self):
        for capacity in (1, 256, 257, 3073, 65536):
            for explicit in (None, 1, 3, 64):
                with self.subTest(capacity=capacity, num_splits=explicit):
                    splits = explicit if explicit is not None else max(1, min(20, (capacity + 63) // 64, 128))
                    length = (((capacity + splits - 1) // splits + 63) // 64) * 64
                    self.assertEqual(selection(capacity, num_splits=explicit), (length, splits))


def decode():
    return next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'paged_attn_triton_decode')


def source_pick():
    """Use the checkout's actual cached selector and tile ladder, not copies.

    Only these function definitions execute. Framework import, module setup,
    resource queries and compilation are replaced by inert CPU doubles.
    """
    module = ast.parse(SMEM.read_text())
    functions = [copy.deepcopy(n) for n in module.body
                 if isinstance(n, ast.FunctionDef) and n.name in ('pick_config', 'halving_ladder')]
    ns = dict(_dev_index=lambda d: 0, smem_limit=lambda d: 64,
              torch=SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda d: (11, 5))),
              _picks={}, _debug=False, NoFittingConfig=RuntimeError)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SMEM), 'exec'), ns)
    return ns


def execute_selection_path(split_size, warm):
    """Actual outer setup, validation, pick_config and final prepare before append.

    Inert prepare enforces the original per-selected-tile validation; pick_config
    is the exact production function. No framework/device/module init executes.
    """
    fn = decode()
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'candidates')
    nodes = fn.body[start:]
    with_node = next(n for n in nodes if isinstance(n, ast.With))
    stop = next(i for i, n in enumerate(with_node.body) if isinstance(n, ast.If)
                and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == '_paged_kv_update'
                        for x in ast.walk(n)))
    pick_body = with_node.body[:stop]
    # Preserve any new outer validation nodes through the next group_size assign.
    end = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == 'group_size')
    prelude = nodes[:end]
    ns = source_pick()
    calls = []
    def prepare(bn):
        calls.append(bn)
        if type(split_size) is not int or split_size <= 0 or split_size % bn:
            raise ValueError('split_size alignment')
        return ((), 2, None, None)
    def probe(kernel, args, **kw):
        return 65 if calls[-1] == 64 else 32
    from contextlib import nullcontext
    ns.update(split_size=split_size, num_splits=None, block_n=None, cfg_block_n=64,
              prepare=prepare,
              q=SimpleNamespace(device=0), head_dim=96, hd_pad=128, qck=0, qcv=0,
              q_len=1, block_rows=16, h_blocks=1, causal=True, window_left=127,
              window_right=0, has_sinks=False, num_warps=4, num_stages=1,
              _paged_attn_decode_split_kernel=object(), shared_bytes=probe,
              torch=SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda d: (11, 5))))
    # Key is independent of split length: seed the exact safe-length selection.
    if warm:
        safe = 256
        def safe_prepare(bn):
            calls.append(bn)
            if safe % bn:
                raise ValueError('split_size alignment')
            return ((), 2, None, None)
        ns['prepare'] = safe_prepare
        ns['split_size'] = safe
        exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(prelude + pick_body), type_ignores=[])), str(SOURCE), 'exec'), ns)
        ns['block_n'] = None
        ns['num_splits'] = None  # Fresh caller frame, not the prior call's selected split count.
        ns['prepare'] = prepare
        ns['split_size'] = split_size
        calls.clear()
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(prelude + pick_body), type_ignores=[])), str(SOURCE), 'exec'), ns)
    return ns['block_n'], calls


class FollowupTests(unittest.TestCase):
    def test_rejects_fallback_only_alignment_cold_and_warm(self):
        for warm in (False, True):
            with self.subTest(warm=warm), self.assertRaisesRegex(ValueError, 'split_size'):
                execute_selection_path(32, warm)

    def test_leading_candidate_alignment_works_cold_and_warm(self):
        for warm in (False, True):
            with self.subTest(warm=warm):
                selected, calls = execute_selection_path(256, warm)
                self.assertEqual(selected, 32)
                self.assertIn(32, calls)

    def test_validation_is_outside_prepare_before_cached_pick(self):
        fn = decode()
        pick_scope = next(n for n in fn.body if isinstance(n, ast.With))
        prefix = fn.body[:fn.body.index(pick_scope)]
        self.assertTrue(any(isinstance(n, ast.If) and ast.unparse(n.test) == 'split_size is not None'
                            and any(isinstance(x, ast.Raise) for x in ast.walk(n)) for n in prefix))

    def test_every_append_is_after_final_prepare_in_actual_scope(self):
        scope = next(n for n in decode().body if isinstance(n, ast.With))
        order = []
        for n in scope.body:
            calls = [x.func.id for x in ast.walk(n) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)]
            if 'prepare' in calls and 'pick_config' not in calls:
                order.append('prepare')
            if '_paged_kv_update' in calls:
                order.append('append')
        self.assertEqual(order, ['prepare', 'append'])

    def test_no_bare_asserts_in_public_cpu_test(self):
        path = TREE / 'tests/kernels/attention/test_paged_fixed_partition_cpu.py'
        self.assertEqual([n.lineno for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Assert)], [])


if __name__ == "__main__":
    unittest.main()
