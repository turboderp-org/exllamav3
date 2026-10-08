"""
Canonical test model registry.

Model-level and end-to-end tests never hard-code checkpoint paths. They name a model by role ("dense",
"moe", "recurrent", ...) and the registry resolves the role to a directory:

    tests/models.yaml          tracked: the canonical roles, their tags and default paths relative to the model root
    tests/models.local.yaml    untracked, optional: per-machine overrides and extra entries
    $EXL3_TEST_MODELS          optional: one more override file (highest precedence)

The model root comes from --model-root or $EXL3_TEST_MODEL_ROOT. An entry whose directory is missing makes
the tests that need it skip, so the suite runs anywhere and covers whatever the machine has.

Entry fields:
    path    directory of the model (absolute, or relative to the model root)
    tags    feature tags used to build test matrices (e.g. "moe", "recurrent", "vision", "mtp")
    draft   optional registry id of a draft model paired with this one
    vram    GiB of VRAM a single-device load needs; tests skip on smaller devices (default: no requirement)
    notes   free text
"""

import os
from dataclasses import dataclass, field

import yaml

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ModelEntry:
    id: str
    path: str | None
    tags: frozenset[str] = field(default_factory = frozenset)
    draft: str | None = None
    vram: float = 0.0
    notes: str = ""

    @property
    def available(self) -> bool:
        return self.path is not None and os.path.isfile(os.path.join(self.path, "config.json"))


class Registry:

    def __init__(self, root: str | None = None, extra_files: list[str] | None = None):
        self.root = root
        self.sources = []
        self.entries: dict[str, ModelEntry] = {}
        files = [os.path.join(TESTS_DIR, "models.yaml"), os.path.join(TESTS_DIR, "models.local.yaml")]
        files += [f for f in (extra_files or []) if f]
        for f in files:
            if os.path.isfile(f):
                self._load(f)

    def _load(self, filename: str):
        with open(filename, encoding = "utf8") as f:
            data = yaml.safe_load(f) or {}
        self.sources.append(filename)
        if data.get("root") and self.root is None:
            self.root = os.path.expanduser(data["root"])
        for mid, spec in (data.get("models") or {}).items():
            spec = spec or {}
            prev = self.entries.get(mid)
            path = spec.get("path", prev.path if prev else None)
            self.entries[mid] = ModelEntry(
                id = mid,
                path = path,
                tags = frozenset(spec["tags"]) if "tags" in spec else (prev.tags if prev else frozenset()),
                draft = spec.get("draft", prev.draft if prev else None),
                vram = float(spec.get("vram", prev.vram if prev else 0.0)),
                notes = spec.get("notes", prev.notes if prev else ""),
            )

    def resolve(self, entry: ModelEntry) -> ModelEntry:
        path = entry.path
        if path is not None:
            path = os.path.expanduser(path)
            if not os.path.isabs(path):
                path = os.path.join(self.root, path) if self.root else None
        return ModelEntry(entry.id, path, entry.tags, entry.draft, entry.vram, entry.notes)

    def get(self, mid: str) -> ModelEntry:
        if mid not in self.entries:
            raise KeyError(f"Unknown test model '{mid}' (not in {', '.join(self.sources) or 'any registry file'})")
        return self.resolve(self.entries[mid])

    def ids(self, *tags: str, all_tags: bool = True) -> list[str]:
        """Registry ids whose tags include all (or any, all_tags = False) of the given tags"""
        want = set(tags)
        out = []
        for mid, e in self.entries.items():
            if not want or (want <= e.tags if all_tags else want & e.tags):
                out.append(mid)
        return out
