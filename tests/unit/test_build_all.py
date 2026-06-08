"""Unit tests for the build/bake script (the bake logic, not SVG)."""
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "demos" / "proof-compare"
for p in (REPO_ROOT, BUILD_DIR, BUILD_DIR / "tracers"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_all


class TestBuildAll(unittest.TestCase):
    def test_build_all_has_four_scenarios_with_nodes(self):
        data = build_all.build_all()
        self.assertEqual(set(data), {"inference", "spec", "training", "coding"})
        for k, g in data.items():
            self.assertTrue(g["nodes"], f"{k} has no nodes")
            self.assertIn("edges", g)

    def test_bake_replaces_the_data_line(self):
        html = "<script>\nconst DATA = {};\nrenderAll();\n</script>\n"
        baked = build_all.bake(html, {"x": [1, 2]})
        # the DATA line is replaced and parses back to our object
        line = [ln for ln in baked.splitlines() if ln.startswith("const DATA = ")][0]
        obj = json.loads(line[len("const DATA = "):-1])
        self.assertEqual(obj, {"x": [1, 2]})
        self.assertIn("renderAll();", baked)  # rest of the file untouched

    def test_bake_survives_unicode_escapes_in_json(self):
        # The real footgun: JSON with \u sequences must not break re.sub.
        html = "const DATA = {};\n"
        build_all.bake(html, {"tok": "café ⟂"})  # must not raise

    def test_bake_requires_exactly_one_data_line(self):
        with self.assertRaises(RuntimeError):
            build_all.bake("no data line here\n", {"x": 1})


if __name__ == "__main__":
    unittest.main()
