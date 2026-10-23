import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from import_sphere_gui_contacts import BRANCHES, import_contacts, read_contacts


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv = self.root / "source.csv"
        self.rows = []
        for i, branch in enumerate(BRANCHES):
            self.rows.append({"session_id": "one", "branch": branch, "trigger_di1": "1",
                              "command": "probe", "sample_index": str(i+1),
                              **{f"trigger_flange_{n}": str(i) for n in ("x", "y", "z", "rx", "ry", "rz")},
                              **{f"trigger_joint_{n}": str(n) for n in range(1, 7)}})
        self.write()

    def write(self):
        with self.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)

    def run_import(self, **kw):
        return import_contacts(self.csv, self.root / "run", installation_id="install",
                               standard_sphere_id="S1", **kw)

    def test_capture_pose_excluded_and_source_unchanged(self):
        self.rows[0]["command"] = "capture_pose"
        self.write()
        before = self.csv.read_bytes()
        report = self.run_import()
        self.assertEqual(report["added_rows"], 4)
        self.assertEqual(report["excluded"][0]["reason"], "not_a_trigger")
        self.assertEqual(before, self.csv.read_bytes())
        self.assertFalse(report["motion_authorized"])
        self.assertIsNone(report["sphere_center_mm"])

    def test_reimport_and_append_do_not_duplicate(self):
        self.assertEqual(self.run_import()["added_rows"], 5)
        self.assertEqual(self.run_import()["added_rows"], 0)
        self.rows.append({**self.rows[-1], "sample_index": "6"})
        self.write()
        self.assertEqual(self.run_import()["added_rows"], 1)
        self.assertEqual(len((self.root / "run/P0_manual.jsonl").read_text().splitlines()), 6)

    def test_modified_imported_row_rejected_without_partial_append(self):
        self.run_import()
        target = self.root / "run/P0_manual.jsonl"
        before = target.read_bytes()
        self.rows[0]["trigger_flange_x"] = "99"
        self.write()
        with self.assertRaises(ValueError):
            self.run_import()
        self.assertEqual(before, target.read_bytes())

    def test_multiple_sessions_require_explicit_selection(self):
        self.rows[0]["session_id"] = "two"
        self.write()
        with self.assertRaises(ValueError):
            self.run_import()
        self.assertEqual(self.run_import(source_session="one")["added_rows"], 4)

    def test_stale_selector_not_inferred_from_axes(self):
        for row in self.rows:
            row["branch"] = "y_neg"
        self.rows[0]["command"] = "x+"
        self.write()
        with self.assertRaisesRegex(ValueError, "标签"):
            read_contacts(self.csv)

    def test_frozen_plan_and_identity_changes_rejected(self):
        self.run_import()
        with self.assertRaises(ValueError):
            self.run_import(robot_ip="different")
        (self.root / "run/P0_manifest.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "冻结"):
            self.run_import()

    def test_nonfinite_joints_and_path_escape_rejected(self):
        self.rows[0]["trigger_joint_1"] = "nan"
        self.write()
        with self.assertRaises(ValueError):
            self.run_import()
        with self.assertRaises(ValueError):
            self.run_import(placement_id="../P0")


if __name__ == "__main__":
    unittest.main()
