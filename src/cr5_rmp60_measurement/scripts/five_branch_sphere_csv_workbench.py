#!/usr/bin/env python3
"""Extend the installed workbench with legacy trigger CSV import.

Keeps the installed robot runner, fitting and collision checks unchanged.
This entry point itself sends no robot commands.
"""
import argparse
import sys
from pathlib import Path

from import_sphere_gui_contacts import BRANCHES, import_contacts


def make_window_class(base):
    from PyQt5.QtWidgets import QFileDialog, QPushButton

    class CsvWorkbench(base):
        def on_jog_busy(self, busy):
            self.csv_import_jog_busy = bool(busy)
            super().on_jog_busy(busy)

        def _build_ui(self):
            super()._build_ui()
            self.import_csv_button = QPushButton("导入已有GUI触发CSV（保留原数据）")
            self.import_csv_button.clicked.connect(self.import_trigger_csv)
            self.centralWidget().layout().insertWidget(2, self.import_csv_button)

        def import_trigger_csv(self):
            try:
                if self.acquire_worker is not None and self.acquire_worker.isRunning():
                    raise ValueError("自动采集中不能导入，请等待本批结束")
                if getattr(self, "csv_import_jog_busy", False) or getattr(self, "armed", False):
                    raise ValueError("请先结束点动并解除等待触发，再导入")
                path, _ = QFileDialog.getOpenFileName(self, "选择同一固定球位的触发CSV", "", "CSV (*.csv)")
                if not path:
                    return
                report = import_contacts(path, self.session(), installation_id=self.install_edit.text().strip(),
                                         standard_sphere_id=self.sphere_id_edit.text().strip(),
                                         placement_id=self.placement(), robot_ip=self.ip_edit.text().strip())
                self.update_counts()
                counts = report["total_branch_counts"]
                self.log(f"CSV已导入{report['added_rows']}条，跳过重复{report['duplicate_rows']}条；" +
                         "，".join(f"{b}:{counts[b]}" for b in BRANCHES))
                missing = {b: n for b, n in report["missing_to_existing_two_per_branch_gate"].items() if n}
                if missing:
                    self.log(f"尚缺分支触发：{missing}。这是现有粗定位的最低数量要求；还需几何和覆盖检查。")
                else:
                    self.log("数量已满足既有粗定位要求，可点击“拟合粗球心”；拟合通过后继续原仿真与自动采集。")
            except Exception as exc:
                self.show_error(exc)

    return CsvWorkbench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True,
                        help="包含已部署实验台及其完整依赖的项目根目录")
    args, qt_args = parser.parse_known_args()
    runtime = args.runtime_root.expanduser().resolve()
    if not (runtime / "scripts/five_branch_sphere_workbench.py").is_file():
        parser.error("指定目录中不存在已部署的五向标准球实验台")
    sys.path.insert(0, str(runtime / "scripts"))
    from five_branch_sphere_workbench import Workbench
    from PyQt5.QtWidgets import QApplication
    app = QApplication([sys.argv[0], *qt_args])
    window = make_window_class(Workbench)()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
