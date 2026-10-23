#!/usr/bin/env python3
"""Install a desktop entry pointing at this checkout; never starts the application."""
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
destination = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "applications"
destination.mkdir(parents=True, exist_ok=True)
# Desktop Exec uses its own quoting/percent expansion, not shell quoting.
def quoted(value):
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$').replace('%', '%%') + '"'

entry = destination / "cr5-five-branch-sphere-workbench.desktop"
entry.write_text("\n".join([
    "[Desktop Entry]", "Type=Application", "Version=1.0",
    "Name=CR5 五向测针标准球实验台", "Comment=模型配置、定位、仿真与采集",
    f"Exec=bash {quoted(root.parents[1] / 'launch_console.sh')}",
    f"Path={root}", f"Icon={root / 'desktop/cr5-five-branch-workbench.svg'}",
    "Terminal=false", "StartupNotify=true", "Categories=Science;Engineering;", ""
]), encoding="utf-8")
entry.chmod(0o755)
print(entry)
