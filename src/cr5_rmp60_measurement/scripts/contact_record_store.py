"""Read, save and remove contact rows without renumbering retained samples."""

import csv
import io
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def resolve_contact_output(value, default_name="gui_probe_contacts.csv"):
    """Accept a CSV filename or a directory, including '~' and trailing '/'."""
    path = Path(value).expanduser()
    if path.is_dir() or str(value).endswith(("/", os.sep)):
        path = path / Path(default_name).name
    if path.suffix.lower() != ".csv":
        path = path.with_name(path.name + ".csv")
    return path.resolve()


def _parse_records(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("CSV 表头为空或包含重复列")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV 数据列数与表头不一致，请先检查文件")
    return fields, rows


def read_contact_records(path):
    path = Path(path)
    if not path.exists():
        return []
    return _parse_records(path.read_bytes())[1]


def save_contact_records(source, destination):
    """Save the complete current CSV, preserving an existing destination backup."""
    source = Path(source).expanduser().resolve()
    destination = resolve_contact_output(destination, source.name)
    data = source.read_bytes()
    _, rows = _parse_records(data)
    if source == destination:
        with source.open("rb") as stream:
            os.fsync(stream.fileno())
        return destination, rows, None

    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = destination.read_bytes() if destination.exists() else None
    backup = None
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as temp:
            temp_path = Path(temp.name)
            temp.write(data)
            temp.flush()
            os.fsync(temp.fileno())
            os.fchmod(temp.fileno(), source.stat().st_mode & 0o777)
        current = destination.read_bytes() if destination.exists() else None
        if current != previous:
            raise ValueError("目标 CSV 正在被其他程序修改，请重试保存")
        if previous is not None and previous != data:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = destination.with_name(f"{destination.name}.before_save_{stamp}_{uuid4().hex[:8]}.bak")
            with backup.open("xb") as saved:
                saved.write(previous)
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return destination, rows, backup


def delete_contact_records(path, selected):
    """Delete selected full rows, backing up exact bytes before atomic replacement.

    Called on the ROS worker so deletion and collection writes are serialized.
    A stale selection causes the whole operation to fail without changing CSV.
    """
    path = Path(path).resolve()
    if not selected:
        raise ValueError("请先选择要删除的采集记录")
    original = path.read_bytes()
    fields, rows = _parse_records(original)

    def key(row):
        return tuple(str(row.get(field, "")) for field in fields)

    if any(set(row) != set(fields) for row in selected):
        raise ValueError("CSV 列结构已变化，请刷新记录后重新选择")
    requested = Counter(key(row) for row in selected)
    available = Counter(key(row) for row in rows)
    if requested - available:
        raise ValueError("选中记录已发生变化，请刷新记录后重新选择")
    remaining = []
    for row in rows:
        row_key = key(row)
        if requested[row_key]:
            requested[row_key] -= 1
        else:
            remaining.append(row)

    newline = "\r\n" if b"\r\n" in original else "\n"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator=newline)
    writer.writeheader()
    writer.writerows(remaining)
    encoding = "utf-8-sig" if original.startswith(b"\xef\xbb\xbf") else "utf-8"
    updated = stream.getvalue().encode(encoding)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(f"{path.name}.before_delete_{stamp}_{uuid4().hex[:8]}.bak")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as temp:
            temp_path = Path(temp.name)
            temp.write(updated)
            temp.flush()
            os.fsync(temp.fileno())
            os.fchmod(temp.fileno(), path.stat().st_mode & 0o777)
        if path.read_bytes() != original:
            raise ValueError("CSV 正在被其他程序修改，请刷新后重试")
        with backup.open("xb") as saved:
            saved.write(original)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return remaining, backup
