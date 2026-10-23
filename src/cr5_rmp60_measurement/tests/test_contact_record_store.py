"""Deletion uses temporary CSVs only; no robot connection is required."""

import csv
import os
import sys
import time
from pathlib import Path
from threading import Event

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import contact_record_store as store
import probe_calibration_gui as gui
from PyQt5.QtCore import QItemSelectionModel
from PyQt5.QtWidgets import QApplication


ROWS = [
    {"sample_index": "1", "session_id": "A", "timestamp": "1.0", "operator_note": "误采,点一"},
    {"sample_index": "2", "session_id": "A", "timestamp": "2.0", "operator_note": "保留\n备注"},
    {"sample_index": "3", "session_id": "A", "timestamp": "3.0", "operator_note": "误采点三"},
]


def write_rows(path, rows=ROWS):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ROWS[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_multiselect_delete_backs_up_exact_bytes_and_retains_values(tmp_path):
    path = tmp_path / "contacts.csv"
    write_rows(path)
    original = path.read_bytes()
    remaining, backup = store.delete_contact_records(path, [ROWS[0], ROWS[2]])
    assert remaining == [ROWS[1]]
    assert store.read_contact_records(path) == [ROWS[1]]
    assert backup.read_bytes() == original
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_duplicate_sample_number_in_different_session_is_not_deleted(tmp_path):
    path = tmp_path / "contacts.csv"
    other = dict(ROWS[0], session_id="B")
    write_rows(path, [ROWS[0], other])
    store.delete_contact_records(path, [other])
    assert store.read_contact_records(path) == [ROWS[0]]


def test_identical_rows_delete_only_selected_count(tmp_path):
    path = tmp_path / "contacts.csv"
    write_rows(path, [ROWS[0], ROWS[0]])
    store.delete_contact_records(path, [ROWS[0]])
    assert store.read_contact_records(path) == [ROWS[0]]


@pytest.mark.parametrize("selected", [[], [dict(ROWS[0], timestamp="stale")], [{"sample_index": "1"}]])
def test_stale_or_empty_selection_leaves_file_untouched(tmp_path, selected):
    path = tmp_path / "contacts.csv"
    write_rows(path)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        store.delete_contact_records(path, selected)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_deleting_all_keeps_header_for_subsequent_collection(tmp_path):
    path = tmp_path / "contacts.csv"
    write_rows(path)
    store.delete_contact_records(path, ROWS)
    assert store.read_contact_records(path) == []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        assert csv.DictReader(stream).fieldnames == list(ROWS[0])
    gui.write_stable_row(path, dict(ROWS[0], sample_index="4"))
    assert store.read_contact_records(path)[0]["sample_index"] == "4"


def test_replace_failure_preserves_original_and_backup(tmp_path, monkeypatch):
    path = tmp_path / "contacts.csv"
    write_rows(path)
    original = path.read_bytes()

    def fail(*args):
        raise OSError("模拟替换失败")

    monkeypatch.setattr(store.os, "replace", fail)
    with pytest.raises(OSError):
        store.delete_contact_records(path, [ROWS[0]])
    assert path.read_bytes() == original
    assert next(tmp_path.glob("*.bak")).read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_malformed_csv_refuses_deletion(tmp_path):
    path = tmp_path / "contacts.csv"
    path.write_text("sample_index,session_id\n1,A,extra\n")
    original = path.read_bytes()
    with pytest.raises(ValueError):
        store.delete_contact_records(path, [ROWS[0]])
    assert path.read_bytes() == original


def test_external_change_is_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "contacts.csv"
    write_rows(path)
    external = dict(ROWS[0], sample_index="4")
    original_fsync = store.os.fsync

    def change_during_write(fd):
        write_rows(path, [*ROWS, external])
        original_fsync(fd)

    monkeypatch.setattr(store.os, "fsync", change_during_write)
    with pytest.raises(ValueError, match="其他程序"):
        store.delete_contact_records(path, [ROWS[0]])
    assert store.read_contact_records(path) == [*ROWS, external]
    assert not list(tmp_path.glob("*.bak"))


def test_save_to_folder_preserves_complete_csv_and_source(tmp_path):
    source = tmp_path / "original.csv"
    write_rows(source)
    original = source.read_bytes()
    folder = tmp_path / "采集数据"
    folder.mkdir()
    saved, rows, backup = store.save_contact_records(source, folder)
    assert saved == folder / source.name
    assert saved.read_bytes() == original == source.read_bytes()
    assert rows == ROWS
    assert backup is None


def test_save_to_new_nested_filename_adds_csv_extension(tmp_path):
    source = tmp_path / "original.csv"
    write_rows(source)
    saved, _, _ = store.save_contact_records(source, tmp_path / "新目录" / "记录")
    assert saved.name == "记录.csv"
    assert saved.read_bytes() == source.read_bytes()


def test_save_same_file_never_duplicates_rows(tmp_path):
    source = tmp_path / "original.csv"
    write_rows(source)
    original = source.read_bytes()
    for _ in range(2):
        saved, rows, backup = store.save_contact_records(source, source)
        assert rows == ROWS
        assert backup is None
    assert saved.read_bytes() == original


def test_save_existing_destination_is_backed_up(tmp_path):
    source = tmp_path / "original.csv"
    target = tmp_path / "export.csv"
    write_rows(source)
    write_rows(target, [ROWS[1]])
    previous = target.read_bytes()
    saved, _, backup = store.save_contact_records(source, target)
    assert saved.read_bytes() == source.read_bytes()
    assert backup.read_bytes() == previous


def test_failed_save_preserves_source_and_destination(tmp_path, monkeypatch):
    source = tmp_path / "original.csv"
    target = tmp_path / "export.csv"
    write_rows(source)
    write_rows(target, [ROWS[1]])
    original_source, original_target = source.read_bytes(), target.read_bytes()

    def fail(*args):
        raise PermissionError("模拟保存失败")

    monkeypatch.setattr(store.os, "replace", fail)
    with pytest.raises(PermissionError):
        store.save_contact_records(source, target)
    assert source.read_bytes() == original_source
    assert target.read_bytes() == original_target
    assert not list(tmp_path.glob("*.tmp"))


def test_directory_and_home_paths_resolve_to_real_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert store.resolve_contact_output("~/", "contacts.csv") == tmp_path / "contacts.csv"
    assert store.resolve_contact_output(str(tmp_path / "new") + "/", "contacts.csv") == tmp_path / "new" / "contacts.csv"
    assert store.resolve_contact_output(tmp_path / "UPPER.CSV") == tmp_path / "UPPER.CSV"


def wait_until(predicate):
    deadline = time.monotonic() + 2.0
    pause = Event()
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        pause.wait(0.001)
    assert predicate()


@pytest.fixture
def window(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "gui.csv"
    write_rows(path)
    monkeypatch.setattr(sys, "argv", ["test", "--output", str(path)])
    window = gui.ProbeCalibrationWindow(gui.parse_args())
    window.refresh_timer.stop()
    wait_until(lambda: window.records_path == str(path) and not window.gui_busy)
    yield window, path
    window.close()
    app.processEvents()
    assert not window.worker_thread.isRunning()


def select_rows(window, indices):
    for index in indices:
        window.record_table.selectionModel().select(
            window.record_table.model().index(index, 0),
            QItemSelectionModel.Select | QItemSelectionModel.Rows,
        )


def test_gui_loads_history_and_deletes_multiple_rows_from_csv(window):
    window, path = window
    original = path.read_bytes()
    assert window.record_table.rowCount() == 3
    select_rows(window, [0, 2])
    assert window.delete_records_btn.isEnabled()
    window.delete_records_btn.click()
    wait_until(lambda: not window.gui_busy)
    assert window.record_table.rowCount() == 1
    assert window.record_table.item(0, 0).text() == "2"
    assert store.read_contact_records(path) == [ROWS[1]]
    assert next(path.parent.glob("*.bak")).read_bytes() == original
    assert window.worker.sample_index == 3
    assert window.worker.node is None


def test_gui_busy_and_no_selection_cannot_delete(window):
    window, path = window
    original = path.read_bytes()
    assert not window.delete_records_btn.isEnabled()
    window.delete_selected_records()
    select_rows(window, [0])
    window.on_busy_changed(True)
    assert not window.delete_records_btn.isEnabled()
    window.delete_selected_records()
    assert path.read_bytes() == original
    window.on_busy_changed(False)


def test_switching_output_never_deletes_from_previous_file(window):
    window, path = window
    original = path.read_bytes()
    select_rows(window, [0])
    other = path.with_name("other.csv")
    write_rows(other, [ROWS[2]])
    window.output_edit.setText(str(other))
    window.delete_selected_records()
    wait_until(lambda: window.records_path == str(other) and not window.gui_busy)
    assert path.read_bytes() == original
    assert store.read_contact_records(other) == [ROWS[2]]
    assert window.record_table.rowCount() == 1


def test_gui_manual_save_routes_future_collection_to_new_file(window, monkeypatch):
    window, source = window
    original = source.read_bytes()
    target = source.parent / "另存数据" / "标定记录"
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target), "CSV (*.csv)"))
    window.manual_save_btn.click()
    target = target.with_suffix(".csv")
    wait_until(lambda: window.records_path == str(target) and not window.gui_busy)
    assert target.read_bytes() == original == source.read_bytes()
    assert window.settings()["output"] == str(target)
    assert window.record_table.rowCount() == 3
    assert window.worker.sample_index == 3
    assert "已手动保存 3 条" in window.log_text.toPlainText()
    assert window.worker.node is None


def test_gui_folder_picker_saves_existing_records_with_filename(window, monkeypatch):
    window, source = window
    folder = source.parent / "新采集目录"
    folder.mkdir()
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(folder))
    window.save_folder_btn.click()
    target = folder / source.name
    wait_until(lambda: window.records_path == str(target) and not window.gui_busy)
    assert store.read_contact_records(target) == ROWS
    assert window.output_edit.text() == str(target)
    assert window.settings()["output"] == str(target)


def test_gui_typing_folder_preserves_records_instead_of_clearing_table(window):
    window, source = window
    folder = source.parent / "输入的目录"
    folder.mkdir()
    window.output_edit.setText(str(folder))
    window.output_edit.editingFinished.emit()
    target = folder / source.name
    wait_until(lambda: window.records_path == str(target) and not window.gui_busy)
    assert store.read_contact_records(target) == ROWS
    assert window.record_table.rowCount() == 3


def test_gui_choose_new_csv_preserves_current_records(window, monkeypatch):
    window, source = window
    target = source.parent / "new.csv"
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target), "CSV (*.csv)"))
    window.output_browse_btn.click()
    wait_until(lambda: window.records_path == str(target) and not window.gui_busy)
    assert store.read_contact_records(target) == ROWS
    assert window.record_table.rowCount() == 3


def test_gui_cancel_save_and_busy_state_do_not_write(window, monkeypatch):
    window, source = window
    original = source.read_bytes()
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", lambda *args, **kwargs: ("", ""))
    window.manual_save_btn.click()
    assert source.read_bytes() == original
    window.on_busy_changed(True)
    assert not window.manual_save_btn.isEnabled()
    assert not window.save_folder_btn.isEnabled()
    window.save_records_to(source.parent / "should_not_exist.csv")
    assert not (source.parent / "should_not_exist.csv").exists()
    window.on_busy_changed(False)


def test_gui_failed_save_keeps_original_records_and_output(window, monkeypatch):
    window, source = window

    def fail(*args):
        raise PermissionError("模拟无权限")

    monkeypatch.setattr(gui, "save_contact_records", fail)
    window.save_records_to(source.parent / "failed.csv")
    wait_until(lambda: not window.gui_busy)
    assert window.records_path == str(source)
    assert window.output_edit.text() == str(source)
    assert window.record_table.rowCount() == 3
    assert "手动保存失败" in window.log_text.toPlainText()


def test_gui_selects_folder_before_first_sample(window, monkeypatch):
    window, source = window
    empty = source.parent / "fresh.csv"
    window.records_path = str(empty)
    window.sample_index_output = str(empty)
    window.output_edit.setText(str(empty))
    window.on_records_loaded(str(empty), [])
    folder = source.parent / "先选目录"
    folder.mkdir()
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(folder))
    window.save_folder_btn.click()
    target = folder / empty.name
    wait_until(lambda: window.records_path == str(target) and not window.gui_busy)
    assert window.settings()["output"] == str(target)
    gui.write_stable_row(window.settings()["output"], ROWS[0])
    assert store.read_contact_records(target) == [ROWS[0]]
