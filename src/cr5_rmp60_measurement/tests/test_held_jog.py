"""Offline held-jog tests; no ROS node or robot connection is created.

Run after sourcing scripts/source_runtime.sh, with QT_QPA_PLATFORM=offscreen.
"""

import csv
import os
import sys
import time
from threading import Event
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import held_jog
import jog_and_record_contacts as jog
import probe_calibration_gui as gui
from PyQt5.QtCore import QEvent, QTimer
from PyQt5.QtWidgets import QApplication


class FakeNode:
    def __init__(self, clock):
        self.clock = clock
        self.last_feed_time = clock[0]
        self.pose = [0.0] * 6
        self.di1 = False
        self.stops = 0
        self.moves = []
        self.movl_cli = object()
        self.on_spin = lambda: None
        self.on_speed = lambda: None

    def wait_fresh_feed(self):
        pass

    def set_speed(self, speed):
        self.on_speed()

    def feed_snapshot(self):
        return {"pose": list(self.pose), "di1": self.di1, "joints": [0.0] * 6}

    def stop_fast_then_confirm(self):
        self.stops += 1

    def check_ready_future(self, client, future):
        return future.result()


@pytest.fixture
def simulation(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(held_jog.time, "monotonic", lambda: clock[0])
    node = FakeNode(clock)

    def spin(node, timeout_sec):
        clock[0] += 0.01
        node.last_feed_time = clock[0]
        node.on_spin()

    def issue(node, target):
        node.moves.append(list(target))
        return SimpleNamespace(done=lambda: True, result=lambda: None)

    monkeypatch.setattr(jog.rclpy, "ok", lambda: True)
    monkeypatch.setattr(jog.rclpy, "spin_once", spin)
    monkeypatch.setattr(jog, "issue_movl", issue)
    monkeypatch.setattr(gui, "issue_movl", issue)
    return clock, node


@pytest.fixture
def settings(tmp_path):
    return dict(
        output=str(tmp_path / "contacts.csv"), session_id="test", workpiece_id="test",
        artifact_id="sphere", artifact_type="sphere", physical_ball_id="1",
        branch="y_neg", operator_note="", linear_step_mm=0.2, angular_step_deg=1.0,
        continuous_linear_limit_mm=10.0, continuous_angular_limit_deg=10.0,
        speed=1, auto_retract_mm=1.0, timeout_sec=5.0,
        position_tolerance_mm=0.08, orientation_tolerance_deg=0.08,
    )


def worker_for(node):
    worker = gui.RosProbeWorker()
    worker.node = node
    return worker


def test_expired_heartbeat_cannot_be_revived(simulation):
    clock, _ = simulation
    request = held_jog.HeldJogRequest()
    clock[0] += 0.6
    assert request.stop_requested()
    request.keep_alive()
    assert request.stop_requested()
    assert not held_jog.HeldJogRequest().stop_requested()


@pytest.mark.parametrize("release_during_speed", [False, True])
def test_release_before_dispatch_never_moves(simulation, settings, release_during_speed):
    _, node = simulation
    request = held_jog.HeldJogRequest()
    if release_during_speed:
        node.on_speed = request.stop
    else:
        request.stop()
    worker_for(node).run_held_jog("x", 1.0, settings, request)
    assert node.moves == []


def test_release_during_motion_stops_without_recording(simulation, settings):
    _, node = simulation
    request = held_jog.HeldJogRequest()
    node.on_spin = request.stop
    worker_for(node).run_held_jog("x", 1.0, settings, request)
    assert len(node.moves) == 1
    assert node.moves[0][0] == 10.0
    assert node.stops == 1
    assert not Path(settings["output"]).exists()


def test_heartbeat_loss_stops_motion(simulation, settings):
    _, node = simulation
    worker_for(node).run_held_jog("x", 1.0, settings, held_jog.HeldJogRequest())
    assert len(node.moves) == 1
    assert node.stops == 1


@pytest.mark.parametrize("axis,index,sign", [("x", 0, 1.0), ("rz", 5, -1.0)])
def test_limit_sends_one_move_and_does_not_restart(simulation, settings, axis, index, sign):
    _, node = simulation
    node.on_spin = lambda: node.pose.__setitem__(index, sign * 10.0)
    worker_for(node).run_held_jog(axis, sign, settings, held_jog.HeldJogRequest())
    assert len(node.moves) == 1
    assert node.moves[0][index] == sign * 10.0


def test_existing_di_refuses_motion(simulation, settings):
    _, node = simulation
    node.di1 = True
    worker_for(node).run_held_jog("x", 1.0, settings, held_jog.HeldJogRequest())
    assert node.moves == []


@pytest.mark.parametrize("emergency", [False, True])
def test_di_records_once_and_only_emergency_suppresses_retract(simulation, settings, emergency):
    _, node = simulation
    request = held_jog.HeldJogRequest()

    def contact_then_retract():
        node.di1 = True
        if len(node.moves) == 1:
            node.pose[0] = 0.3
            if emergency:
                request.stop(emergency=True)
        else:
            node.pose = list(node.moves[-1])

    node.on_spin = contact_then_retract
    worker = worker_for(node)
    rows = []
    worker.sampleRecorded.connect(rows.append)
    worker.run_held_jog("x", 1.0, settings, request)
    assert len(rows) == 1
    assert len(node.moves) == (1 if emergency else 2)
    assert rows[0]["trigger_flange_x"] == "0.3000"
    assert rows[0]["trigger_di1"] == "1"
    saved = list(csv.DictReader(Path(settings["output"]).open()))
    assert len(saved) == 1
    assert saved[0]["sample_index"] == "1"


def test_emergency_interrupts_contact_retract(simulation, settings):
    _, node = simulation
    request = held_jog.HeldJogRequest()

    def spin():
        node.di1 = True
        node.pose[0] = 0.3
        if len(node.moves) == 2:
            request.stop(emergency=True)

    node.on_spin = spin
    worker_for(node).run_held_jog("x", 1.0, settings, request)
    assert len(node.moves) == 2
    assert node.stops == 2


def test_stale_feed_stops_motion(simulation, settings):
    clock, node = simulation
    node.on_spin = lambda: setattr(node, "last_feed_time", clock[0] - 1.0)
    worker_for(node).run_held_jog("x", 1.0, settings, held_jog.HeldJogRequest())
    assert node.stops >= 1
    assert not Path(settings["output"]).exists()


def test_timeout_stops_motion(simulation, settings):
    _, node = simulation
    settings["timeout_sec"] = 0.02
    worker_for(node).run_held_jog("x", 1.0, settings, held_jog.HeldJogRequest())
    assert node.stops >= 1


def test_command_rejection_stops_instead_of_waiting_for_timeout(simulation, settings):
    _, node = simulation

    def rejected(client, future):
        raise RuntimeError("MovL rejected")

    node.check_ready_future = rejected
    worker = worker_for(node)
    messages = []
    worker.status.connect(lambda level, message: messages.append(message))
    worker.run_held_jog("x", 1.0, settings, held_jog.HeldJogRequest())
    assert node.stops >= 1
    assert any("MovL rejected" in message for message in messages)
    assert not Path(settings["output"]).exists()


def test_single_step_still_uses_selected_step(simulation, settings):
    _, node = simulation
    node.on_spin = lambda: node.pose.__setitem__(0, 0.2)
    worker_for(node).jog_axis("x", 1.0, settings)
    assert node.moves == [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]


@pytest.mark.parametrize("limit", [0.0, 100.0, float("nan")])
def test_invalid_limit_never_moves(simulation, settings, limit):
    _, node = simulation
    settings["continuous_linear_limit_mm"] = limit
    worker_for(node).run_held_jog("x", 1.0, settings, held_jog.HeldJogRequest())
    assert node.moves == []


def test_released_at_limit_does_not_wait_for_grace_period(simulation):
    clock, node = simulation
    request = held_jog.HeldJogRequest()
    spins = [0]

    def spin():
        spins[0] += 1
        if spins[0] == 2:
            request.stop()

    node.on_spin = spin
    start = clock[0]
    assert jog.wait_jog(node, [0.0] * 6, 5.0, 0.08, 0.08, stop_requested=request.stop_requested) == (None, False)
    assert clock[0] - start < 0.05
    assert node.stops == 1


def test_di_arriving_during_stop_is_preserved(simulation):
    _, node = simulation
    request = held_jog.HeldJogRequest()
    request.stop()
    node.stop_fast_then_confirm = lambda: setattr(node, "di1", True)
    snapshot, reached = jog.wait_jog(node, [10.0] * 6, 5.0, 0.08, 0.08, stop_requested=request.stop_requested)
    assert snapshot["di1"]
    assert not reached


@pytest.fixture
def window(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(sys, "argv", ["test", "--output", str(tmp_path / "gui.csv")])
    window = gui.ProbeCalibrationWindow(gui.parse_args())
    # Disconnect dispatch so a UI test cannot initialize ROS or send commands.
    window.continuousJogStartRequested.disconnect()
    window.stopRequested.disconnect()
    window.refresh_timer.stop()
    window.connected = True
    yield window
    window.close()
    app.processEvents()
    assert not window.worker_thread.isRunning()


def test_gui_heartbeat_never_dispatches_another_move(window):
    sent = []
    window.continuousJogStartRequested.connect(lambda *args: sent.append(args))
    button = window.continuous_buttons[1]
    button.setDown(True)
    button.pressed.emit()
    request = window.continuous_request
    for _ in range(10):
        window.keep_continuous_jog_alive()
    assert len(sent) == 1
    assert button.isEnabled()
    assert not window.continuous_buttons[0].isEnabled()
    window.start_continuous_step_jog("y", 1.0)
    assert len(sent) == 1
    button.setDown(False)
    button.released.emit()
    assert request.stop_requested()
    assert not window.continuous_heartbeat_timer.isActive()
    window.on_continuous_jog_finished(request)
    assert not window.gui_busy


def test_gui_deactivation_stops_and_emergency_still_reaches_released_token(window):
    window.start_continuous_step_jog("x", 1.0)
    request = window.continuous_request
    window.changeEvent(QEvent(QEvent.WindowDeactivate))
    assert request.stop_requested()
    assert not request.emergency
    window.emergency_stop()
    assert request.emergency


def test_gui_release_interrupts_worker_while_its_slot_is_running(window, monkeypatch):
    node = FakeNode([time.monotonic()])
    window.worker.node = node
    released = Event()
    pause = Event()

    def spin(node, timeout_sec):
        pause.wait(0.001)
        node.last_feed_time = time.monotonic()

    def issue(node, target):
        node.moves.append(target)
        return SimpleNamespace(done=lambda: True, result=lambda: None)

    def release():
        window.stop_continuous_step_jog()
        released.set()

    monkeypatch.setattr(jog.rclpy, "ok", lambda: True)
    monkeypatch.setattr(jog.rclpy, "spin_once", spin)
    monkeypatch.setattr(gui, "issue_movl", issue)
    # FakeNode has no ROS resources; detach it before normal window cleanup.
    window.continuousJogStartRequested.connect(window.worker.run_held_jog)
    window.start_continuous_step_jog("x", 1.0)
    QTimer.singleShot(20, release)
    deadline = time.monotonic() + 1.0
    while window.continuous_request is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        pause.wait(0.001)
    assert released.is_set()
    assert window.continuous_request is None
    assert len(node.moves) == 1
    assert node.stops == 1
    window.worker.node = None
