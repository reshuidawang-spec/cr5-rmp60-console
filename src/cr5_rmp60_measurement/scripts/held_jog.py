"""Per-press cancellation and GUI heartbeat for a bounded held jog.

Only this small token is shared between the GUI and ROS worker threads. ROS
calls stay on the worker thread, including the Stop issued after cancellation.
"""

import time
from threading import Event, Lock


class HeldJogRequest:
    HEARTBEAT_TIMEOUT_SEC = 0.5

    def __init__(self):
        self._stopped = Event()
        self._emergency = Event()
        self._lock = Lock()
        self._last_heartbeat = time.monotonic()

    def keep_alive(self):
        with self._lock:
            self._last_heartbeat = time.monotonic()

    def stop(self, emergency=False):
        if emergency:
            self._emergency.set()
        self._stopped.set()

    def stop_requested(self):
        with self._lock:
            expired = time.monotonic() - self._last_heartbeat > self.HEARTBEAT_TIMEOUT_SEC
        if expired:
            self._stopped.set()
        return self._stopped.is_set()

    @property
    def emergency(self):
        return self._emergency.is_set()
