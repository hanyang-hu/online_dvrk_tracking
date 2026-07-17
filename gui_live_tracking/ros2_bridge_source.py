from __future__ import annotations

from typing import Optional

import zmq

from gui_live_tracking.bridge_transport import deserialize_tracking_sample
from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.frame_source import TrackingSample


class Ros2BridgeFrameSource:
    def __init__(self, config: LiveTrackingConfig, context: Optional[zmq.Context] = None):
        self.endpoint = config.bridge_input_endpoint
        self._external_context = context
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._poller: Optional[zmq.Poller] = None

    def start(self) -> None:
        self.stop()
        self._context = self._external_context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.endpoint)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

    def get_sample(self, timeout_sec: float = 0.5) -> Optional[TrackingSample]:
        if self._socket is None or self._poller is None:
            raise RuntimeError("Ros2BridgeFrameSource.start() must be called before get_sample().")

        events = dict(self._poller.poll(max(0, int(timeout_sec * 1000))))
        if self._socket not in events:
            return None

        newest: Optional[TrackingSample] = None
        while True:
            try:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            newest = deserialize_tracking_sample(parts)
        return newest

    def stop(self) -> None:
        if self._poller is not None and self._socket is not None:
            try:
                self._poller.unregister(self._socket)
            except KeyError:
                pass
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = None
        self._poller = None
        if self._external_context is None:
            self._context = None
