from __future__ import annotations

from typing import Optional, Protocol

import zmq

from gui_live_tracking.bridge_transport import TrackingResult, serialize_tracking_result
from gui_live_tracking.config import LiveTrackingConfig


class TrackingResultSink(Protocol):
    def start(self) -> None:
        ...

    def send_result(self, result: TrackingResult) -> None:
        ...

    def stop(self) -> None:
        ...


class NullTrackingResultSink:
    def start(self) -> None:
        pass

    def send_result(self, result: TrackingResult) -> None:
        del result

    def stop(self) -> None:
        pass


class Ros2BridgeResultSink:
    def __init__(self, config: LiveTrackingConfig, context: Optional[zmq.Context] = None):
        self.endpoint = config.bridge_result_endpoint
        self._external_context = context
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None

    def start(self) -> None:
        self.stop()
        self._context = self._external_context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDTIMEO, 1)
        self._socket.connect(self.endpoint)

    def send_result(self, result: TrackingResult) -> None:
        if self._socket is None:
            return
        try:
            self._socket.send_multipart(serialize_tracking_result(result), flags=zmq.NOBLOCK)
        except zmq.Again:
            return

    def stop(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = None
        if self._external_context is None:
            self._context = None
