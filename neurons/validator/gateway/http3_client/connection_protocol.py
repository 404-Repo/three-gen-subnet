import asyncio
import json
import logging
from collections import deque
from typing import Any

import aioquic
import bittensor as bt
from aioquic.asyncio import QuicConnectionProtocol
from aioquic.h3.connection import H3Connection
from aioquic.h3.events import (
    DataReceived,
    H3Event,
    HeadersReceived,
)
from aioquic.quic.events import QuicEvent

from validator.gateway.http3_client.http_request import HttpRequest
from validator.gateway.http3_client.url import URL


logger = logging.getLogger("client")

USER_AGENT = "aioquic/" + aioquic.__version__


class ConnectionProtocol(QuicConnectionProtocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._http: H3Connection = H3Connection(self._quic)
        self._request_events: dict[int, deque[H3Event]] = {}
        self._request_waiter: dict[int, asyncio.Future[deque[H3Event]]] = {}
        self._data: str = ""
        self._status_code: int | None = None

    async def get(self, url: str, headers: dict[str, Any] | None = None, data: bytes | None = None) -> deque[H3Event]:
        if data is not None:
            return await self._request(HttpRequest(method="GET", url=URL(url), headers=headers, content=data))
        return await self._request(HttpRequest(method="GET", url=URL(url), headers=headers))

    async def post(self, url: str, data: bytes, headers: dict[str, Any] | None = None) -> deque[H3Event]:
        return await self._request(HttpRequest(method="POST", url=URL(url), content=data, headers=headers))

    def get_smoothed_rtt(self) -> float:
        """
        Get the smoothed RTT (Round Trip Time) from the QUIC connection.
        Returns the current estimate of the connection's smoothed RTT in seconds.
        """
        return float(self._quic._loss._rtt_smoothed)

    def http_event_received(self, event: H3Event) -> None:
        if isinstance(event, (HeadersReceived | DataReceived)):
            stream_id = event.stream_id
            if stream_id in self._request_events:
                self._request_events[event.stream_id].append(event)

                # TODO: remove hack to handle bad error code
                if isinstance(event, HeadersReceived):
                    for h in event.headers:
                        if h[0] == b":status":
                            self._status_code = int(h[1])
                            return

                # TODO: remove hack when server will be fixed (stream_end issue). Check by stream_end.
                if isinstance(event, DataReceived):
                    bt.logging.trace(f"Data received: {event.data!r}")
                    data = event.data.decode()
                    self._data += data
                if self._status_code != 200 or self._data == "Ok":
                    request_waiter = self._request_waiter.pop(stream_id)
                    request_waiter.set_result(self._request_events.pop(stream_id))
                try:
                    json.loads(self._data)
                    request_waiter = self._request_waiter.pop(stream_id)
                    request_waiter.set_result(self._request_events.pop(stream_id))
                except ValueError:
                    return

    def quic_event_received(self, event: QuicEvent) -> None:
        if self._http is not None:
            for http_event in self._http.handle_event(event):
                self.http_event_received(http_event)

    async def _request(self, request: HttpRequest) -> deque[H3Event]:
        stream_id = self._quic.get_next_available_stream_id()
        headers = [
            (b":method", request.method.encode()),
            (b":scheme", request.url.scheme.encode()),
            (b":authority", request.url.authority.encode()),
            (b":path", request.url.full_path.encode()),
            (b"user-agent", USER_AGENT.encode()),
        ]
        headers += [(k.encode(), v.encode() if isinstance(v, str) else v) for (k, v) in request.headers.items()]
        self._http.send_headers(
            stream_id=stream_id,
            headers=headers,
            end_stream=not request.content,
        )
        if request.content:
            self._http.send_data(stream_id=stream_id, data=request.content, end_stream=True)

        waiter = self._loop.create_future()
        self._request_events[stream_id] = deque()
        self._request_waiter[stream_id] = waiter
        self.transmit()

        try:
            return await asyncio.wait_for(asyncio.shield(waiter), timeout=30)
        except TimeoutError:
            logger.warning(f"Timeout waiting for response on stream {stream_id}")
            self._request_waiter.pop(stream_id, None)
            self._request_events.pop(stream_id, None)
            raise
