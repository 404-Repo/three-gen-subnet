import json
import logging
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

import bittensor as bt
from aioquic.asyncio.client import connect
from aioquic.h3.connection import H3_ALPN, ErrorCode
from aioquic.h3.events import (
    DataReceived,
    H3Event,
    HeadersReceived,
)
from aioquic.quic.configuration import QuicConfiguration
from aioquic.tls import SessionTicket
from pydantic import BaseModel

from validator.gateway.http3_client.connection_protocol import ConnectionProtocol
from validator.gateway.http3_client.url import URL


logger = logging.getLogger(__name__)


class NotAbleToGetStatus(Exception):
    pass


class Http3Exception(Exception):
    pass


class Http3Response(BaseModel):
    data: str
    latency: float


class Http3Client:
    def __init__(self) -> None:
        self._host_to_session_ticket: dict[str, SessionTicket] = {}

    async def get(
        self, *, url: str, headers: dict[str, Any] | None = None, payload: dict[str, Any] | None = None
    ) -> Http3Response:
        parsed_url = URL(url)
        web_data = json.dumps(payload).encode("utf-8") if payload else None
        async with self._get_client(
            host=parsed_url.host,
            port=parsed_url.port,
        ) as client:
            web_headers = {}
            if web_data:
                web_headers.update(
                    {
                        "content-length": str(len(web_data)),
                        "content-type": "application/json",
                    }
                )
            if headers:
                web_headers.update(headers)
            http_events = await client.get(url, headers=web_headers, data=web_data)
            status_code = Http3Client._get_status(http_events=http_events)
            if status_code != 200:
                raise Http3Exception(f"Status code: {status_code}")
            response = Http3Client._get_data(http_events=http_events)
            latency = client.get_smoothed_rtt()
            return Http3Response(data=response, latency=latency)

    async def post(self, *, url: str, payload: dict[str, Any], headers: dict[str, Any] | None = None) -> Http3Response:
        web_data = json.dumps(payload).encode("utf-8")
        parsed_url = URL(url)
        async with self._get_client(
            host=parsed_url.host,
            port=parsed_url.port,
        ) as client:
            web_headers = {
                "content-length": str(len(web_data)),
                "content-type": "application/json",
            }
            if headers:
                web_headers.update(headers)
            http_events = await client.post(url, headers=web_headers, data=web_data)
            status_code = Http3Client._get_status(http_events=http_events)
            if status_code != 200:
                raise Http3Exception(f"Status code: {status_code}")
            response = Http3Client._get_data(http_events=http_events)
            latency = client.get_smoothed_rtt()
            return Http3Response(data=response, latency=latency)

    async def post_form_data(self, *, url: str, data: dict[str, Any]) -> Http3Response:
        parsed_url = URL(url)
        async with self._get_client(
            host=parsed_url.host,
            port=parsed_url.port,
        ) as client:
            boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
            headers = {"content-type": f"multipart/form-data; boundary={boundary}"}
            payload_parts = []
            for key, value in data.items():
                payload_parts.append(f"--{boundary}\r\n".encode())
                payload_parts.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
                if isinstance(value, bytes):
                    payload_parts.append(value)
                else:
                    payload_parts.append(str(value).encode())
                payload_parts.append(b"\r\n")
            payload_parts.append(f"--{boundary}--\r\n".encode())
            payload_data = b"".join(payload_parts)
            headers["content-length"] = str(len(payload_data))
            http_events = await client.post(
                url,
                data=payload_data,
                headers=headers,
            )
            status_code = Http3Client._get_status(http_events=http_events)
            if status_code != 200:
                raise Http3Exception(f"Status code: {status_code}")
            response = Http3Client._get_data(http_events=http_events)
            latency = client.get_smoothed_rtt()
            return Http3Response(data=response, latency=latency)

    @staticmethod
    def _get_data(*, http_events: deque[H3Event]) -> str:
        data: bytes = b""
        for http_event in http_events:
            if isinstance(http_event, DataReceived):
                data += http_event.data
        return data.decode()

    @staticmethod
    def _get_status(*, http_events: deque[H3Event]) -> int:
        for http_event in http_events:
            if isinstance(http_event, HeadersReceived):
                for key, value in http_event.headers:
                    if key == b":status":
                        status_code = int(value.decode())
                        return status_code
        raise NotAbleToGetStatus()

    def _make_session_ticket_saver(self, *, host: str) -> Callable[[SessionTicket], None]:
        def save_session_ticket(ticket: SessionTicket) -> None:
            self._host_to_session_ticket[host] = ticket

        return save_session_ticket

    def _get_config(self, *, host: str) -> QuicConfiguration:
        config = QuicConfiguration(
            is_client=True,
            alpn_protocols=H3_ALPN,
        )
        if host in self._host_to_session_ticket:
            config.session_ticket = self._host_to_session_ticket[host]
        # Uncomment to enable quic logging. Create quic folder in the root of the project.
        # config.quic_logger = QuicFileLogger(Path(__file__).resolve().parent.parent / "quic")
        return config

    @asynccontextmanager
    async def _get_client(self, *, host: str, port: int) -> AsyncGenerator[ConnectionProtocol, None]:
        save_session_ticket = self._make_session_ticket_saver(host=host)
        config = self._get_config(host=host)
        try:
            async with connect(
                host,
                port,
                configuration=config,
                create_protocol=ConnectionProtocol,
                session_ticket_handler=save_session_ticket,
                wait_connected=False,
            ) as client:
                try:
                    yield client  # type: ignore
                finally:
                    client.close(error_code=ErrorCode.H3_NO_ERROR)
        except Exception as e:
            bt.logging.error(f"Error connecting to {host}:{port}: {e}")
            raise e
