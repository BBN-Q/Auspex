"""Local server service and its synchronous client API."""

from .client import InstrumentServerClient, RemoteError
from .service import PROTOCOL_VERSION, default_socket, main

__all__ = ["InstrumentServerClient", "RemoteError", "PROTOCOL_VERSION", "default_socket", "main"]
