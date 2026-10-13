"""Local coordinator service and its synchronous client API."""

from .client import CoordinatorClient, RemoteError
from .service import PROTOCOL_VERSION, default_socket, main

__all__ = ["CoordinatorClient", "RemoteError", "PROTOCOL_VERSION", "default_socket", "main"]
