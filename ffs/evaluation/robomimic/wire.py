from __future__ import annotations

import socket
from typing import Any

import msgpack
import numpy as np


def pack_array(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tobytes(),
            "dtype": obj.dtype.str,
            "shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            "__npgeneric__": True,
            "data": obj.item(),
            "dtype": obj.dtype.str,
        }
    return obj


def unpack_array(obj: dict[str, Any]) -> Any:
    if obj.get("__ndarray__"):
        array = np.ndarray(
            buffer=obj["data"],
            dtype=np.dtype(obj["dtype"]),
            shape=tuple(obj["shape"]),
        )
        return array.copy()
    if obj.get("__npgeneric__"):
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


def send_msg(sock: socket.socket, data: dict[str, Any]) -> None:
    payload = msgpack.packb(data, default=pack_array, use_bin_type=True)
    sock.sendall(len(payload).to_bytes(4, "big"))
    sock.sendall(payload)


def recv_msg(sock: socket.socket) -> dict[str, Any]:
    size = int.from_bytes(_recv_exact(sock, 4), "big")
    return msgpack.unpackb(_recv_exact(sock, size), raw=False, object_hook=unpack_array)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


__all__ = ["pack_array", "recv_msg", "send_msg", "unpack_array"]

