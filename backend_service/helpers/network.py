"""Network utility functions."""
from __future__ import annotations

import ipaddress
import socket

import psutil


def _local_ipv4_addresses() -> list[str]:
    discovered: set[str] = set()
    for interface_addresses in psutil.net_if_addrs().values():
        for address in interface_addresses:
            if address.family != socket.AF_INET:
                continue
            value = str(address.address or "").strip()
            if not value or value.startswith("127.") or value.startswith("169.254."):
                continue
            try:
                ip = ipaddress.ip_address(value)
            except ValueError:
                continue
            if ip.is_loopback or ip.is_link_local:
                continue
            discovered.add(value)
    return sorted(discovered)
