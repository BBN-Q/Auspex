"""Hardware-free VISA address normalization; no PyVISA dependency or DNS lookup."""

import ipaddress
import re


def normalize_visa_address(address, *, ipv4_device=None):
    """Canonicalize common VISA spellings, expanding bare IPv4 only with a driver's default.

    Preserve case-sensitive serial paths, USB serial numbers and LAN device names.
    Unknown aliases are left untouched; this is not physical-device discovery and
    does not equate different endpoints on the same host.
    """
    if not isinstance(address, str) or not address.strip():
        raise ValueError("A nonempty instrument address is required")
    address = address.strip()
    try:
        ipv4 = str(ipaddress.IPv4Address(address))
    except ipaddress.AddressValueError:
        ipv4 = None
    if ipv4 is not None:
        if ipv4_device is None:
            raise ValueError("Bare IPv4 addresses require a driver-specific endpoint; use a complete VISA address")
        address = f"TCPIP0::{ipv4}::{ipv4_device}"
    parts = address.split("::")
    match = re.fullmatch(r"(TCPIP|GPIB|USB)(\d*)", parts[0], re.IGNORECASE)
    if match:
        kind = match[1].upper()
        parts[0] = kind + str(int(match[2] or "0"))
        if parts[-1].upper() not in ("INSTR", "SOCKET", "INTFC", "RAW"):
            parts.append("INSTR")
        parts[-1] = parts[-1].upper()
        if kind == "TCPIP" and len(parts) >= 3:
            parts[1] = parts[1].lower()  # DNS names are case-insensitive; do not resolve them.
            if len(parts) == 3 and parts[-1] == "INSTR":
                parts.insert(2, "inst0")
            if len(parts) == 4:
                if parts[2].lower() == "inst0":
                    parts[2] = "inst0"
                elif parts[-1] == "SOCKET":
                    parts[2] = str(int(parts[2]))
        elif kind == "GPIB" and parts[-1] == "INSTR":
            parts[1:-1] = [str(int(value)) for value in parts[1:-1]]
            if len(parts) == 4 and parts[2] == "0":
                del parts[2]  # Absent secondary address is zero in VISA.
        elif kind == "USB" and len(parts) in (5, 6):
            parts[1:3] = [str(int(value, 16 if value.lower().startswith("0x") else 10)) for value in parts[1:3]]
            if len(parts) == 5:
                parts.insert(4, "0")
            parts[4] = str(int(parts[4]))
        return "::".join(parts)
    # ASRL may contain a Unix device path. Uppercasing the entire resource changes it.
    match = re.match(r"(PRLGX-ASRL|PRLGX-TCPIP|ASRL)(.*)", parts[0], re.IGNORECASE)
    if match:
        suffix = match[2]
        parts[0] = match[1].upper() + (str(int(suffix)) if suffix.isdecimal() else suffix)
        if len(parts) == 1 or parts[-1].upper() not in ("INSTR", "INTFC"):
            parts.append("INSTR")
        parts[-1] = parts[-1].upper()
        return "::".join(parts)
    return address
