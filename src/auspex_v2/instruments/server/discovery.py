"""VISA candidates and explicit SCPI identity probes, without driver construction."""

import errno
import ipaddress
import re

from ...paths import auspex_directory

from ..drivers.registry import DRIVERS


def resource_key(resource):
    """Recognize common TCPIP INSTR aliases, including drivers' bare IPv4 shorthand.

    This does not resolve DNS or claim to detect every physical-device alias.
    """
    value = resource.upper()
    try:
        return f"TCPIP::{ipaddress.IPv4Address(resource)}::INST0::INSTR"
    except ValueError:
        pass
    match = re.fullmatch(r"TCPIP(?:0)?::([^:]+)(?:::(.*))?::INSTR", value)
    if match:
        return f"TCPIP::{match[1]}::{match[2] or 'INST0'}::INSTR"
    return value


def read_whitelist():
    """Reload server-local display overrides; a missing file is normal, other failures are visible."""
    path = auspex_directory().parent / "instrument_whitelist.txt"
    try:
        addresses = {resource_key(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.lstrip().startswith("#")}
    except FileNotFoundError:
        addresses = set()
    except (OSError, UnicodeError) as exc:
        return set(), str(path), f"Cannot read whitelist {path}: {exc}"
    return addresses, str(path), None


def permission_denied(error):
    """Recognize attributable OS/VISA/serial access-denial errors, never backend log messages."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if (isinstance(error, PermissionError) or
                isinstance(error, OSError) and error.errno in (errno.EACCES, errno.EPERM) or
                getattr(getattr(error, "error_code", None), "name", None) == "error_no_permission"):
            return True
        text = str(error).casefold()
        if "permission denied" in text or "vi_error_npermission" in text:
            return True
        error = error.__cause__ or error.__context__
    return False


def hidden_reason(candidate, *, show_serial=False, show_prologix=False, hide_permission_denied=True):
    """Apply independent display filters; whitelist entries override visibility, not probe eligibility."""
    if candidate.get("whitelisted"):
        return None
    if hide_permission_denied and candidate.get("permission_denied"):
        return "permission denied"
    if candidate.get("prologix"):
        return None if show_prologix else "Prologix"
    if candidate["serial"] and not show_serial:
        return "serial"
    return None


def bulk_identifiable(candidate):
    """Serial and Prologix controller endpoints always require explicit per-resource probing."""
    return not candidate["registered"] and not candidate["serial"] and not candidate.get("prologix")


def filter_summary(candidates, **filters):
    """Count hidden rows by their first applicable filter, so totals do not double-count."""
    counts = {"serial": 0, "Prologix": 0, "permission denied": 0}
    whitelisted = 0
    for candidate in candidates:
        reason = hidden_reason(candidate, **filters)
        if reason:
            counts[reason] += 1
        whitelisted += bool(candidate.get("whitelisted"))
    return ("Hidden: " + ", ".join(f"{count} {reason}" for reason, count in counts.items()) +
            f". Whitelist: {whitelisted} matching resources shown.")


def resource_manager():
    """Use the same optional backend as live VISA drivers."""
    import pyvisa
    return pyvisa.ResourceManager("@py")


def enumerate_visa():
    """Include serial and socket resources, not just VISA's default INSTR subset."""
    manager = resource_manager()
    try:
        return [{"resource": resource, "serial": resource.upper().startswith("ASRL"),
                 "prologix": resource.upper().startswith("PRLGX"), "permission_denied": False,
                 "identity": None, "drivers": [], "error": None}
                for resource in sorted(set(manager.list_resources("?*")))]
    finally:
        manager.close()


def identify_visa(resource):
    """Send only *IDN?, with bounded open/I/O waits, and always close the resource."""
    manager = resource_manager()
    try:
        device = manager.open_resource(resource, open_timeout=1000)
        try:
            device.timeout = 1000
            device.read_termination = "\n"
            device.write_termination = "\n"
            identity = device.query("*IDN?").strip()
            if not identity:
                raise ValueError("Empty *IDN? response")
            return {"identity": identity, "drivers": [name for name, driver in DRIVERS.items()
                    if driver.IDN_STRING and driver.IDN_STRING in identity], "error": None,
                    "permission_denied": False}
        finally:
            device.close()
    finally:
        manager.close()
