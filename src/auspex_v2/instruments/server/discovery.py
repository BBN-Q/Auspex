"""VISA candidates and explicit SCPI identity probes, without driver construction."""

import ipaddress
import re

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


def resource_manager():
    """Use the same optional backend as live VISA drivers."""
    import pyvisa
    return pyvisa.ResourceManager("@py")


def enumerate_visa():
    """Include serial and socket resources, not just VISA's default INSTR subset."""
    manager = resource_manager()
    try:
        return [{"resource": resource, "serial": resource.upper().startswith("ASRL"),
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
                    if driver.IDN_STRING and driver.IDN_STRING in identity], "error": None}
        finally:
            device.close()
    finally:
        manager.close()
