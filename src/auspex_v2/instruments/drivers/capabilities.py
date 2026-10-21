"""Offline descriptions of explicitly declared driver members (no construction/I/O)."""


def describe_driver(driver):
    settings = {}
    for name, spec in driver.commands.items():
        settings[name] = {"read": spec.get_string is not None,
                          "write": spec.set_string is not None,
                          "args": list(spec.additional_args)}
    for name, spec in driver.properties.items():
        settings[name] = {"read": spec.readable, "write": spec.writable, "args": []}
    return {"settings": settings, "operations": list(driver.operations)}
