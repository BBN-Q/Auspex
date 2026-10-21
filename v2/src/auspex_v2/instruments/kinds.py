"""Fixed instrument role families used by plans, drivers, and services."""
from enum import StrEnum


class InstrumentKind(StrEnum):
    MICROWAVE_SOURCE = "MICROWAVE_SOURCE"
    DC_SOURCE = "DC_SOURCE"
    AWG = "AWG"
    DIGITIZER = "DIGITIZER"
    VNA = "VNA"
    SPECTRUM_ANALYZER = "SPECTRUM_ANALYZER"
    OSCILLOSCOPE = "OSCILLOSCOPE"
    FUNCTION_GENERATOR = "FUNCTION_GENERATOR"
    PULSE_GENERATOR = "PULSE_GENERATOR"
    LOCKIN = "LOCKIN"
    ATTENUATOR = "ATTENUATOR"
    SWITCH = "SWITCH"
    MAGNET_SUPPLY = "MAGNET_SUPPLY"
    TEMPERATURE_CONTROLLER = "TEMPERATURE_CONTROLLER"
    DAQ = "DAQ"
    SENSOR = "SENSOR"


def instrument_kind(value):
    """Normalize an enum member or exact saved string; reject open-ended kinds."""
    try:
        return InstrumentKind(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown instrument KIND: {value!r}") from exc
