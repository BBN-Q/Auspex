"""Optional PyVISA adapter; importing drivers never opens a VISA resource."""

from .addresses import normalize_visa_address


class VisaTransport:
    """Adapt a lazily opened PyVISA resource to the V2 SCPI interface."""

    def __init__(self, resource_name):
        try:
            import pyvisa
        except ImportError as exc:
            raise ImportError("PyVISA is required for real VISA instruments; install auspex-v2[visa]") from exc
        self._manager = pyvisa.ResourceManager("@py")
        try:
            self._resource = self._manager.open_resource(normalize_visa_address(resource_name))
        except BaseException:
            self._manager.close()
            raise

    def __getattr__(self, name):
        # Expose VISA configuration (timeout, terminations, chunk_size) but do
        # not silently invent missing attributes, unlike a mock transport.
        return getattr(self._resource, name)

    def __setattr__(self, name, value):
        if name in ("_resource", "_manager"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._resource, name, value)

    def write(self, command):
        return self._resource.write(command)

    def query(self, command):
        return self._resource.query(command)

    def query_binary_values(self, command, **kwargs):
        return self._resource.query_binary_values(command, **kwargs)

    def close(self):
        try:
            self._resource.close()
        finally:
            self._manager.close()
