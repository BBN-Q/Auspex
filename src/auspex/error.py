class AuspexError(Exception):
    """Base class for auspex specific errors"""
    pass

class DatafileError(AuspexError):
    """Error involving auspex data format"""
    pass

class InstrumentError(AuspexError):
    """Error involving instrument"""
    pass

class FakeDataError(InstrumentError):
    """Error from bad config of fake data interface"""
    pass

class InstrumentConstructionError(InstrumentError):
    """Error in creating the auspex experiment class."""
    pass

class DigitizerError(InstrumentError):
    """Error involving digitizer"""
    pass

class PipelineError(AuspexError):
    """Error occuring in pipeline processing"""
    pass

class ChannelLibraryError(AuspexError):
    """Error occuring in channel library error"""
    pass

class PlottingError(AuspexError):
    """Error occuring in plot assembly or rendering"""
    pass

class CalibrationError(AuspexError):
    """Error occuring during calibration"""
    pass

class DatabaseError(AuspexError):
    """Error occuring with database access"""
    pass