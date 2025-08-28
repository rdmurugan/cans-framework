"""Custom exceptions for the CANS framework"""


class CANSError(Exception):
    """Base exception class for CANS framework"""
    pass


class ConfigurationError(CANSError):
    """Raised when configuration is invalid"""
    pass


class ModelError(CANSError):
    """Raised when model-related errors occur"""
    pass


class DataError(CANSError):
    """Raised when data-related errors occur"""
    pass


class DimensionMismatchError(ModelError):
    """Raised when tensor dimensions don't match expectations"""
    pass


class ValidationError(CANSError):
    """Raised when validation fails"""
    pass


class TrainingError(CANSError):
    """Raised when training encounters errors"""
    pass


class InferenceError(CANSError):
    """Raised during inference/prediction"""
    pass