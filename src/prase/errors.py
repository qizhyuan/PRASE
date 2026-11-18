"""Errors raised by PRASE."""


class PraseError(Exception):
    """Base class for PRASE errors."""


class DataError(PraseError, ValueError):
    """An input graph, link file, or saved output is invalid."""


class ConfigError(PraseError, ValueError):
    """A setting, or a combination of settings, is invalid."""
