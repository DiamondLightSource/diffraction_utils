"""
This module contains the DataFileBase class, returned by parser methods in the
islatu.io module. This class provides a consistent way to refer to metadata
returned by different detectors/instruments.
"""

from abc import abstractmethod, ABC

import numpy as np


class DataFileBase(ABC):
    """
    An ABC for classes that store metadata parsed from data files. This defines
    the properties that must be implemented by parsing classes.
    """

    def __init__(self, local_path):
        self.local_path = local_path

    @property
    @abstractmethod
    def probe_energy(self):
        """
        This must be overridden.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def default_axis(self) -> np.ndarray:
        """
        Returns a numpy array of data associated with the default axis, where
        "default axis" should be understood in the NeXus sense to mean the
        experiment's dependent variable.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def default_axis_name(self) -> str:
        """
        Returns the name of the default axis, as it was recorded in the data
        file stored at local_path.
        """
        raise NotImplementedError()

    @property
    def scan_length(self) -> int:
        """
        Returns the number of data points collected during this scan.
        """
        return len(self.default_axis)
