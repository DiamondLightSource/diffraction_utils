"""
This module stores the Vector class, which is a light wrapper around a numpy
array. It contains some convenience methods/attributes for dealing with
coordinate system changes.
"""

import numpy as np

from .diffractometer_base import DiffractometerBase


class Vector3:
    """
    This class is a light wrapper around a numpy array, with some convenience
    methods/attributes for dealing with coordinate system changes.
    """

    def __init__(self, array: np.ndarray, frame: str):
        self.array = np.array(array)
        self.frame = frame

    @property
    def azimuthal_angle(self):
        """
        Returns this vector's azimuthal angle in its current reference frame.
        """
        return np.arctan2(self.array[0], self.array[2])

    @property
    def polar_angle(self):
        """
        Returns this vector's polar angle in its current reference frame.
        """
        return np.arccos(self.unit[1])

    @property
    def unit(self):
        """
        Returns the unit vector parallel to this Vector3.
        """
        return self.array/np.linalg.norm(self.array)

    def to_frame(self, frame: str, diffractometer: DiffractometerBase):
        """
        Transforms to a frame with name `frame`.

        Args:
            frame:
                The name of the frame of reference to transform to.
            detector:
                The diffractometer we should use to carry out the
                transformation.
        """
        raise NotImplementedError()
