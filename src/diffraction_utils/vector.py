"""
This module stores the Vector class, which is a light wrapper around a numpy
array. It contains some convenience methods/attributes for dealing with
coordinate system changes.
"""

import numpy as np

from .diffractometer_base import DiffractometerBase
from .frame_of_reference import Frame


class Vector3:
    """
    This class is a light wrapper around a numpy array, with some convenience
    methods/attributes for dealing with coordinate system changes.
    """

    def __init__(self, array: np.ndarray, frame: Frame):
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

    def to_frame(self, frame: Frame, diffractometer: DiffractometerBase):
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

    @classmethod
    def from_angles(cls,
                    azimuth: float,
                    polar: float,
                    frame: Frame,
                    length=1.0):
        """
        Constructs a new Vector3 from an azimuthal angle, a polar angle and a
        frame of reference.

        Args:
            azimuth:
                The azimuthal angle of the vector to create.
            polar:
                The polar angle of the vector to create.
            frame:
                The frame of reference our new vector will be in.
            length:
                The length of the new vector. Defaults to 1.0.
        """
        array = length * np.array([
            np.sin(polar)*np.sin(azimuth),
            np.cos(polar),
            np.sin(polar)*np.cos(azimuth)
        ])
        return cls(array, frame)
