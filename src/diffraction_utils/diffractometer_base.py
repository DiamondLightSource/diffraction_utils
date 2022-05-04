"""
This module contains the DiffractometerBase class.
"""

from abc import ABC, abstractmethod

from scipy.spatial.transform import Rotation

from .frame_of_reference import Frame
from .io import NexusBase
from .vector import Vector3


class DiffractometerBase(ABC):
    """
    This contains a generic description of what all diffractometers need to
    have.
    """

    def __init__(self, nexus: NexusBase) -> None:
        self.nexus = nexus

    @property
    @abstractmethod
    def u_matrix(self) -> Rotation:
        """
        The Rotation from of the so-called "U" rotation matrix.
        """

    @property
    @abstractmethod
    def b_matrix(self) -> Rotation:
        """
        The Rotation form of the so-called "B" rotation matrix.
        """

    @property
    @abstractmethod
    def ub_matrix(self) -> Rotation:
        """
        The Rotation form of the so-called "UB" rotation matrix.
        """

    @property
    @abstractmethod
    def to_frame(self, vector: Vector3, frame: Frame) -> None:
        """
        Rotates the vector passed as an argument into the frame specified by the
        frame argument.

        Args:
            vector:
                The vector to rotate.
            frame:
                The frame into which the vector will be rotated.
        """
