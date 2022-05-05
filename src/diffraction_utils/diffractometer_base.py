"""
This module contains the DiffractometerBase class.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation

from .frame_of_reference import Frame
from .data_file import DataFileBase
from .vector import Vector3, rot_from_a_to_b


class DiffractometerBase(ABC):
    """
    This contains a generic description of what all diffractometers need to
    have.
    """

    def __init__(self, data_file: DataFileBase, sample_oop: np.ndarray) -> None:
        self.data_file = data_file
        if not isinstance(sample_oop, Vector3):
            sample_oop = np.array(sample_oop)
            frame = Frame(Frame.sample_holder, self)
            self.sample_oop = Vector3(sample_oop, frame)

    @abstractmethod
    def get_detector_vector(self, frame: Frame) -> Vector3:
        """
        Returns a unit vector that points towards the detector in the frame
        given by the frame argument.

        Args:
            frame (Frame):
                An instance of Frame describing the frame in which we want a
                unit vector that points towards the detector.

        Returns:
            An instance of Vector3 corresponding to a unit vector that points
            towards the detector.
        """

    @abstractmethod
    def get_u_matrix(self, scan_index: int) -> Rotation:
        """
        The scipy Rotation from of the so-called "U" rotation matrix. This must
        be calculated in children of DiffractometerBase on a diffractometer-by-
        diffractometer basis.

        Args:
            scan_index:
                The U matrix generally varies throughout a scan. The scan_index
                parameter specified which step of the scan we want to generate
                a U matrix for.

        Returns:
            Instance of Rotation corresponding to the U matrix of interest.
        """

    def get_b_matrix(self) -> Rotation:
        """
        The scipy Rotation form of the so-called "B" rotation matrix. This
        matrix maps vectors from the reciprocal lattice's hkl frame to a
        coordinate frame anchored to the sample holder. This could be made a
        property, but is left as a method for symmetry with the U and UB
        matrices.

        TODO: generalize so that this works for non-cubic crystals. This should
            be implemented by making Vector3's sentient of their basis vectors,
            and then modifying rot_from_a_to_b.

        Returns:
            Instance of Rotation corresponding to the B matrix for your sample.
        """
        # Generate a rotation from the sample_oop to the holder_oop
        holder_oop = Vector3([0, 1, 0], Frame(Frame.sample_holder, self))
        return rot_from_a_to_b(self.sample_oop, holder_oop)

    def get_ub_matrix(self, scan_index: int) -> Rotation:
        """
        The scipy Rotation form of the so-called "UB" rotation matrix.

        Args:
            scan_index:
                The UB matrix generally varies throughout a scan, as the motion
                of the diffractometer motors affects the U matrix. The
                scan_index parameter specified which step of the scan we want to
                generate a U matrix (and therefore also the UB matrix) for.

        Returns:
            Instance of Rotation corresponding to the UB matrix for the
            scan_index of interest.
        """
        return self.get_u_matrix(scan_index) * self.get_b_matrix()

    def rotate_vector_to_frame(self, vector: Vector3, to_frame: Frame) -> None:
        """
        Rotates the vector passed as an argument into the frame specified by the
        frame argument.

        Args:
            vector:
                The vector to rotate.
            frame:
                The frame into which the vector will be rotated.
        """
        # Don't rotate if no rotation is required.
        if vector.frame.frame_name == to_frame.frame_name:
            return

        # Okay, we're changing frame. We have to handle each case individually.
        match vector.frame.frame_name, to_frame.frame_name:
            case Frame.lab, Frame.hkl:
                # To go from the lab to hkl we need the inverse of UB.
                rot = self.get_ub_matrix(vector.frame.scan_index).inv()
            case Frame.lab, Frame.sample_holder:
                # To go from the lab to the sample holder we just need U^-1.
                rot = self.get_u_matrix(vector.frame.scan_index).inv()

            case Frame.sample_holder, Frame.lab:
                # We can use U to go from the sample holder to the lab.
                rot = self.get_u_matrix(to_frame.scan_index)
            case Frame.sample_holder, Frame.hkl:
                # We can use B^-1 to go from the sample holder to hkl space.
                rot = self.get_b_matrix().inv()

            case Frame.hkl, Frame.lab:
                # This is precisely what the UB matrix is for!
                rot = self.get_ub_matrix(to_frame.scan_index)
            case Frame.hkl, Frame.sample_holder:
                # This is what defines the B matrix.
                rot = self.get_b_matrix()

            case _:
                # Invalid frame name, raise an error
                raise ValueError(
                    "Tried to rotate to or from a frame with an invalid name.")

        # Apply the rotation to the vector we were given.
        vector.array = rot.apply(vector.array)
        vector.frame = to_frame
