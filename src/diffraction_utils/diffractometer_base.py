"""
This module contains the DiffractometerBase class.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation

from .data_file import DataFileBase
from .frame_of_reference import Frame
from .vector import Vector3, rot_from_a_to_b


class DiffractometerBase(ABC):
    """
    This contains a generic description of what all diffractometers need to
    have.
    """

    def __init__(self, data_file: DataFileBase, sample_oop: np.ndarray) -> None:
        self.data_file = data_file
        self.sample_oop = sample_oop
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

    def get_detector_rotation(self, frame: Frame) -> Rotation:
        """
        Returns an instance of Rotation that, when applied, maps the [0, 0, 1]
        vector to a vector that points at the detector's point of normal
        incidence in the lab frame.
        """
        # Work out what rotation has been applied to our detector.
        lab_frame = Frame(Frame.lab, frame.diffractometer, frame.scan_index)
        detector_vec_lab = self.get_detector_vector(lab_frame)
        unrotated_detector_lab = Vector3([0, 0, 1], lab_frame)
        rot = rot_from_a_to_b(unrotated_detector_lab, detector_vec_lab)

        # And finally, return it.
        return rot

    def get_detector_vertical(self, frame: Frame) -> Vector3:
        """
        Returns a unit vector that points vertically upwards on the detector.
        This is normal to the output of get_detector_vector and
        get_detector_horizontal. If all diffractometer motors are zeroed, this
        is parallel to the y-axis.

        Args:
            frame (Frame):
                An instance of Frame describing the frame in which we want a
                unit vector that points vertically upwards along the detector.

        Returns:
            An instance of Vector3 corresponding to a unit vector that points
            vertically upwards on the surface of the detector (parallel to the
            detector's slow axis).
        """
        rot = self.get_detector_rotation(frame)
        lab_frame = Frame(Frame.lab, frame.diffractometer, frame.scan_index)

        # Apply this to the y-axis in the lab frame to get the detector vertical
        # in the lab frame.
        detector_vert_lab_arr = rot.apply(np.array([0, 1, 0]))
        detector_vertical = Vector3(detector_vert_lab_arr, lab_frame)

        # Now put this in the correct frame of reference and return it.
        detector_vertical.to_frame(frame)
        return detector_vertical

    def get_detector_horizontal(self, frame: Frame) -> Vector3:
        """
        Returns a unit vector that points horizontally across the detector.
        This is normal to the output of get_detector_vector and
        get_detector_vertical. If all diffractometer motors are zeroed, this
        is parallel to the x-axis.

        Args:
            frame (Frame):
                An instance of Frame describing the frame in which we want a
                unit vector that points horizontally across the detector.

        Returns:
            An instance of Vector3 corresponding to a unit vector that points
            horizontally across the detector (parallel to the detector's fast
            axis).
        """
        rot = self.get_detector_rotation(frame)
        lab_frame = Frame(Frame.lab, frame.diffractometer, frame.scan_index)

        # Apply this to the y-axis in the lab frame to get the detector vertical
        # in the lab frame.
        detector_horiz_lab_arr = rot.apply(np.array([1, 0, 0]))
        detector_horizontal = Vector3(detector_horiz_lab_arr, lab_frame)

        # Now put this in the correct frame of reference and return it.
        detector_horizontal.to_frame(frame)
        return detector_horizontal

    def get_incident_beam(self, frame: Frame) -> Vector3:
        """
        Returns a unit vector that points in the direction of the incident beam
        in the frame given by the frame argument.

        Args:
            frame (Frame):
                An instance of Frame describing the frame of reference in which
                we want a unit vector pointing parallel to the incident beam.

        Returns:
            An instance of Vector3 corresponding to a unit vector that points
            parallel to the incident beam.
        """
        # In the lab frame, our coordinate system is defined such that the beam
        # is always travelling along [0, 0, 1] (unless beam is being bent by
        # e.g. a double crystal deflector (DCD) setup).
        lab_beam = Vector3([0, 0, 1], Frame(Frame.lab, self, frame.scan_index))

        # Now simply rotate the beam into the desired frame and return it!
        self.rotate_vector_to_frame(lab_beam, frame)
        return lab_beam

    @abstractmethod
    def get_diff_matrix(self, scan_index: int) -> Rotation:
        """
        This must be calculated in children of DiffractometerBase on a diffractometer-by-
        diffractometer basis.

        Args:
            scan_index:
                The diffractometer position varies throughout a scan. The scan_index
                parameter specified which step of the scan we want to generate
                a diffractometer matrix for.

        Returns:
            Instance of Rotation corresponding to the diffractometer matrix of interest.
        """

    def get_sample_matrix(self) -> Rotation:
        """
        This matrix maps vectors from the reciprocal lattice's hkl frame to a
        coordinate frame anchored to the sample holder.

        TODO: generalize so that this works for non-cubic crystals. This should
            be implemented by making Vector3's sentient of their basis vectors,
            and then modifying rot_from_a_to_b.

        Returns:
            Instance of Rotation corresponding to the B matrix for your sample.
        """
        # Generate a rotation from the sample_oop to the holder_oop
        holder_oop = Vector3([0, 1, 0], Frame(Frame.sample_holder, self))
        return rot_from_a_to_b(self.sample_oop, holder_oop)

    def get_diffsample_matrix(self, scan_index: int) -> Rotation:
        """
        A combination of the diffractometer and sample matrices.

        Args:
            scan_index:
                The diffractometer position varies throughout a scan. The scan_index
                parameter specified which step of the scan we want to generate
                a diffractometer matrix for.

        Returns:
            Instance of Rotation corresponding to the combined diffractometer and sample matrix for the
            scan_index of interest.
        """
        return self.get_diff_matrix(scan_index) * self.get_sample_matrix()

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

        # If the vector doesn't have a scan_index, default to to_frame's index.
        if vector.frame.scan_index is None:
            vector.frame.scan_index = to_frame.scan_index

        # Okay, we're changing frame. We have to handle each case individually.
        match vector.frame.frame_name, to_frame.frame_name:
            case Frame.lab, Frame.hkl:
                # To go from the lab to hkl we need the inverse of diffractometer matrix.
                # the inverse sample UB is applied in image.q_vectors when hkl is the output frame.
                rot = self.get_diffsample_matrix(vector.frame.scan_index).inv()
            case Frame.lab, Frame.qxqyqz:
                # To go from the lab to cartesian, on the sample surface,\
                #  but not matched to lattice parameters
                # the inverse sample U is applied in image.q_vectors when qxqyqz is the output frame
                rot = self.get_diffsample_matrix(vector.frame.scan_index).inv()
            case Frame.lab, Frame.sample_holder:
                # To go from the lab to the sample holder we need the inverse of diffractometer matrix.
                rot = self.get_diff_matrix(vector.frame.scan_index).inv()

            case Frame.sample_holder, Frame.lab:
                # We can use the diffractometer matrix to go from the sample holder to the lab.
                rot = self.get_diff_matrix(to_frame.scan_index)
            case Frame.sample_holder, Frame.hkl:
                # We can use B^-1 to go from the sample holder to hkl space.
                # the inverse sample UB is applied in image.q_vectors when hkl is the output frame.
                rot = self.get_sample_matrix().inv()

            case Frame.hkl, Frame.lab:
                # This has not been tested, to start from hkl the sample UB needs to be applied in image.q_vectors when hkl is the input frame.
                rot = self.get_diffsample_matrix(to_frame.scan_index)

            case Frame.hkl, Frame.sample_holder:
                # This has not been tested, to start from hkl the sample UB needs to be applied in image.q_vectors when hkl is the input frame.
                rot = self.get_sample_matrix()

            case _:
                # Invalid frame name, raise an error
                raise ValueError(
                    "Tried to rotate to or from a frame with an invalid name."
                )

        # Apply the rotation to the vector we were given.
        vector.array = rot.apply(vector.array)
        vector.frame = to_frame
