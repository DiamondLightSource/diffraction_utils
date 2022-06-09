"""
This module contains an implementation of

diffraction_utils.diffractometer_base.DiffractometerBase

for the diffractometer in I07's experimental hutch 1.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from ..diffractometer_base import DiffractometerBase
from ..frame_of_reference import Frame
from ..io import I07Nexus
from ..vector import Vector3


class I07Diffractometer(DiffractometerBase):
    """
    Implementation of DiffractometerBase for the diffractometer in Diamond's I07
    beamline's experimental hutch 1.

    Args:
        data_file (I10Nexus):
            An instance of diffraction_utils.io.I10Nexus corresponding to the
            nexus file that contains the diffractometer description.
        sample_oop (np.ndarray-like):
            A [h, k, l] array describing the sample's OOP vector in hkl-space.
    """

    horizontal = "horizontal"
    vertical = "vertical"
    dcd = "DCD"

    def __init__(self, data_file: I07Nexus, sample_oop: np.ndarray,
                 setup: str = "horizontal") -> None:
        super().__init__(data_file, sample_oop)
        self.setup = setup
        if self.setup == I07Diffractometer.dcd:
            raise NotImplementedError("DCD not yet implemented.")
        if self.setup == I07Diffractometer.vertical:
            raise NotImplementedError("Vertical geometry not yet supported.")

    def get_u_matrix(self, scan_index: int) -> Rotation:
        # The following are the axis in the lab frame when all motors are @0.
        # Note that omega is like theta but for the vertical axis (I think!)
        alpha_axis = np.array([0, 1, 0])
        chi_axis = np.array([1, 0, 0])
        theta_axis = np.array([0, 1, 0])

        if self.setup != I07Diffractometer.horizontal:
            raise NotImplementedError("Only horizontal setup is supported.")
        if self.setup == I07Diffractometer.horizontal:
            alpha = self.data_file.alpha[scan_index]
            chi = self.data_file.chi[scan_index]
            theta = self.data_file.theta[scan_index]

        # Create the rotation objects.
        alpha_rot = Rotation.from_rotvec(alpha_axis*alpha, degrees=True)
        chi_rot = Rotation.from_rotvec(chi_axis*chi, degrees=True)
        theta_rot = Rotation.from_rotvec(theta_axis*theta, degrees=True)

        # Alpha acts after chi, which acts after theta. So, apply rotations
        # in the correct order to get the U matrix:
        return alpha_rot*chi_rot*theta_rot

    def get_detector_vector(self, frame: Frame) -> Vector3:
       # The following are the axis in the lab frame when all motors are @0.
        gamma_axis = np.array([0, 1, 0])
        delta_axis = np.array([1, 0, 0])

        if self.setup == I07Diffractometer.dcd:
            raise NotImplementedError("DCD setup is not yet supported.")
        else:
            gamma = self.data_file.gamma[frame.scan_index]
            delta = self.data_file.delta[frame.scan_index]

        # Create the rotation objects.
        gamma_rot = Rotation.from_rotvec(gamma_axis*gamma, degrees=True)
        delta_rot = Rotation.from_rotvec(delta_axis*delta, degrees=True)

        # Combine them (gamma acts after delta).
        total_rot = gamma_rot * delta_rot

        # Act this rotation on the beam with the beam in the lab frame.
        beam_direction = np.array([0, 0, 1])
        detector_vec = Vector3(total_rot.apply(beam_direction),
                               Frame(Frame.lab, self, frame.scan_index))

        # Finally, rotate this vector into the frame that we need it in.
        self.rotate_vector_to_frame(detector_vec, frame)
        return detector_vec
