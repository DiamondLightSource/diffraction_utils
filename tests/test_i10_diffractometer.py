"""
This module contains tests for the I10RasorDiffractometer class. It does not
test anything covered by DiffractometerBase tests. TODO: run tests for the point
detector.
"""

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from diffraction_utils.diffractometers.diamond_i10 import I10RasorDiffractometer
from diffraction_utils.frame_of_reference import Frame


def test_u_matrix(rasor: I10RasorDiffractometer):
    """
    Make sure that we're properly calculating RASOR's U matrix. These values are
    calculated from a data file in which the area detector (pimtetiff) was used.
    """
    u_mat_0 = rasor.get_u_matrix(70)

    # Motor values read manually from .nxs file.
    theta = 49.6284 - 3.5/2
    chi = 1

    # These are absolutely the rotations required to go from the lab frame to
    # the sample frame, having done a very big think. Anyone that thinks
    # otherwise (namely: me in the future) can just fight me. (Note that I'm not
    # actually 100% on the chi, but who cares about chi on i10?)
    reverse_th_rot = Rotation.from_rotvec(np.array([1, 0, 0])*theta,
                                          degrees=True)
    reverse_chi_rot = Rotation.from_rotvec(np.array([0, 0, 1])*chi,
                                           degrees=True)
    random_vec = np.random.random(3)

    assert_allclose(random_vec,
                    (reverse_chi_rot*reverse_th_rot*u_mat_0).apply(random_vec),
                    rtol=1e-5)


def test_detector_vector(rasor: I10RasorDiffractometer):
    """
    Make sure that our detector vector is correct in the lab frame.
    """
    # tthArea on frame 70.
    tth_area = 96.519

    # Detector vector on frame 70 in the lab frame.
    frame = Frame(Frame.lab, rasor, 70)
    det_vec = rasor.get_detector_vector(frame)

    # To rotate it back to [0, 0, 1] we definitely need to apply the following:
    rot_back = Rotation.from_rotvec(np.array([1, 0, 0])*tth_area, degrees=True)

    assert_allclose(rot_back.apply(det_vec.array), [0, 0, 1], atol=1e-5)


def test_detector_vector_frame_change(rasor: I10RasorDiffractometer):
    """
    Make sure that our detector vector is being properly generated in different
    frames when asked. This test exploits the fact that vector reference frame
    changes are thoroughly tested elsewhere.
    """
    # We know that the lab frame det_vec is good, so get that.
    frame1 = Frame(Frame.lab, rasor, 70)
    det_vec_1_lab = rasor.get_detector_vector(frame1)

    # Now get an hkl frame det_vec.
    frame2 = Frame(Frame.hkl, rasor, 70)
    det_vec_2_hkl = rasor.get_detector_vector(frame2)

    assert not np.allclose(det_vec_1_lab.array, det_vec_2_hkl.array)

    # Rotate first det_vec into hkl frame.
    rasor.rotate_vector_to_frame(det_vec_1_lab, frame2)

    # Now they should be roughly equal.
    assert np.allclose(det_vec_1_lab.array, det_vec_2_hkl.array)
