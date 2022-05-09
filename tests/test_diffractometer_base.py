"""
Since the DiffractometerBase class is an ABC, we can't test the methods it
implements directly. Here we use the I10RasorDiffractometer class to test the
methods implemented in DiffractometerBase.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from diffraction_utils.diffractometers.diamond_i10 import I10RasorDiffractometer
from diffraction_utils.frame_of_reference import Frame
from diffraction_utils.vector import Vector3


def test_b_matrix_01(rasor: I10RasorDiffractometer):
    """
    Make sure that our B matrix doesn't do anything when l is OOP.

    TODO: update these tests when B-matrices can handle non-orthonormal bases.
    """
    vec = np.array([1, 0, 0])
    assert (rasor.get_b_matrix().apply(vec) == np.array([1, 0, 0])).all()


def test_b_matrix_02(rasor: I10RasorDiffractometer):
    """
    Make sure that our B matrix can handle some simple rotations for crystals
    whose l isn't OOP.
    """
    # Set the sample to be lying with h OOP; get the b matrix.
    rasor.sample_oop.array = np.array([1, 0, 0])
    b_rot = rasor.get_b_matrix()

    mapped_vec_1 = b_rot.apply([1, 0, 0])  # Should map to [0, 1, 0].
    mapped_vec_2 = b_rot.apply([0, 0, 1])  # Should map to itself.
    mapped_vec_3 = b_rot.apply([1, 1, 0])  # Should map to [1, -1, 0].
    assert_allclose(mapped_vec_1, [0, 1, 0], atol=1e-7)
    assert_allclose(mapped_vec_2, [0, 0, 1], atol=1e-7)
    assert_allclose(mapped_vec_3, [-1, 1, 0], atol=1e-7)


def test_rotate_vector_to_frame(rasor: I10RasorDiffractometer):
    """
    Test all of the cases that can be tested for a DiffractometerBase (that is,
    sample <-> hkl and frame_x <-> frame_x).
    """
    # Prepare a non-trivial B matrix and a vector to rotate.
    rasor.sample_oop.array = np.array([1, 0, 0])
    vec = Vector3([1, 0, 0], Frame(Frame.sample_holder, rasor))

    # Use rasor to rotate the vector into the hkl frame.
    rasor.rotate_vector_to_frame(vec, Frame(Frame.hkl, rasor))

    # In the sample frame, vec's array should be along -k.
    assert_allclose(vec.array, [0, -1, 0], atol=1e-7)

    # Make sure that our frame attribute has been affected.
    assert vec.frame == Frame(Frame.hkl, rasor)


def test_rotate_vector_to_frame_02(rasor: I10RasorDiffractometer):
    """
    Make sure that two subsequent attempts to rotate a vector to a frame have
    the same effect as a single rotation (the second should do nothing since
    we're already in that frame).
    """
    # Prepare a non-trivial B matrix and a vector to rotate.
    rasor.sample_oop.array = np.array([1, 0, 0])
    vec = Vector3([1, 0, 0], Frame(Frame.sample_holder, rasor))

    # Rotate twice; perform same assertions as in previous test.
    rasor.rotate_vector_to_frame(vec, Frame(Frame.hkl, rasor))
    rasor.rotate_vector_to_frame(vec, Frame(Frame.hkl, rasor))
    assert_allclose(vec.array, [0, -1, 0], atol=1e-7)
    assert vec.frame == Frame(Frame.hkl, rasor)


def test_rotate_vector_to_frame_03(rasor: I10RasorDiffractometer):
    """
    Make sure that a ValueError is raised if the frame's name is invalid.
    """
    # Prepare a non-trivial B matrix and a vector to rotate.
    rasor.sample_oop.array = np.array([1, 0, 0])
    vec = Vector3([1, 0, 0], Frame("bad frame name", rasor))

    with pytest.raises(ValueError):
        rasor.rotate_vector_to_frame(vec, Frame(Frame.hkl, rasor))


def test_get_incident_beam(rasor: I10RasorDiffractometer):
    """
    Make sure that our incident beam is being generated correctly.
    """
    sh_frame = Frame(Frame.sample_holder, rasor, 70)
    beam_on_sample = rasor.get_incident_beam(sh_frame)

    # Motor values read manually from .nxs file.
    theta = 49.6284 - 3.5/2
    chi = 1

    # Prepare to rotate the beam array back to the lab.
    rot = Rotation.from_rotvec(np.array([-1, 0, 0])*theta, degrees=True)
    rot *= Rotation.from_rotvec(np.array([0, 0, -1])*chi, degrees=True)

    assert_allclose(rot.apply(beam_on_sample.array), [0, 0, 1], atol=1e-6)
