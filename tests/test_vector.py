"""
This module provides tests for the diffraction_utils.vector module's Vector3
class.
"""

import numpy as np
from numpy.testing import assert_allclose

from diffraction_utils.vector import Vector3


def test_attrs():
    """
    I've always been fond of having a test that blows up when things change in
    a completely trivial way. This is that test.
    """
    vec = Vector3([0, 1, 2], 'lab')
    assert (vec.array == np.array([0, 1, 2])).all()
    assert vec.frame == 'lab'


def test_unit():
    """
    Make sure that we can properly make unit vectors.
    """
    vec = Vector3([0, 1, 1], '')
    desired = np.array([0, 1, 1])/np.sqrt(2)

    assert_allclose(vec.unit, desired)


def test_azimuthal_angle():
    """
    Make sure that we're correctly calculating azimuthal angles, where the
    azimuthal angle is the angle about the y-axis measured from the z-axis.
    """
    vec = Vector3([0, 1, 1], '')
    assert_allclose(vec.azimuthal_angle, 0)

    vec = Vector3([1, 1, 0], '')
    assert_allclose(vec.azimuthal_angle, np.pi/2)

    vec = Vector3([-1, 1, 0], '')
    assert_allclose(vec.azimuthal_angle, -np.pi/2)

    vec = Vector3([1, 0, 1], '')
    assert_allclose(vec.azimuthal_angle, np.pi/4)


def test_polar_angle():
    """
    Make sure that we're correctly calculating polar angles, where the polar
    angle is the angle between the vector and the y-axis.
    """
    vec = Vector3([0, 1, 1], '')
    assert_allclose(vec.polar_angle, np.pi/4)

    vec = Vector3([1, 1, 0], '')
    assert_allclose(vec.polar_angle, np.pi/4)

    vec = Vector3([-1, -1, 0], '')
    assert_allclose(vec.polar_angle, 3*np.pi/4)

    vec = Vector3([1, 0, 1], '')
    assert_allclose(vec.polar_angle, np.pi/2)


def test_from_angles():
    """
    Make sure that we can construct vectors from spherical polar angles.
    """
    vec = Vector3.from_angles(np.pi/2, np.pi/4, '', np.sqrt(2))
    assert_allclose(np.array([1, 1, 0]), vec.array, atol=1e-7)

    vec = Vector3.from_angles(np.pi/2, np.pi/4, '')
    assert_allclose(np.array([1, 1, 0])/np.sqrt(2), vec.array, atol=1e-7)

    vec = Vector3.from_angles(0, 0, '')
    assert_allclose(np.array([0, 1, 0]), vec.array, atol=1e-7)

    vec = Vector3.from_angles(-np.pi/2, 3*np.pi/4, '', length=np.sqrt(2))
    assert_allclose(np.array([-1, -1, 0]), vec.array, atol=1e-7)
