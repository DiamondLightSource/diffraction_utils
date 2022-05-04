"""
This module tests diffraction_utils' io module.
"""

# Of course we need this while testing.
# pylint: disable=protected-access

import pytest

from diffraction_utils.io import I07Nexus, I10Nexus, MissingMetadataWarning


def test_attributes_i07_2021(path_to_2021_i07_nxs: str):
    """
    Make sure that i07 2021 .nxs files can be parsed and have valid attributes.
    """
    i07nexus = I07Nexus(path_to_2021_i07_nxs)

    assert i07nexus.local_path == path_to_2021_i07_nxs
    assert i07nexus.probe_energy == 12.5e3
    assert i07nexus.entry == i07nexus.nxfile['/entry/']
    assert i07nexus.src_path == "/dls/i07/data/2021/si28707-1/i07-404875.nxs"
    assert i07nexus.instrument == i07nexus.nxfile['/entry/instrument/']
    assert i07nexus.detector == i07nexus.nxfile['/entry/instrument/excroi/']
    assert i07nexus.default_signal[0] == 342.7288888888887
    assert i07nexus.default_axis[0] == 0.010000170375764542
    assert i07nexus.default_signal_name == "Region_1_average"
    assert i07nexus.default_axis_name == "qdcd"
    assert i07nexus.default_nxdata_name == "excroi"
    assert i07nexus.default_nxdata.signal == "Region_1_average"
    assert i07nexus.default_axis_type == 'q'
    assert i07nexus.scan_length == 51


def test_excalibur_name():
    """
    Make sure that we're spelling the detector names properly!
    """
    assert I07Nexus.excalibur_detector_2021 == "excroi"
    assert I07Nexus.excalibur_04_2022 == "exr"


def test_attributes_i07_04_2022(path_to_04_2022_i07_nxs: str):
    """
    Make sure that .nxs files from i07 in April 2022 can be parsed.
    """
    i07nexus = I07Nexus(path_to_04_2022_i07_nxs)

    assert i07nexus.local_path == path_to_04_2022_i07_nxs
    assert i07nexus.probe_energy == 19.9e3
    assert i07nexus.entry == i07nexus.nxfile['/entry/']
    assert i07nexus.src_path == "/dls/i07/data/2022/si31166-1/i07-426911.nxs"
    assert i07nexus.instrument == i07nexus.nxfile['/entry/instrument/']
    assert i07nexus.detector == i07nexus.nxfile['/entry/instrument/exr/']
    assert i07nexus.default_signal[0] == 1.0
    assert i07nexus.default_axis[0] == 0.900000841
    assert i07nexus.default_signal_name == "frameNo"  # ?????? ??????
    assert i07nexus.default_axis_name == "diff1delta"
    assert i07nexus.default_nxdata_name == "exr"
    assert i07nexus.default_nxdata.signal == "frameNo"  # ?????? ??????
    assert i07nexus.default_axis_type == 'tth'
    assert i07nexus.scan_length == 13


def test_attributes_i10_2022(path_to_2022_i10_nxs):
    """
    Make sure that we can parse i10 .nxs files from 2022.
    """
    # We didn't set the detector distance.
    with pytest.warns(MissingMetadataWarning):
        i10nexus = I10Nexus(path_to_2022_i10_nxs)
    assert i10nexus.detector_distance is None

    # Check all the attributes that we should be able to get.
    assert i10nexus.local_path == path_to_2022_i10_nxs
    assert i10nexus.probe_energy == 931.7725  # Cu L3 edge.
    assert i10nexus.entry == i10nexus.nxfile['/entry/']
    assert i10nexus.src_path == "/dls/i10/data/2022/mm30383-1/i10-693862.nxs"
    assert i10nexus.instrument == i10nexus.nxfile['/entry/instrument/']
    assert i10nexus.detector == i10nexus.nxfile['/entry/instrument/pimtetiff/']
    assert i10nexus.default_signal[10] == \
        b"/dls/i10/data/2022/mm30383-1/693862-pimte-files/pimte-00010.tiff"
    assert i10nexus.default_axis[0] == 130.37158560119
    assert i10nexus.default_signal_name == "image_data"
    assert i10nexus.default_axis_name == "th"
    assert i10nexus.default_nxdata_name == "pimtetiff"
    assert i10nexus.default_nxdata.signal == "image_data"
    assert i10nexus.scan_length == 141


def test_motor_positions_i10_2022(path_to_2022_i10_nxs):
    """
    Make sure that we can read out raw and virtual motor positions from the i10
    nexus file.
    """
    # We didn't set the detector distance.
    with pytest.warns(MissingMetadataWarning):
        i10nexus = I10Nexus(path_to_2022_i10_nxs)

    assert i10nexus._motors["th"][0] == 130.37158560119
    assert i10nexus._motors["tth"][1] == -9.969088891715
    assert i10nexus._motors["chi"][2] == 88.9999998346
    assert i10nexus.theta[0] == 130.37158560119
    assert i10nexus.two_theta[1] == -9.969088891715
    assert i10nexus.theta_area[0] == 180 - 130.37158560119
    assert i10nexus.two_theta_area[1] == 90 - 9.969088891715
    assert i10nexus.chi[5] == 88.9999998346  # Chi wasn't scanned.


def test_i10_2022_detector_distance(path_to_2022_i10_nxs):
    """
    Make sure that we can set detector distance manually for i10 .nxs files.
    """
    i10nexus = I10Nexus(path_to_2022_i10_nxs, 10)

    assert i10nexus.detector_distance == 10
