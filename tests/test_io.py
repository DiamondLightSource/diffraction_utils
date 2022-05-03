"""
This module tests diffraction_utils' io module.
"""


from diffraction_utils.io import I07Nexus, I10Nexus


def test_attributes_i07_2021(path_to_2021_i07_nxs: str):
    """
    Make sure that i07 2021 .nxs files can be parsed and have valid attributes.
    """
    i07nexus = I07Nexus(path_to_2021_i07_nxs)

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


def test_attributes_i07_04_2022(path_to_04_2022_i07_nxs: str):
    """
    Make sure that .nxs files from i07 in April 2022 can be parsed.
    """
    i07nexus = I07Nexus(path_to_04_2022_i07_nxs)


def test_attributes_i10_2022(path_to_2022_i10_nxs):
    """
    Make sure that we can parse i10 .nxs files from 2022.
    """
    i10nexus = I10Nexus(path_to_2022_i10_nxs)
