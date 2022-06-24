"""
This module contains:

Parsing functions used to extract information from experimental files.

Classes used to help make parsing more modular. These include the NexusBase
class and its children.
"""

# We've gotta access the _value attribute on some NXobjects.
# pylint: disable=protected-access

# Some useless super delegations are useful for quick abstractmethod overrides.
# pylint: disable=useless-super-delegation


import json
from abc import abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Union
from warnings import warn


import nexusformat.nexus.tree as nx
import numpy as np
from nexusformat.nexus import nxload


from .region import Region
from .data_file import DataFileBase


BAD_NEXUS_FILE = (
    "Nexus files suck. It turns out your nexus file sucked too. "
    "If you're seeing this message, it means some non-essential data couldn't "
    "be parsed by diffraction_utils.")


class MissingMetadataWarning(UserWarning):
    """
    Warns a user that some metadata is missing.
    """


class NexusBase(DataFileBase):
    """
    This class contains *mostly* beamline agnostic nexus parsing convenience
    stuff. It's worth noting that this class still makes a series of assumptions
    about how data is laid out in a nexus file that can be broken. Instead of
    striving for some impossible perfection, this class is practical in its
    assumptions of how data is laid out in a .nxs file, and will raise if an
    assumption is violated. All instrument-specific assumptions that one must
    inevitably make to extract truly meaningful information from a nexus file
    are made in children of this class.

    Attrs:
        file_path:
            The local path to the file on the local filesystem.
        nxfile:
            The object produced by loading the file at file_path with nxload.
    """

    def __init__(self,
                 local_path: Union[str, Path],  # The path to this file.
                 local_data_path: Union[str, Path] = '',  # Path to the data.
                 locate_local_data=True):

        # Set up the nexus specific attributes.
        # This needs to be done *before* calling super().__init__!
        self.nxfile = nxload(local_path)
        self.nx_entry = self._parse_nx_entry()
        self.default_nx_data_name = self._parse_default_nx_data_name()
        self.default_nx_data = self._parse_default_nx_data()
        self.nx_instrument = self._parse_nx_instrument()
        self.nx_detector = self._parse_nx_detector()

        # Now we can call super().__init__ to run the remaining parsers.
        super().__init__(local_path, local_data_path, locate_local_data)

        # Finally, parse the motors.
        self.motors = self._parse_motors()

    def _parse_nx_detector(self):
        """
        Returns the NXdetector instance stored in this NexusFile. This will
        need to be overridden for beamlines that put more than 1 NXdetector in
        a nexus file.

        Raises:
            ValueError if more than one NXdetector is found.
        """
        det, = self.nx_instrument.NXdetector
        return det

    def _parse_nx_instrument(self):
        """
        Returns the NXinstrument instanced stored in this NexusFile.

        Raises:
            ValueError if more than one NXinstrument is found.
        """
        instrument, = self.nx_entry.NXinstrument
        return instrument

    def _parse_nx_entry(self) -> nx.NXentry:
        """
        Returns this nexusfile's entry.

        Raises:
            ValueError if more than one entry is found.
        """
        entry, = self.nxfile.NXentry
        return entry

    def _parse_default_signal(self) -> np.ndarray:
        """
        The numpy array of intensities pointed to by the signal attribute in the
        nexus file.
        """
        # pylint: disable=bare-except
        try:
            return self.default_nx_data[self.default_signal_name].nxdata
        except:
            return BAD_NEXUS_FILE

    def _parse_default_axis(self) -> np.ndarray:
        """
        Returns the nxdata associated with the default axis.
        """
        # pylint: disable=bare-except
        try:
            return self.default_nx_data[self.default_axis_name].nxdata
        except:
            return BAD_NEXUS_FILE

    def _parse_default_signal_name(self):
        """
        Returns the name of the default signal.
        """
        # pylint: disable=bare-except
        try:
            return self.default_nx_data.signal
        except:
            return BAD_NEXUS_FILE

    def _parse_default_axis_name(self) -> str:
        """
        Returns the name of the default axis.
        """
        # pylint: disable=bare-except
        try:
            return self.nx_entry[self.nx_entry.default].axes
        except:
            return BAD_NEXUS_FILE

    def _parse_default_nx_data_name(self):
        """
        Returns the name of the default nxdata.
        """
        # pylint: disable=bare-except
        try:
            return self.nx_entry.default
        except:
            return BAD_NEXUS_FILE

    def _parse_default_nx_data(self) -> np.ndarray:
        """
        Returns the default NXdata.
        """
        # pylint: disable=bare-except
        try:
            return self.nx_entry[self.default_nx_data_name]
        except:
            return BAD_NEXUS_FILE

    @abstractmethod
    def _parse_motors(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary taking the form:

            motor_name: motor_values

        where motor_values is an array containing the value of the motor with
        name motor_name at every point in the scan (even if the motor's value
        is unchanging).
        """


class I07Nexus(NexusBase):
    """
    This class extends NexusBase with methods useful for scraping information
    from nexus files produced at the I07 beamline at Diamond.
    """
    # Detectors.
    excalibur_detector_2021 = "excroi"
    excalibur_04_2022 = "exr"
    excalibur_2022_fscan = "EXCALIBUR"
    pilatus_2021 = "pil2roi"
    pilatus_2022 = "PILATUS"

    # Setups.
    horizontal = "horizontal"
    vertical = "vertical"
    dcd = "DCD"

    def __init__(self,
                 local_path: Union[str, Path],
                 local_data_path: Union[str, Path] = '',
                 detector_distance=None,
                 setup: str = 'horizontal',
                 diff_1=True,
                 locate_local_data=True):
        # We need to know what detector we're using before doing any further
        # initialization.
        self.nxfile = nxload(local_path)
        self.nx_entry = self._parse_nx_entry()
        self.detector_name = self._parse_detector_name()

        # Now we can call super().__init__
        super().__init__(local_path, local_data_path, locate_local_data)

        # Only a subset of i07's capabilities can be handled by this library.
        if not diff_1:
            raise NotImplementedError(
                "Diffractometer 2 has not been implemented.")
        if setup == I07Nexus.dcd:
            raise NotImplementedError(
                "DCD nexus parsing has not been implemented.")
        if setup != I07Nexus.horizontal:
            raise NotImplementedError(
                "Only horizontal sample stage has been implemented.")

        # Parse the various i07-specific stuff.
        self.detector_distance = detector_distance
        self.transmission = self._parse_transmission()
        self.delta = self._parse_delta()
        self.gamma = self._parse_gamma()
        self.omega = self._parse_omega()
        self.theta = self._parse_theta()
        self.alpha = self._parse_alpha()
        self.chi = self._parse_chi()

        # Get the UB and U matrices, if they have been stored.
        self.ub_matrix = self._parse_ub()
        self.u_matrix = self._parse_u()

        # ROIs currently only implemented for the excalibur detector.
        if self.is_excalibur:
            self.signal_regions = self._parse_signal_regions()

    @property
    def has_image_data(self) -> bool:
        """
        It would be pretty weird to not have image data on an i07 nexus file.
        """
        return True

    @property
    def has_hdf5_data(self) -> bool:
        """
        Currently seems like a reasonable way of determining this.
        """
        # If something goes seriously wrong while checking if the file has hdf5
        # data, it probably doesn't! So, we use a broad except in this case.
        # pylint: disable=broad-except
        try:
            # Try to see if our detector's data points at an h5 file.
            if isinstance(self.nx_detector["data"], nx.NXlink):
                if self.nx_detector["data"]._filename.endswith('.h5'):
                    return True
        except Exception:
            # If something went really wrong, there mustn't be .h5 data.
            return False
        return False

    def _parse_hdf5_internal_path(self) -> str:
        """
        This needs to be implemented properly, as i07 scans *can* have data
        stored in .h5 files.
        """
        return self.nx_detector["data"]._target

    def _parse_raw_hdf5_path(self) -> Union[str, Path]:
        """
        This needs to be implemented properly, as i07 scans *can* have data
        stored in .h5 files.
        """
        return self.nx_detector["data"]._filename

    def _parse_probe_energy(self):
        """
        Returns the energy of the probe particle parsed from this NexusFile.
        """
        return float(self.nx_instrument.dcm1energy.value)*1e3

    def _parse_pixel_size(self) -> float:
        """
        Returns the side length of pixels in the detector that's being used.
        """
        if self.is_excalibur:
            return 55e-6
        if self.is_pilatus:
            return 172e-6
        raise ValueError(f"Detector name {self.detector_name} is unknown.")

    def _parse_image_shape(self) -> float:
        """
        Returns the shape of the images we expect to be recorded by this
        detector.
        """
        if self.is_excalibur:
            return 515, 2069
        if self.is_pilatus:
            return 1679, 1475
        raise ValueError(f"Detector name {self.detector_name} is unknown.")

    def _parse_raw_image_paths(self):
        """
        Returns the raw path to the data file. This is useless if you aren't on
        site, but used to guess where you've stored the data file locally.
        """
        if self.is_pilatus:
            path_array = self.nx_detector["image_data"]._value
        if self.is_excalibur:
            path_array = [
                self.nx_instrument["excalibur_h5_data/exc_path"]._value]

        return [x.decode('utf-8') for x in path_array]

    def _parse_nx_detector(self):
        """
        This override is necessary because some i07 .nxs files have multiple
        NXdetectors in their nexus files. What we really want is the appropriate
        camera, which we can parse exploiting the fact that we work out what
        the detector name is elsewhere.
        """
        return self.nx_instrument[self.detector_name]

    def _parse_motors(self) -> Dict[str, np.ndarray]:
        """
        A dictionary of all of the motor positions. This is only useful if you
        know some diffractometer specific keys, so it's kept private to
        encourage users to directly access the cleaner theta, two_theta etc.
        properties.
        """
        instr_motor_names = ["diff1delta", "diff1gamma", "diff1omega",
                             "diff1theta", "diff1alpha", "diff1chi"]

        motors_dict = {}
        ones = np.ones(self.scan_length)
        for name in instr_motor_names:
            # This could be a link to the data, a single value or a numpy array
            # containing varying values. We need to handle all three cases. The
            # last two cases are handled by multiplying by an array of ones.
            if "value" in dir(self.nx_instrument[name]):
                motors_dict[name] = self.nx_instrument[name].value._value*ones
            if "value_set" in dir(self.nx_instrument[name]):
                motors_dict[name] = \
                    self.nx_instrument[name].value_set.nxlink._value

        return motors_dict

    def _parse_transmission(self):
        """
        Proportional to the fraction of probe particles allowed by an attenuator
        to strike the sample.
        """
        return float(self.nx_instrument.filterset.transmission)

    def _parse_delta(self) -> np.ndarray:
        """
        Returns a numpy array of the delta values throughout the scan.
        """
        return self.motors["diff1delta"]

    def _parse_gamma(self) -> np.ndarray:
        """
        Returns a numpy array of the gamma values throughout the scan.
        """
        return self.motors["diff1gamma"]

    def _parse_omega(self) -> np.ndarray:
        """
        Returns a numpy array of the omega values throughout the scan.
        """
        return self.motors["diff1omega"]

    def _parse_theta(self) -> np.ndarray:
        """
        Returns a numpy array of the theta values throughout the scan.
        """
        return self.motors["diff1theta"]

    def _parse_alpha(self) -> np.ndarray:
        """
        Returns a numpy array of the alpha values throughout the scan.
        """
        return self.motors["diff1alpha"]

    def _parse_chi(self) -> np.ndarray:
        """
        Returns a numpy array of the chi values throughout the scan.
        """
        return self.motors["diff1chi"]

    def _parse_detector_name(self) -> str:
        """
        Returns the name of the detector that we're using. Because life sucks,
        this is a function of time.
        """
        if "excroi" in self.nx_entry:
            return I07Nexus.excalibur_detector_2021
        if "exr" in self.nx_entry:
            return I07Nexus.excalibur_04_2022
        if "pil2roi" in self.nx_entry:
            return I07Nexus.pilatus_2021
        if "PILATUS" in self.nx_entry:
            return I07Nexus.pilatus_2022
        if "EXCALIBUR" in self.nx_entry:
            return I07Nexus.excalibur_2022_fscan

        # Couldn't recognise the detector.
        raise NotImplementedError("Couldn't recognise detector name.")

    def _parse_signal_regions(self) -> List[Region]:
        """
        Returns a list of region objects that define the location of the signal.
        Currently there is nothing better to do than assume that this is a list
        of length 1.
        """
        if self.detector_name == I07Nexus.excalibur_detector_2021:
            return [self._get_ith_region(i=1)]
        if self.detector_name == I07Nexus.excalibur_04_2022:
            # Make sure our code executes for bytes and strings.
            try:
                json_str = self.nx_instrument[
                    "ex_rois/excalibur_ROIs"]._value.decode("utf-8")
            except AttributeError:
                json_str = self.nx_instrument["ex_rois/excalibur_ROIs"]._value
            # This is badly formatted and cant be loaded by the json lib. We
            # need to make a series of modifications.
            json_str = json_str.replace('u', '')
            json_str = json_str.replace("'", '"')
            json_str = json_str.replace('x', '"x"')
            json_str = json_str.replace('y', '"y"')
            json_str = json_str.replace('width', '"width"')
            json_str = json_str.replace('height', '"height"')
            json_str = json_str.replace('angle', '"angle"')

            roi_dict = json.loads(json_str)
            return [Region.from_dict(roi_dict['Region_1'])]

        raise NotImplementedError()

    def _get_ith_region(self, i: int):
        """
        Returns the ith region of interest found in the .nxs file.

        Args:
            i:
                The region of interest number to return. This number should
                match the ROI name as found in the .nxs file (generally not 0
                indexed).

        Returns:
            The ith region of interest found in the .nxs file.
        """
        x_1 = self.nx_detector[self._get_region_bounds_key(i, 'x_1')][0]
        x_2 = self.nx_detector[self._get_region_bounds_key(
            i, 'Width')][0] + x_1
        y_1 = self.nx_detector[self._get_region_bounds_key(i, 'y_1')][0]
        y_2 = self.nx_detector[self._get_region_bounds_key(
            i, 'Height')][0] + y_1
        return Region(x_1, x_2, y_1, y_2)

    @property
    def background_regions(self) -> List[Region]:
        """
        Returns a list of region objects that define the location of background.
        Currently we just ignore the zeroth region and call the rest of them
        background regions.
        """
        if self.detector_name == I07Nexus.excalibur_detector_2021:
            return [self._get_ith_region(i)
                    for i in range(2, self._number_of_regions+1)]
        if self.detector_name == I07Nexus.excalibur_04_2022:
            # Make sure our code executes for bytes and strings.
            try:
                json_str = self.nx_instrument[
                    "ex_rois/excalibur_ROIs"]._value.decode("utf-8")
            except AttributeError:
                json_str = self.nx_instrument[
                    "ex_rois/excalibur_ROIs"]._value
            # This is badly formatted and cant be loaded by the json lib. We
            # need to make a series of modifications.
            json_str = json_str.replace('u', '')
            json_str = json_str.replace("'", '"')
            json_str = json_str.replace('x', '"x"')
            json_str = json_str.replace('y', '"y"')
            json_str = json_str.replace('width', '"width"')
            json_str = json_str.replace('height', '"height"')
            json_str = json_str.replace('angle', '"angle"')

            roi_dict = json.loads(json_str)
            bkg_roi_list = list(roi_dict.values())[1:]
            return [Region.from_dict(x) for x in bkg_roi_list]

        raise NotImplementedError()

    @property
    def _region_keys(self) -> List[str]:
        """
        Parses all of the detector's dictionary keys and returns all keys
        relating to regions of interest.
        """
        return [key for key in self.nx_detector.keys()
                if key.startswith("Region")]

    @property
    def _number_of_regions(self) -> int:
        """
        Returns the number of regions of interest described by this nexus file.
        This *assumes* that the region keys take the form f'region_{an_int}'.
        """
        split_keys = [key.split('_') for key in self._region_keys]

        return max([int(split_key[1]) for split_key in split_keys])

    def _get_region_bounds_key(self, region_no: int, kind: str) -> List[str]:
        """
        Returns the detector key relating to the bounds of the region of
        interest corresponding to region_no.

        Args:
            region_no:
                An integer corresponding the the particular region of interest
                we're interested in generating a key for.
            kind:
                The kind of region bounds keys we're interested in. This can
                take the values:
                    'x_1', 'width', 'y_1', 'height'
                where '1' can be replaced with 'start' and with/without caps on
                first letter of width/height.

        Raises:
            ValueError if 'kind' argument is not one of the above.

        Returns:
            A list of region bounds keys that is ordered by region number.
        """
        # Note that the x, y swapping is a quirk of the nexus standard, and is
        # related to which axis on the detector varies most rapidly in memory.
        if kind in ('x_1', 'x_start'):
            insert = 'X'
        elif kind in ('width', 'Width'):
            insert = 'Width'
        elif kind in ('y_1', 'y_start'):
            insert = 'Y'
        elif kind in ('height', 'Height'):
            insert = 'Height'
        else:
            raise ValueError("Didn't recognise 'kind' argument.")

        return f"Region_{region_no}_{insert}"

    @property
    def is_excalibur(self) -> bool:
        """
        Returns whether or not we're currently using the excalibur detector.
        """
        return self.detector_name in ['excroi', 'exr', 'EXCALIBUR']

    @property
    def is_pilatus(self) -> bool:
        """
        Returns whether or not we're currently using the pilatus detector.
        """
        return self.detector_name in [I07Nexus.pilatus_2021,
                                      I07Nexus.pilatus_2022]

    def _parse_u(self) -> np.ndarray:
        """
        Parses the UB matrix from a .nxs file, if it has been stored. If it
        hasn't, returns None.
        """
        # This quantity has only been determined for pilatus_2022 .nxs files.
        if self.detector_name == self.pilatus_2022:
            return self.nx_instrument["diffcalchdr.diffcalc_u"].value.nxdata

    def _parse_ub(self) -> np.ndarray:
        """
        Parses the UB matrix from a .nxs file, if it has been stored. If it
        hasn't, returns None.
        """
        # This quantity has only been determined for pilatus_2022 .nxs files.
        if self.detector_name == self.pilatus_2022:
            return self.nx_instrument["diffcalchdr.diffcalc_ub"].value.nxdata


class I10Nexus(NexusBase):
    """
    This class extends NexusBase with methods useful for scraping information
    from nexus files produced at the I10 beamline at Diamond.
    """

    # We might need to check which instrument we're using at some point.
    rasor_instrument = "rasor"

    def __init__(self,
                 local_path: Union[str, Path],
                 local_data_path: Union[str, Path] = '',
                 detector_distance: float = None,
                 locate_local_data: bool = True):
        super().__init__(local_path, local_data_path, locate_local_data)

        # Warn the user if detector distance hasn't been set.
        if detector_distance is None:
            warn(MissingMetadataWarning(
                "Detector distance has not been set. At I10, sample-detector "
                "distance is not recorded in the nexus file, and must be "
                "input manually when using this library if it is needed."))

        # Initialize the i10 specific stuff.
        self.detector_distance = detector_distance
        self.theta = self._parse_theta()
        self.theta_area = self._parse_theta_area()
        self.two_theta = self._parse_two_theta()
        self.two_theta_area = self._parse_two_theta_area()
        self.chi = self._parse_chi()

    @property
    def has_image_data(self) -> bool:
        """For now, assume all i10 data we're given is image data."""
        return True

    @property
    def has_hdf5_data(self) -> bool:
        """As of 31/05/2022, i10 does not output hdf5 data, only .tiffs."""
        return False

    def _parse_hdf5_internal_path(self) -> str:
        """Trivially raises, but we need to implement the abstractmethod"""
        return super()._parse_hdf5_internal_path()

    def _parse_raw_hdf5_path(self) -> Union[str, Path]:
        """Trivially raises, but we need to implement the abstractmethod"""
        return super()._parse_raw_hdf5_path()

    def _parse_raw_image_paths(self) -> List[str]:
        """
        Returns a list of paths to the .tiff images recorded during this scan.
        These are the same paths that were originally recorded during the scan,
        so will point at some directory in the diamond filesystem.
        """
        return [x.decode('utf-8') for x in self.default_signal]

    def _parse_probe_energy(self):
        """
        Returns the energy of the probe particle parsed from this NexusFile.
        """
        return float(self.nx_instrument.pgm.energy)

    def _parse_pixel_size(self) -> float:
        """
        All detectors on I10 have 13.5 micron pixels.
        """
        return 13.5e-6

    def _parse_image_shape(self) -> Tuple[int]:
        """
        Returns the shape of detector images. This is easy in I10, since they're
        both 2048**2 square detectors.
        """
        return 2048, 2048

    def _parse_motors(self) -> Dict[str, np.ndarray]:
        """
        A dictionary of all of the motor positions. This is only useful if you
        know some diffractometer specific keys, so it's kept private to
        encourage users to directly access the cleaner theta, two_theta etc.
        properties.
        """
        instr_motor_names = ["th", "tth", "chi"]
        diff_motor_names = ["theta", "2_theta", "chi"]

        motors_dict = {
            x: np.ones(self.scan_length) *
            self.nx_instrument.rasor.diff[y]._value
            for x, y in zip(instr_motor_names, diff_motor_names)}

        for name in instr_motor_names:
            try:
                motors_dict[name] = self.nx_instrument[name].value._value
            except KeyError:
                pass
        return motors_dict

    def _parse_theta(self) -> np.ndarray:
        """
        Returns the current theta value of the diffractometer, as parsed from
        the nexus file. Note that this will be different to thArea in GDA.
        """
        return self.motors["th"]

    def _parse_two_theta(self) -> np.ndarray:
        """
        Returns the current two-theta value of the diffractometer, as parsed
        from the nexus file. Note that this will be different to tthArea in GDA.
        """
        return self.motors["tth"]

    def _parse_theta_area(self) -> np.ndarray:
        """
        Returns the values of the thArea virtual motor during this scan.
        """
        return 180 - self.theta

    def _parse_two_theta_area(self) -> np.ndarray:
        """
        Returns the values of the tthArea virtual motor during this scan.
        """
        return 90 - self.two_theta

    def _parse_chi(self) -> np.ndarray:
        """
        Returns the current chi value of the diffractometer.
        """
        return 90 - self.motors["chi"]
