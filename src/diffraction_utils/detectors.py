from dataclasses import dataclass


@dataclass
class detector_pilatus_2022:
    name = "PILATUS"
    pixel_size = 172e-6
    image_shape = (195, 487)
    is_pilatus = True
    is_dectris = False
    is_excalibur = False

    def get_background_regions():
        return NotImplementedError()

    def get_signal_regions():
        return NotImplementedError()


@dataclass
class detector_pilatus_2021:
    name = "pil2roi"
    pixel_size = 172e-6
    image_shape = (1679, 1475)
    is_pilatus = True
    is_dectris = False
    is_excalibur = False

    def get_background_regions():
        return NotImplementedError()

    def get_signal_regions():
        return NotImplementedError()


@dataclass
class detector_pilatus_2_stats:
    name = "pil2stats"
    pixel_size = 172e-6
    image_shape = (1679, 1475)
    is_pilatus = True
    is_dectris = False
    is_excalibur = False

    def get_background_regions():
        return NotImplementedError()

    def get_signal_regions():
        return NotImplementedError()


@dataclass
class detector_pilatus_eh2_2022:
    name = "pil3roi"
    pixel_size = 172e-6
    image_shape = (195, 487)
    is_pilatus = True
    is_dectris = False
    is_excalibur = False

    def get_background_regions():
        return NotImplementedError()

    def get_signal_regions():
        return NotImplementedError()


@dataclass
class detector_pilatus_eh2_stats:
    name = "pil3stats"
    pixel_size = 172e-6
    image_shape = (195, 487)
    is_pilatus = True
    is_dectris = False
    is_excalibur = False

    def get_background_regions():
        return NotImplementedError()

    def get_signal_regions():
        return NotImplementedError()


@dataclass
class detector_pilatus_eh2_scan:
    name = "p3r"
    pixel_size = 172e-6
    image_shape = (195, 487)
    is_pilatus = True
    is_dectris = False
    is_excalibur = False

    def get_background_regions():
        return NotImplementedError()

    def get_signal_regions():
        return NotImplementedError()


@dataclass
class detector_p2r:
    name = "p2r"
    pixel_size = 172e-6
    image_shape = (1679, 1475)
    is_pilatus = True
    is_dectris = False
    is_excalibur = False

    def get_background_regions():
        return NotImplementedError()

    def get_signal_regions():
        return NotImplementedError()


@dataclass
class detector_excalibur_08_2023_stats:
    name = "excstats"
    pixel_size = 55e-6
    image_shape = (515, 2069)
    is_pilatus = False
    is_dectris = False
    is_excalibur = True

    def get_background_regions():
        return NotImplementedError()

    def get_signal_regions():
        return NotImplementedError()


@dataclass
class detector_excalibur_08_2023_roi:
    name = "excroi"
    pixel_size = 55e-6
    image_shape = (515, 2069)
    is_pilatus = False
    is_dectris = False
    is_excalibur = True

    def get_background_regions(i07nxs: I07Nexus):
        return NotImplementedError()

    def get_signal_regions():
        return NotImplementedError()


@dataclass
class detector_eiger_detector_01_2026:
    name = "eir"
    pixel_size = 55e-6
    image_shape = (2162, 2068)
    is_pilatus = False
    is_dectris = False
    is_excalibur = True

    def get_background_regions():
        return NotImplementedError()

    def get_signal_regions():
        return NotImplementedError()


@dataclass
class detector_excalibur_04_2022:
    name = "exr"
    pixel_size = 55e-6
    image_shape = (2162, 2068)
    is_pilatus = False
    is_dectris = False
    is_excalibur = True

    def get_background_regions():
        return NotImplementedError()

    def get_signal_regions():
        return NotImplementedError()


@dataclass
class detector_excalibur_2022_fscan:
    name = "EXCALIBUR"
    pixel_size = 55e-6
    image_shape = (2162, 2068)
    is_pilatus = False
    is_dectris = False
    is_excalibur = True

    def get_background_regions():
        return NotImplementedError()

    def get_signal_regions():
        return NotImplementedError()


# Setups.
horizontal = "horizontal"
vertical = "vertical"
dcd = "DCD"
