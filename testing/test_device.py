"""The device geometry lives in ``mxtreme.device``."""

from mxtreme import device


def test_geometry_constants_present():
    assert device.CHIP_WIDTH > 0
    assert device.CHIP_HEIGHT > 0
    assert device.ELEC_SIZE > 0
