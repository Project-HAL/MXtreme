"""The device geometry lives in ``mxtreme.device`` and no longer in ``constants``."""

from mxtreme import device


def test_geometry_constants_present():
    assert device.CHIP_WIDTH > 0
    assert device.CHIP_HEIGHT > 0
    assert device.ELEC_SIZE > 0


def test_geometry_removed_from_constants():
    from mxtreme import constants
    for name in ("CHIP_WIDTH", "CHIP_HEIGHT", "ELEC_SIZE"):
        assert not hasattr(constants, name), f"{name} should have moved to mxtreme.device"
