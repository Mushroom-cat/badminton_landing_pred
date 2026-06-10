__all__ = ["CamCalib"]


def __getattr__(name):
    if name == "CamCalib":
        from .cam_calib import CamCalib
        return CamCalib
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
