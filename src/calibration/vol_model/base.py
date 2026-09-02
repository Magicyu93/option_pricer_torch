from __future__ import annotations
import datetime as dt
from abc import ABC, abstractmethod

from src.calibration.surface import VolSurface
from ...calibration.curve_model.curves import RateCurve



class VolModel(ABC):
    """General vol model"""
    @abstractmethod
    def calibrate(self, quotes, curves: RateCurve, vol_surface: VolSurface):
        pass

    