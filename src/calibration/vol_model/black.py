from .base import VolModel
from ..curve_model.curves import RateCurve
from ..surface import VolSurface

class blackscholes(VolModel):

    def calibrate(self, quotes, curves: RateCurve, vol_surface: VolSurface):
        return NotImplemented

