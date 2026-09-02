### Ingests raw market quotes and produces a clean, arbitrage-free implied-vol surface

from abc import ABC, abstractmethod
import datetime as dt

class VolSurface(ABC):

    @abstractmethod
    def vol(self, strike: float, expiry: float) -> float:
        pass

    @property
    @abstractmethod
    def reference_date(self) -> dt.date:
        ...


class FlatVolSurface(VolSurface):
    """a simple flat surface"""
    
    def __init__(self, vol: float, as_of: dt.date):
        self._vol = vol
        self._ref = as_of

    def vol(self, strike: float, expiry: float) -> float:
        return self._vol

    def reference_date(self) -> dt.date:
        return self._ref



class SVISurface(VolSurface):

    def vol(self, strike: float, expiry: float) -> float:
        pass

