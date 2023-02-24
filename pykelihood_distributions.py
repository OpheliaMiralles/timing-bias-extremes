from scipy.stats import genpareto, genextreme

from pykelihood.distributions import ScipyDistribution
from pykelihood.parameters import ConstantParameter
from pykelihood.utils import ifnone
import numpy as np

class GEV_reparametrized(ScipyDistribution):
    params_names = ("loc", "r", "shape", "p")
    base_module = genextreme

    def __init__(self, loc=0.0, r=1.0, shape=0.0, p=1 / 200):
        super(GEV_reparametrized, self).__init__(loc, r, shape, ConstantParameter(p))

    def _to_scipy_args(self, loc=None, r=None, shape=None):
        loc = ifnone(loc, self.loc())
        shape = ifnone(shape, self.shape())
        r = ifnone(r, self.r())
        p = float(self.p())
        scale = (r - loc) * shape / ((-np.log(1 - p)) ** (-shape) - 1) if shape != 0 else (r - loc) / (-(np.log(-np.log(1 - p))))
        if shape is not None:
            shape = -shape
        return {
            "c": ifnone(shape, -self.shape()),
            "loc": loc,
            "scale": ifnone(scale, scale),
        }


class GEV_reparametrized_loc(ScipyDistribution):
    params_names = ("r", "scale", "shape", "p")
    base_module = genextreme

    def __init__(self, r=0.0, scale=1.0, shape=0.0, p=1 / 200):
        super(GEV_reparametrized_loc, self).__init__(r, scale, shape, ConstantParameter(p))

    def _to_scipy_args(self, r=None, scale=None, shape=None):
        scale = ifnone(scale, self.scale())
        shape = ifnone(shape, self.shape())
        r = ifnone(r, self.r())
        p = float(self.p())
        loc = r - (scale / shape) * ((-np.log(1 - p)) ** (-shape) - 1) if shape != 0 else r - scale * (-(np.log(-np.log(1 - p))))
        if shape is not None:
            shape = -shape
        return {
            "c": ifnone(shape, -self.shape()),
            "loc": ifnone(loc, loc),
            "scale": scale
        }


class GPD_reparametrized(ScipyDistribution):
    params_names = ("loc", "r", "shape", "p")
    base_module = genpareto

    def __init__(self, loc=0.0, r=1.0, shape=0.0, p=1 / 200):
        super(GPD_reparametrized, self).__init__(loc, r, shape, ConstantParameter(p))

    def _to_scipy_args(self, loc=None, r=None, shape=None):
        loc = ifnone(loc, self.loc())
        shape = ifnone(shape, self.shape())
        r = ifnone(r, self.r())
        p = float(self.p())
        scale = (r - loc) * shape / (p ** (-shape) - 1) if shape != 0 else (r - loc) / np.log(1 / p)
        return {
            "c": ifnone(shape, self.shape()),
            "loc": loc,
            "scale": ifnone(scale, scale),
        }
