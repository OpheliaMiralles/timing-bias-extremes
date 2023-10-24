from scipy.stats import genpareto, genextreme

from pykelihood.distributions import ScipyDistribution
from pykelihood.parameters import ConstantParameter
from pykelihood.utils import ifnone
import numpy as np


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
        loc = r - (scale / shape) * ((-np.log(1 - p)) ** (-shape) - 1) if shape != 0 else r - scale * (
            -(np.log(-np.log(1 - p))))
        if shape is not None:
            shape = -shape
        return {
            "c": ifnone(shape, -self.shape()),
            "loc": ifnone(loc, loc),
            "scale": scale
        }

class GEV_reparametrized_p(ScipyDistribution):
    params_names = ("loc", "p", "shape", "r")
    base_module = genextreme

    def __init__(self, loc=0.0, p=1/200, shape=0.0, r=1.0):
        super(GEV_reparametrized_p, self).__init__(loc, p, shape, ConstantParameter(r))

    def _to_scipy_args(self, loc=None, p=None, shape=None):
        loc = ifnone(loc, self.loc())
        shape = ifnone(shape, self.shape())
        r = float(self.r())
        p = ifnone(p, self.p())
        scale = (r - loc) * shape / ((-np.log(1 - p)) ** (-shape) - 1) if shape != 0 else (r - loc) / (-(np.log(-np.log(1 - p))))
        if shape is not None:
            shape = -shape
        return {
            "c": ifnone(shape, -self.shape()),
            "loc": loc,
            "scale": ifnone(scale, scale),
        }

class GPD_reparametrized_r_loc(ScipyDistribution):
    params_names = ("r", "scale", "shape", "p")
    base_module = genpareto

    def __init__(self, r=0.0, scale=1.0, shape=0.0, p=1 / 200):
        super(GPD_reparametrized_r_loc, self).__init__(r, scale, shape, ConstantParameter(p))

    def _to_scipy_args(self, r=None, scale=None, shape=None):
        scale = ifnone(scale, self.scale())
        shape = ifnone(shape, self.shape())
        r = ifnone(r, self.r())
        p = float(self.p())
        loc = r - (p ** -shape - 1) * scale / shape if shape != 0 else r + scale * np.log(p)
        return {
            "c": ifnone(shape, self.shape()),
            "loc": loc,
            "scale": ifnone(scale, scale),
        }
