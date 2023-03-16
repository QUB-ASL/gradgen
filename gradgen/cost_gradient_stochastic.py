import casadi.casadi as cs
import subprocess as subp
import os
import shutil
import jinja2
from gradgen.definitions import *
from cost_gradient import CostGradient


class CostGradientStochastic(CostGradient):

    def __init__(self, x, u, w, f, ell, vf, N):
        """
        Create new stochastic CostGradient object

        :param x: state symbol
        :param u: input symbol
        :param w: list of events
        :param f: list of system dynamics symbol (depends on x, u, w)
        :param ell: list of cost function symbol (depends on x, u, w)
        :param vf: terminal cost symbol (depends on x)
        :param N: prediction horizon
        """
        super().__init__(x, u, f, ell, vf, N)
        self.__x = x
        self.__u = u
        self.__w = w
        self.__f = f
        self.__ell = ell
        self.__vf = vf
        self.__N = N
        self.__nx = x.size()[0]
        self.__nu = u.size()[0]
        self.__d = cs.SX.sym('d', self.__nx)
        self.__jfx = None
        self.__jfu = None
        self.__ellx = None
        self.__ellu = None
        self.__vfx = None
        self.__f_fun = None
        self.__jfx_fun = None
        self.__jfu_fun = None
        self.__ell_fun = None
        self.__ellx_fun = None
        self.__ellu_fun = None
        self.__vf_fun = None
        self.__vfx_fun = None
        self.__name = 'gradgen'
        self.__destination_path = 'codegenz'