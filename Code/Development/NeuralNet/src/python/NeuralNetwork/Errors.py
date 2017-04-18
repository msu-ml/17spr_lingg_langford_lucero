# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:59:59 2017

@author: mick
"""

import numpy as np

class Errors(object):
    class CategoricalCrossEntropy(object):
        @staticmethod
        def func(y, t):
            return np.sum(-t*np.log(y) - (1-t)*np.log(1.0-y))
        @staticmethod
        def deriv(y, t):
            return y - t

    class MeanSquared(object):
        @staticmethod
        def func(y, t):
            return 0.5 * np.linalg.norm(y - t)**2
        @staticmethod
        def deriv(y, t):
            return y - t