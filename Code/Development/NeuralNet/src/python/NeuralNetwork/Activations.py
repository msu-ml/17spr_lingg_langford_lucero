# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:59:59 2017

@author: mick
"""

import numpy as np
import sys

class Activations(object):
    class Linear(object):
        @staticmethod
        def func(z):
            return z
        @staticmethod
        def deriv(z):
            return np.ones(z.shape)

    class RectifiedLinear(object):
        @staticmethod
        def func(z):
            return z * (z > 0.0).astype(np.float32)
        @staticmethod
        def deriv(z):
            return 1.0 * (z > 0.0).astype(np.float32)

    class PiecewiseLinear(object):
        min_bound = 0.0
        max_bound = 1.0
        @staticmethod
        def func(z):
            m1 = Activations.PiecewiseLinear.min_bound
            m2 = Activations.PiecewiseLinear.max_bound
            return z * (z >= m1).astype(np.float32) * (z <= m2).astype(np.float32)
        @staticmethod
        def deriv(z):
            m1 = Activations.PiecewiseLinear.min_bound
            m2 = Activations.PiecewiseLinear.max_bound
            return 1.0 * (z >= m1).astype(np.float32) * (z <= m2).astype(np.float32)

    class Bipolar(object):
        @staticmethod
        def func(z):
            return -1.0 * (z < 0.0).astype(np.float32) + 1.0 * (z > 0.0).astype(np.float32)
        @staticmethod
        def deriv(z):
            return np.zeros(z.shape)

    class Sigmoid(object):
        @staticmethod
        def func(z):
            return 1.0 / (1.0 + np.exp(-z))
        @staticmethod
        def deriv(z):
            sig_z = Activations.Sigmoid.func(z)
            return sig_z - sig_z**2

    class BipolarSigmoid(object):
        @staticmethod
        def func(z):
            return -1.0 + 2.0 / (1.0 + np.exp(-z))
        @staticmethod
        def deriv(z):
            bsig_z = Activations.BipolarSigmoid.func(z)
            return 0.5 * (1.0 + bsig_z) * (1.0 - bsig_z)

    class HyperTangent(object):
        @staticmethod
        def func(z):
            return np.tanh(z)
        @staticmethod
        def deriv(z):
            tanh_z = np.tanh(z)
            return 1.0 - tanh_z**2
        
    class Softmax(object):
        @staticmethod
        def func(z):
            return np.exp(z) / np.sum(np.exp(z))
        @staticmethod
        def deriv(z):
            smax_z = Activations.Softmax.func(z)
            return smax_z * (1.0 - smax_z)
	