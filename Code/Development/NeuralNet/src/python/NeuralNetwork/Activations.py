# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:59:59 2017

@author: mick
"""

import numpy as np
import sys

class Activations(object):
    class Sigmoid(object):
        @staticmethod
        def func(z):
            return 1.0 / (1.0 + np.exp(-z))
        @staticmethod
        def deriv(z):
            sig_z = Activations.Sigmoid.func(z)
            return sig_z - sig_z**2
        
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
	