# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:59:59 2017

@author: mick
"""

import numpy as np

class Activations(object):
    class Sigmoid(object):
        @staticmethod
        def func(z):
            return 1.0 / (1.0 + np.exp(-z))
        @staticmethod
        def deriv(z):
            sig_z = Activations.Sigmoid.func(z)
            return sig_z * (1.0 - sig_z)
        
    class HyperTangent(object):
        @staticmethod
        def func(z):
            return np.tanh(z)
        @staticmethod
        def deriv(z):
            tanh_z = np.tanh(z)
            return 1.0 - tanh_z * tanh_z
        
    class Softmax(object):
        @staticmethod
        def func(z):
            exp_z = np.exp(z - np.max(z))
            return exp_z / np.sum(exp_z)
        @staticmethod
        def deriv(z):
            smax_z = Activations.Softmax.func(z)
            return smax_z * (1.0 - smax_z)
	