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
            sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
            return sigmoid(z)
        @staticmethod
        def deriv(z):
            sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
            return sigmoid(z) * (1.0 - sigmoid(z))