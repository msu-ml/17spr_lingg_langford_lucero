# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 23:10:38 2017

@author: mick
"""

import getopt
"""
import matplotlib.cbook
"""
import numpy as np
import sys
import warnings
from Experiment import Experiment

# set RNG with a specific seed
seed = 69
np.random.seed(seed)

def main(argv):
    """
    # Suppress deprecation warnings. I don't care.
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore",
                            category=matplotlib.cbook.MatplotlibDeprecationWarning) 
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'n:', ['iters='])
    except getopt.GetoptError as err:
        sys.exit(2)

    num_iters=1000
    for opt, arg in opts:
        if opt in ('-n', '--iters'):
            num_iters = int(arg)

    experiment = Experiment()
    experiment.run(num_iters)

if __name__ == '__main__':
    main(sys.argv[1:])
