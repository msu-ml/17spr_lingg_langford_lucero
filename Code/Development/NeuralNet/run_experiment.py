# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 23:10:38 2017

@author: mick
"""

import getopt
import matplotlib.cbook
import numpy as np
import sys
import warnings
from Experiment import Experiment

# set RNG with a specific seed
seed = 69
np.random.seed(seed)

def main(argv):
    # Suppress deprecation warnings. I don't care.
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore",
                            category=matplotlib.cbook.MatplotlibDeprecationWarning) 

    experiment = Experiment()
    experiment.run()
    #experiment.cross_validate()

if __name__ == '__main__':
    main(sys.argv[1:])
