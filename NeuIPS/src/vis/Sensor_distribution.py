import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
import sys
import yaml
from types import SimpleNamespace as SN
import torch

import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import pickle


def plot_compared_results_bar(pos):
    import matplotlib.pyplot as plt
    plt.title("Distribution of sensors")
    # plt.xlim(xmax=7, xmin=0)
    # plt.ylim(ymax=7, ymin=0)
    plt.xlabel("x Locations")
    plt.ylabel("y Locations")
    plt.plot(pos[:,0], pos[:,1], 'ro')
    plt.show()




if __name__ == '__main__':
    pos = np.random.randn(10,2)
    plot_compared_results_bar(pos)
