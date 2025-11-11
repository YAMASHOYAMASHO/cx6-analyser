from matplotlib import pyplot as plt
import numpy as np
import fabio
from tkinter import filedialog
from tqdm import tqdm
import numpy as np
import math
import pyFAI.detectors
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from SPring8 import SP8_data
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import json
import os
import matplotlib.animation as animation
import matplotlib.cm as cm

plt.rcParams['figure.figsize'] = [5, 4]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'