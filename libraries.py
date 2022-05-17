import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import exists
import pickle

from scipy.fftpack import dct, fft
from scipy.signal import hamming, lfilter

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Resample

from tqdm import tqdm
import warnings
