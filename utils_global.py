import numpy as np
import pandas as pd
import random
import math

from collections import defaultdict
import json
import pickle


import os

import pdb

MODEL_PATH = ['./rf.pkl','./gb.pkl']
SOURCE_PATH = './data/HR200709to201901.csv'
MODIFIED_PATH = './data/data_with_chg.csv'
#####################
mode = 'test'
#####################
if mode == 'train':
    if os.path.exists(MODIFIED_PATH):
        FILEPATH = MODIFIED_PATH
    else:
        FILEPATH = SOURCE_PATH
if mode == 'test':
    FILEPATH = './data/Sample_test.csv'

GOING_SPACE = [
 'WET FAST',
 'GOOD TO FIRM',
 'FAST',
 'SLOW',
 'WET SLOW',
 'SOFT',
 'YIELDING TO SOFT',
 'GOOD',
 'GOOD TO YIELDING',
 'YIELDING'
]

TRACK_SPACE = ['TURF', 'ALL WEATHER TRACK']

VENUE_SPACE = ['HV', 'ST']

prob_thres = 0.74
