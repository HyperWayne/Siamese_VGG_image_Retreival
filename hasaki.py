# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 23:02:34 2017

@author: Wayne Lee
"""

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


def create_image_lists(image_dir, testing_percentage, validation_percentage):

 if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None

