#!/usr/bin/env python3

import numpy as np

import os
import datetime
import glob
import pathlib

from plotter import Plotter

folders = ["CF", ""]
HOME_FOLDER = os.path.abspath(os.path.dirname(__file__))

for folder in folders:
    for filename in glob.glob(os.path.join(HOME_FOLDER, "..", "data", folder, "*.csv")):
        # https://pynative.com/python-file-creation-modification-datetime/
        
        f_name = pathlib.Path(filename)

        # get creation time on windows
        c_timestamp = f_name.stat().st_ctime
        c_time = datetime.datetime.fromtimestamp(c_timestamp)
        print(c_time)