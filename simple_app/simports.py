"""
Importing all packages
"""
# external packages
from astrodbkit2.astrodb import Database, REFERENCE_TABLES  # used for pulling out database and querying
from astropy.coordinates import SkyCoord  # coordinates
import astropy.units as u  # units
from astropy.table import Table  # tables in astropy
from bokeh.embed import components  # converting python bokeh to javascript
from bokeh.layouts import row, column  # bokeh displaying nicely
from bokeh.models import ColumnDataSource, Range1d, CustomJS,\
    Select, Toggle, TapTool, OpenURL, HoverTool, Span, RangeSlider, Label  # bokeh models
from bokeh.palettes import Colorblind8
from bokeh.plotting import figure, curdoc  # bokeh plotting
from bokeh.resources import CDN  # resources for webpage
from bokeh.themes import built_in_themes, Theme  # appearance of bokeh glyphs
from flask import Flask, render_template, jsonify  # website functionality
from flask_cors import CORS  # cross origin fix (aladin mostly)
from flask_wtf import FlaskForm  # web forms
from markdown2 import markdown  # using markdown formatting
import numpy as np  # numerical python
import pandas as pd  # running dataframes
import pytest  # testing
from specutils import Spectrum1D  # spectrum objects
from sqlalchemy.exc import ResourceClosedError, OperationalError  # errors from sqlalchemy
from sqlite3 import Warning as SqliteWarning  # errors from sqlite
from tqdm import tqdm  # progress bars
from wtforms import StringField, SubmitField, TextAreaField, ValidationError  # web forms

# internal packages
import argparse  # parsing the arguments given with file
from copy import deepcopy
import os  # operating system
from shutil import copy  # copying files
import sys  # system arguments
from typing import Tuple, Optional, List, Union, Dict  # type hinting (good in IDEs)
from urllib.parse import quote  # handling strings into url friendly form
