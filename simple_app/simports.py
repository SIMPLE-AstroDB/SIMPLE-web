"""
Importing all packages
"""

import argparse  # parsing the arguments given with file
import os  # operating system
import sys  # system arguments
from copy import deepcopy  # memory control
from difflib import get_close_matches  # for redirecting bad file paths
from io import BufferedIOBase, BytesIO, StringIO  # writing files without saving to disk
from shutil import copy  # copying files
from sqlite3 import Warning as SqliteWarning  # errors from sqlite
from time import localtime, strftime  # time stuff for naming files
from typing import Dict, Generator, List, Optional, Tuple, Union  # type hinting (good in IDEs)
from urllib.parse import quote, unquote, urlparse  # handling strings into url friendly form
from zipfile import ZipFile  # zipping files together

import astropy.units as u  # units
import multiprocess as mp  # multiprocessing for efficiency
import numpy as np  # numerical python
import pandas as pd  # running dataframes
import pytest  # testing
import requests  # accessing internet
from astrodbkit.astrodb import Database  # used for pulling out database and querying
from astrodb_utils.utils import AstroDBError  # error message from database settings
from astropy.coordinates import SkyCoord  # coordinates
from astropy.io import fits  # handling fits files
from astropy.table import Table  # tables in astropy
from bokeh.embed import components  # converting python bokeh to javascript
from bokeh.layouts import column, row  # bokeh displaying nicely
from bokeh.models import (
                   ColorBar,
                   ColumnDataSource,
                   CustomJS,
                   FixedTicker,
                   HoverTool,
                   Label,
                   OpenURL,
                   Range1d,
                   RangeSlider,
                   Select,  # bokeh models
                   Span,
                   TapTool,
                   Toggle,
)
from bokeh.palettes import Colorblind8, Turbo256  # plotting palettes
from bokeh.plotting import curdoc, figure  # bokeh plotting
from bokeh.resources import CDN  # resources for webpage
from bokeh.themes import Theme, built_in_themes  # appearance of bokeh glyphs
from bokeh.transform import linear_cmap  # making colour maps
from flask import (
                   Flask,
                   Response,  # website
                   abort,
                   jsonify,
                   redirect,
                   render_template,
                   request,
                   send_from_directory,
                   session,
                   url_for,
)
from flask_cors import CORS  # cross origin fix (aladin mostly)
from flask_wtf import FlaskForm  # web forms
from markdown2 import markdown  # using markdown formatting
from specutils import Spectrum
from sqlalchemy.exc import OperationalError, ProgrammingError, ResourceClosedError  # errors from sqlalchemy
from tqdm import tqdm  # progress bars
from werkzeug.exceptions import HTTPException  # underlying http
from wtforms import StringField, SubmitField, TextAreaField, ValidationError  # web forms
import tomllib
with open("database.toml", "rb") as f:
    settings = tomllib.load(f)
    REFERENCE_TABLES = settings['lookup_tables']
