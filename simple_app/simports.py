"""
Importing all packages
"""
# external packages
from astrodbkit.astrodb import Database  # used for pulling out database and querying
from astropy.coordinates import SkyCoord  # coordinates
from astropy.io import fits  # handling fits files
import astropy.units as u  # units
from astropy.table import Table  # tables in astropy
from bokeh.embed import components  # converting python bokeh to javascript
from bokeh.layouts import row, column  # bokeh displaying nicely
from bokeh.models import ColumnDataSource, Range1d, CustomJS, \
    Select, Toggle, TapTool, OpenURL, HoverTool, Span, RangeSlider, Label, ColorBar, FixedTicker  # bokeh models
from bokeh.palettes import Colorblind8, Turbo256  # plotting palettes
from bokeh.plotting import figure, curdoc  # bokeh plotting
from bokeh.resources import CDN  # resources for webpage
from bokeh.themes import built_in_themes, Theme  # appearance of bokeh glyphs
from bokeh.transform import linear_cmap  # making colour maps
from flask import (Flask, render_template, jsonify, send_from_directory, redirect, url_for,
                   Response, abort, request, session)  # website
from flask_cors import CORS  # cross origin fix (aladin mostly)
from flask_wtf import FlaskForm  # web forms
from markdown2 import markdown  # using markdown formatting
import numpy as np  # numerical python
import pandas as pd  # running dataframes
import pytest  # testing
from specutils import Spectrum1D  # spectrum objects
from sqlalchemy.exc import ResourceClosedError, OperationalError, ProgrammingError  # errors from sqlalchemy
from sqlite3 import Warning as SqliteWarning  # errors from sqlite
from tqdm import tqdm  # progress bars
from werkzeug.exceptions import HTTPException  # underlying http
from wtforms import StringField, SubmitField, TextAreaField, ValidationError  # web forms

# internal packages
import argparse  # parsing the arguments given with file
from copy import deepcopy  # memory control
from difflib import get_close_matches  # for redirecting bad file paths
from io import StringIO, BytesIO, BufferedIOBase  # writing files without saving to disk
import multiprocess as mp  # multiprocessing for efficiency
import os  # operating system
import requests  # accessing internet
from shutil import copy  # copying files
import sys  # system arguments
from time import strftime, localtime  # time stuff for naming files
from typing import Tuple, Optional, List, Union, Dict, Generator  # type hinting (good in IDEs)
from urllib.parse import quote  # handling strings into url friendly form
from zipfile import ZipFile  # zipping files together
