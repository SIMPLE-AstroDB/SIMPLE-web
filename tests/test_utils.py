"""
Testing the utils functions

Non directly tested functions:
find_colours -- called by all_photometry
parse_photometry -- called by all_photometry
absmags -- called by results_concat
coordinate_project -- called by results_concat
"""
# external packages
import pandas as pd
import pytest
# internal packages
import os
from shutil import copy
# local packages
from simple_app.utils import *


db_name = 'temp.db'
db_cs = 'sqlite:///temp.db'


@pytest.fixture(scope='module')
def db():
    if os.path.exists(db_name):
        os.remove(db_name)
    copy('SIMPLE.db', db_name)
    assert os.path.exists(db_name)
    # Connect to the new database and confirm it has the Sources table
    db = SimpleDB(db_cs)
    assert db
    assert 'source' in [c.name for c in db.Sources.columns]
    return db


@pytest.fixture(scope='module')
def test_all_sources(db):
    assert db
    assert all_sources(db_cs)
    allresults, fullresults = all_sources(db_cs)
    assert len(allresults) and len(fullresults)
    assert type(allresults) == list
    assert type(fullresults) == pd.DataFrame
    return allresults, fullresults


@pytest.fixture(scope='module')
def test_all_photometry(db):
    assert db
    allphoto, allbands = all_photometry(db_cs)
    assert len(allphoto) and len(allbands)
    assert type(allphoto) == pd.DataFrame
    assert type(allbands) == np.ndarray
    return allphoto, allbands


@pytest.fixture(scope='module')
def test_all_parallaxes(db):
    assert db
    allplx = all_parallaxes(db_cs)
    assert len(allplx)
    assert type(allplx) == pd.DataFrame
    return allplx


def test_find_colours(db, test_all_photometry):
    assert db
    allphoto, allbands = test_all_photometry
    photodf = find_colours(allphoto, allbands)
    assert len(photodf)
    assert type(photodf) == pd.DataFrame
    return


def test_results_concat(db, test_all_photometry, test_all_sources, test_all_parallaxes):
    assert db
    allphoto, allbands = test_all_photometry
    allresults, fullresults = test_all_sources
    allplx = test_all_parallaxes
    allresultsconcat = results_concat(fullresults, allphoto, allplx, allbands)
    assert all([col in allresultsconcat.columns for col in ('dist', 'raproj', 'decproj')])
    assert all([f'M_{band}' in allresultsconcat.columns for band in allbands])
    return


def test_remove_database(db):
    # Clean up temporary database
    db.session.close()
    db.engine.dispose()
    if os.path.exists(db_name):
        os.remove(db_name)
    return
