"""
Testing the utils functions

Non directly tested functions:
find_colours -- called by all_photometry
parse_photometry -- called by all_photometry
absmags -- called by results_concat
coordinate_project -- called by results_concat
"""
# external packages
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
def test_all_photometry(db, test_get_filters):
    assert db
    photfilters = test_get_filters
    allphoto, allbands = all_photometry(db_cs, photfilters)
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


@pytest.fixture(scope='module')
def test_get_filters(db):
    assert db
    photfilters = get_filters(db_cs)
    assert len(photfilters)
    assert len(photfilters.columns)
    assert 'effective_wavelength' in photfilters.index
    assert photfilters.at['effective_wavelength', 'WISE.W1']
    with pytest.raises(KeyError):
        _ = photfilters.at['effective_wavelength', 'notaband']
    return photfilters


def test_inventory(db):
    assert db
    args = argparse.ArgumentParser()
    args = args.parse_args([])
    args.debug = False
    good_query = '2MASS J00192626+4614078'
    resultdict: dict = db.inventory(good_query)
    assert len(resultdict)
    everything = Inventory(resultdict, args)
    assert all([hasattr(everything, s) for s in
                ('photometry', 'sources', 'names', 'spectra', 'ra', 'dec', 'propermotions')])
    return


def test_find_colours(db, test_all_photometry, test_get_filters):
    assert db
    photfilters = test_get_filters
    allphoto, allbands = test_all_photometry
    photodf = find_colours(allphoto, allbands, photfilters)
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
