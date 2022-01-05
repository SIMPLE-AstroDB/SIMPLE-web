"""
Testing the plot functions
"""
# local packages
from simple_app.plots import *
from simple_app.tests.test_utils import *


@pytest.fixture(scope='session')
def test_mainplots():
    nightskytheme, jscallbacks = mainplots()
    assert type(nightskytheme) == Theme
    assert type(jscallbacks) == JSCallbacks
    return nightskytheme, jscallbacks


def test_multiplotbokeh(db, test_all_sources, test_all_photometry, test_all_parallaxes, test_mainplots):
    assert db
    allphoto, allbands = test_all_photometry
    allresults, fullresults = test_all_sources
    allplx = test_all_parallaxes
    nightskytheme, jscallbacks = test_mainplots
    script, div = multiplotbokeh(fullresults, allbands, allphoto, allplx, jscallbacks, nightskytheme)
    assert all([type(s) == str for s in (script, div)])
    return


def test_specplot(db, test_mainplots):
    assert db
    nightskytheme, jscallbacks = test_mainplots
    good_query = '2MASS J00192626+4614078'
    bad_query = 'thisisabadquery'
    goodscript, gooddiv = specplot(good_query, db_cs, nightskytheme, jscallbacks)[:2]
    badscript, baddiv = specplot(bad_query, db_cs, nightskytheme, jscallbacks)[:2]
    assert all([type(s) == str for s in (goodscript, gooddiv)])
    assert all([s is None for s in (badscript, baddiv)])
    return


def test_camdplot(db, test_mainplots, test_all_photometry, test_all_sources, test_all_parallaxes, test_get_filters):
    assert db
    allphoto, allbands = test_all_photometry
    photfilters = test_get_filters
    args = argparse.ArgumentParser()
    args = args.parse_args([])
    args.debug = False
    good_query = '2MASS J00192626+4614078'
    bad_query = 'thisisabadquery'
    nightskytheme, jscallbacks = test_mainplots
    resultdict: dict = db.inventory(good_query)
    goodeverything = Inventory(resultdict, args)
    resultdict = db.inventory(bad_query)
    badeverything = Inventory(resultdict, args)
    allresults, allresultsfull = test_all_sources
    allplx = test_all_parallaxes
    goodscript, gooddiv = camdplot(good_query, goodeverything, allbands,
                                   allresultsfull, allplx, photfilters, allphoto, jscallbacks, nightskytheme)
    badscript, baddiv = camdplot(bad_query, badeverything, allbands,
                                 allresultsfull, allplx, photfilters, allphoto, jscallbacks, nightskytheme)
    assert all([type(s) == str for s in (goodscript, gooddiv)])
    assert all([s is None for s in (badscript, baddiv)])
    return