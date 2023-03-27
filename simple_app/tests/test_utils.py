"""
Testing the utils functions

Non directly tested functions:
find_colours -- called by all_photometry
parse_photometry -- called by all_photometry
absmags -- called by results_concat
coordinate_project -- called by results_concat
"""
import sys
sys.path.append('simple_app')
# local packages
from utils import *

db_name = 'simple_root/temp.db'
db_cs = f'sqlite:///{db_name}'


@pytest.fixture(scope='module')
def db():
    if os.path.exists(db_name):
        os.remove(db_name)
    copy('simple_root/SIMPLE.db', db_name)
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
def test_get_version(db):
    assert db
    vstr = get_version(db_cs)
    assert type(vstr) == str
    return vstr


@pytest.fixture(scope='module')
def test_all_spectraltypes(db):
    assert db
    allspts = all_spectraltypes(db_cs)
    assert len(allspts)
    assert type(allspts) == pd.DataFrame
    return allspts


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


def test_results_concat(db, test_all_photometry, test_all_sources, test_all_parallaxes, test_all_spectraltypes):
    assert db
    allphoto, allbands = test_all_photometry
    allresults, fullresults = test_all_sources
    allplx = test_all_parallaxes
    allspts = test_all_spectraltypes
    allresultsconcat = results_concat(fullresults, allphoto, allplx, allspts, allbands)
    assert all([col in allresultsconcat.columns for col in ('raproj', 'decproj')])
    assert all([f'M_{band}' in allresultsconcat.columns for band in allbands])
    return


def test_onedfquery(db):
    assert db
    badquery = 'thisisabadquery'
    goodquery = 'twa'
    # test search object
    results = db.search_object(badquery, fmt='pandas')
    assert not len(results)  # bad query should return empty table
    results = db.search_object(goodquery, fmt='pandas')
    assert isinstance(results, pd.DataFrame)
    # test search string
    with pytest.raises(KeyError):
        refresults: Optional[dict] = db.search_string(badquery, fmt='pandas', verbose=False)
        _ = refresults['Sources']
    refresults: Optional[dict] = db.search_string(goodquery, fmt='pandas', verbose=False)
    assert isinstance(refresults, dict)
    assert 'Sources' in refresults
    refsources = refresults['Sources']
    filtered_results: Optional[pd.DataFrame] = results.merge(refsources, on='source', suffixes=(None, 'extra'))
    assert isinstance(filtered_results, pd.DataFrame)
    filtered_results.drop(columns=list(filtered_results.filter(regex='extra')), inplace=True)
    # test onedfquery
    stringed_results = onedfquery(filtered_results)
    assert isinstance(stringed_results, str)
    # test sql query
    with pytest.raises(OperationalError):
        _ = db.sql_query('notasqlquery', fmt='pandas')
    with pytest.raises(OperationalError):
        _ = db.sql_query('select * from NotaTable', fmt='pandas')
    with pytest.raises(OperationalError):
        _ = db.sql_query('select * from Sources where notacolumn == "asdf"', fmt='pandas')
    rawsqlquery = db.sql_query('select * from Sources where source == "Luhman 16"', fmt='pandas')
    assert isinstance(rawsqlquery, pd.DataFrame)
    stringed_results = onedfquery(rawsqlquery)
    assert isinstance(stringed_results, str)
    return


def test_multidfquery(db):
    assert db
    badquery = 'thisisabadquery'
    goodquery = 'cruz'
    with pytest.raises(KeyError):
        results: Optional[dict] = db.search_string(badquery, fmt='pandas', verbose=False)
        _ = results['Sources']
    results: Optional[dict] = db.search_string(goodquery, fmt='pandas', verbose=False)
    assert isinstance(results, dict)
    assert 'Sources' in results
    assert isinstance(results['Sources'], pd.DataFrame)
    resultsout = multidfquery(results)
    assert isinstance(resultsout, dict)
    assert 'Sources' in resultsout
    assert isinstance(resultsout['Sources'], str)
    return

