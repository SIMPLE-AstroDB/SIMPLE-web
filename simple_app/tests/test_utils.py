"""
Testing the utils functions

Non directly tested functions:
find_colours -- called by get_all_photometry
parse_photometry -- called by get_all_photometry
absmags -- called by results_concat
coordinate_project -- called by results_concat
"""
# local packages
from ..utils import *

db_name = 'temp.sqlite'
db_cs = f'sqlite:///{db_name}'


@pytest.fixture(scope='module')
def db():
    if os.path.exists(db_name):
        os.remove(db_name)
    copy('SIMPLE.sqlite', db_name)
    assert os.path.exists(db_name)
    # Connect to the new database and confirm it has the Sources table
    db = SimpleDB(db_cs)
    assert db
    assert 'source' in [c.name for c in db.Sources.columns]
    return db


@pytest.fixture(scope='module')
def test_get_all_sources(db):
    assert db
    assert get_all_sources(db_cs)
    all_results, all_results_full = get_all_sources(db_cs)
    assert len(all_results) and len(all_results_full)
    assert type(all_results) == list
    assert type(all_results_full) == pd.DataFrame
    return all_results, all_results_full


@pytest.fixture(scope='module')
def test_get_all_photometry(db, test_get_filters):
    assert db
    photometric_filters = test_get_filters
    all_photometry, all_bands = get_all_photometry(db_cs, photometric_filters)
    assert len(all_photometry) and len(all_bands)
    assert type(all_photometry) == pd.DataFrame
    assert type(all_bands) == np.ndarray
    return all_photometry, all_bands


@pytest.fixture(scope='module')
def test_get_all_parallaxes(db):
    assert db
    all_parallaxes = get_all_parallaxes(db_cs)
    assert len(all_parallaxes)
    assert type(all_parallaxes) == pd.DataFrame
    return all_parallaxes


@pytest.fixture(scope='module')
def test_get_version(db):
    assert db
    v_str = get_version(db_cs)
    assert type(v_str) == str
    return v_str


@pytest.fixture(scope='module')
def test_get_all_spectral_types(db):
    assert db
    all_spectral_types = get_all_spectral_types(db_cs)
    assert len(all_spectral_types)
    assert type(all_spectral_types) == pd.DataFrame
    return all_spectral_types


@pytest.fixture(scope='module')
def test_get_filters(db):
    assert db
    photometric_filters = get_filters(db_cs)
    assert len(photometric_filters)
    assert len(photometric_filters.columns)
    assert 'effective_wavelength' in photometric_filters.index
    assert photometric_filters.at['effective_wavelength', 'WISE.W1']
    with pytest.raises(KeyError):
        _ = photometric_filters.at['effective_wavelength', 'notaband']
    return photometric_filters


def test_inventory(db):
    assert db
    args = argparse.ArgumentParser()
    args = args.parse_args([])
    args.debug = False
    good_query = '2MASS J00192626+4614078'
    d_results: dict = db.inventory(good_query)
    assert len(d_results)
    everything = Inventory(d_results, db_cs)
    assert all([hasattr(everything, s) for s in
                ('photometry', 'sources', 'names', 'spectra', 'ra', 'dec', 'propermotions')])
    return


def test_find_colours(db, test_get_all_photometry, test_get_filters):
    assert db
    photometric_filters = test_get_filters
    all_photometry, all_bands = test_get_all_photometry
    df_photometry = find_colours(all_photometry, all_bands, photometric_filters)
    assert len(df_photometry)
    assert type(df_photometry) == pd.DataFrame
    return


def test_results_concat(db, test_get_all_photometry, test_get_all_sources,
                        test_get_all_parallaxes, test_get_all_spectral_types):
    assert db
    all_photometry, all_bands = test_get_all_photometry
    all_results, all_results_full = test_get_all_sources
    all_parallaxes = test_get_all_parallaxes
    all_spectral_types = test_get_all_spectral_types
    wanted_mags = {'GAIA3.G', '2MASS.J', 'WISE.W1'}
    all_results_concat = results_concat(all_results_full, all_photometry, all_parallaxes, all_spectral_types, all_bands)
    assert all([col in all_results_concat.columns for col in ('ra_projected', 'dec_projected')])
    assert all([f'M_{band}' in all_results_concat.columns for band in all_bands if band in wanted_mags])
    return


def test_one_df_query(db):
    assert db
    bad_query = 'thisisabadquery'
    good_query = 'twa'
    
    # test search object
    results = db.search_object(bad_query, fmt='pandas')
    assert not len(results)  # bad query should return empty table
    results = db.search_object(good_query, fmt='pandas')
    assert isinstance(results, pd.DataFrame)

    # test search string
    with pytest.raises(KeyError):
        ref_results: Optional[dict] = db.search_string(bad_query, fmt='pandas', verbose=False)
        _ = ref_results['Sources']
    ref_results: Optional[dict] = db.search_string(good_query, fmt='pandas', verbose=False)
    assert isinstance(ref_results, dict)
    assert 'Sources' in ref_results
    ref_sources = ref_results['Sources']
    filtered_results: Optional[pd.DataFrame] = results.merge(ref_sources, on='source', suffixes=(None, 'extra'))
    assert isinstance(filtered_results, pd.DataFrame)
    filtered_results.drop(columns=list(filtered_results.filter(regex='extra')), inplace=True)

    # test one_df_query
    stringed_results = one_df_query(filtered_results)
    assert isinstance(stringed_results, str)

    # test sql query
    with pytest.raises(OperationalError):
        _ = db.sql_query('notasqlquery', fmt='pandas')
    with pytest.raises(OperationalError):
        _ = db.sql_query('select * from NotaTable', fmt='pandas')
    with pytest.raises(OperationalError):
        _ = db.sql_query('select * from Sources where notacolumn == "asdf"', fmt='pandas')

    # Using a source that returns a single row
    raw_sql_query = db.sql_query('select * from Sources where source == "WISE J104915.57-531906.1"', fmt='pandas')
    assert isinstance(raw_sql_query, pd.DataFrame)

    # Testing conversion to a markdown string
    stringed_results = one_df_query(raw_sql_query)
    assert isinstance(stringed_results, str)


def test_multi_df_query(db):
    assert db
    bad_query = 'thisisabadquery'
    good_query = 'cruz'
    with pytest.raises(KeyError):
        results: Optional[dict] = db.search_string(bad_query, fmt='pandas', verbose=False)
        _ = results['Sources']
    results: Optional[dict] = db.search_string(good_query, fmt='pandas', verbose=False)
    assert isinstance(results, dict)
    assert 'Sources' in results
    assert isinstance(results['Sources'], pd.DataFrame)
    results_out = multi_df_query(results, db_cs)
    assert isinstance(results_out, dict)
    assert 'Sources' in results_out
    assert isinstance(results_out['Sources'], str)
    return


def test_multi_param_str_parse():
    twa_query = '174.96308 	-31.989305'
    empty_query = ''
    hms_query = '12h34m56s \t +78d90m12s'
    for query in (twa_query, empty_query, hms_query):
        a, b, c = CoordQueryForm.multi_param_str_parse(query)
        assert isinstance(a, str)
        assert isinstance(b, str)
        assert isinstance(c, float)
        assert c == 10.
    twa_query = '174.96308 \t	-31.989305 15 '
    a, b, c = CoordQueryForm.multi_param_str_parse(twa_query)
    assert a == '174.96308'
    assert b == '-31.989305'
    assert c == 15.
    return


def test_ra_dec_unit_parse():
    twa_query = '174.96308 	-31.989305'
    a, b, c = CoordQueryForm.multi_param_str_parse(twa_query)
    ra, dec, unit = CoordQueryForm.ra_dec_unit_parse(a, b)
    assert isinstance(ra, float)
    assert isinstance(dec, float)
    assert unit == 'deg'
    hms_query = '12h34m56s \t +78d90m12s'
    a, b, c = CoordQueryForm.multi_param_str_parse(hms_query)
    ra, dec, unit = CoordQueryForm.ra_dec_unit_parse(a, b)
    assert unit == 'hourangle,deg'
    return
