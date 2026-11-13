"""
Testing the utils functions; fixtures are in conftest.py
"""
import pytest

from ..utils import *


def test_get_all_sources(sources):
    all_results, all_results_full = sources
    assert len(all_results) and len(all_results_full)
    assert isinstance(all_results, list)
    assert isinstance(all_results_full, pd.DataFrame)


def test_get_all_photometry(photometry):
    all_photometry, all_bands = photometry
    assert len(all_photometry) and len(all_bands)
    assert isinstance(all_photometry, pd.DataFrame)
    assert isinstance(all_bands, np.ndarray)


def test_get_all_parallaxes(parallaxes):
    assert len(parallaxes)
    assert isinstance(parallaxes, pd.DataFrame)


def test_get_version(db_cs):
    assert get_version(db_cs).startswith('Version ')


def test_get_all_spectral_types(spectral_types):
    assert len(spectral_types)
    assert isinstance(spectral_types, pd.DataFrame)


def test_get_all_modeled_parameters(modeled_parameters):
    assert modeled_parameters.source.is_unique
    assert {'mass [M_jup]', 'T eff [K]', 'log g [dex]'} <= set(modeled_parameters.columns)
    assert 'mass [M_sun]' not in modeled_parameters.columns


def test_get_filters(filters):
    assert 'effective_wavelength' in filters.index
    assert filters.at['effective_wavelength', 'WISE.W1']
    with pytest.raises(KeyError):
        _ = filters.at['effective_wavelength', 'notaband']


def test_inventory(db, db_cs):
    good_query = '2MASS J00192626+4614078'
    d_results: dict = db.inventory(good_query)
    assert len(d_results)
    everything = Inventory(d_results, db_cs)
    assert all([hasattr(everything, s) for s in
                ('photometry', 'sources', 'names', 'spectra', 'ra', 'dec', 'propermotions')])


def test_find_colours(photometry, filters):
    all_photometry, all_bands = photometry
    df_photometry = find_colours(all_photometry, all_bands, filters)
    assert len(df_photometry)
    assert isinstance(df_photometry, pd.DataFrame)


def test_results_concat(photometry, sources, parallaxes, spectral_types):
    all_photometry, all_bands = photometry
    all_results, all_results_full = sources
    all_parallaxes, all_spectral_types = parallaxes, spectral_types
    wanted_mags = {'GAIA3.G', '2MASS.J', 'WISE.W1'}
    all_results_concat = results_concat(all_results_full, all_photometry, all_parallaxes, all_spectral_types, all_bands)
    assert all([col in all_results_concat.columns for col in ('ra_projected', 'dec_projected')])
    assert all([f'M_{band}' in all_results_concat.columns for band in all_bands if band in wanted_mags])


def test_one_df_query(db):
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

    # test one_df_query
    stringed_results = one_df_query(results[results.source.isin(ref_sources.source)].copy())
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


def test_multi_df_query(db, db_cs):
    bad_query = 'thisisabadquery'
    good_query = 'cruz'
    with pytest.raises(KeyError):
        results: Optional[dict] = db.search_string(bad_query, fmt='pandas', verbose=False)
        _ = results['Sources']
    results: Optional[dict] = db.search_string(good_query, fmt='pandas', verbose=False)
    assert isinstance(results, dict)
    assert 'Sources' in results
    assert isinstance(results['Sources'], pd.DataFrame)
    results_out = multi_df_query(results, db_cs, good_query)
    assert isinstance(results_out, dict)
    assert 'Sources' in results_out
    assert isinstance(results_out['Sources'], str)


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


def test_absolute_magnitudes():
    df = pd.DataFrame({
        'parallax': [100.0, -1.0],
        'GAIA3.G': [15.0, 18.0],
        '2MASS.J': [12.0, 14.0],
    })
    out = absolute_magnitudes(df, np.array(['GAIA3.G', '2MASS.J', 'NOT.A.BAND']))
    assert 'M_GAIA3.G' in out.columns
    assert 'M_2MASS.J' in out.columns
    assert np.isfinite(out.loc[0, 'M_GAIA3.G'])
    assert np.isnan(out.loc[1, 'M_GAIA3.G'])


def test_coordinate_project():
    df = pd.DataFrame({'ra': [10.0, 20.0, 30.0], 'dec': [-10.0, 0.0, 10.0]})
    ra_projected, dec_projected = coordinate_project(df)
    assert len(ra_projected) == len(df)
    assert len(dec_projected) == len(df)


def test_control_response():
    response = Response('ok')
    out = control_response(response, key='Sources', app_type='csv')
    assert 'attachment; filename=simplequery-' in out.headers['Content-Disposition']
    assert '_Sources.csv' in out.headers['Content-Disposition']


def test_get_header_filename():
    assert get_header_filename(None) == ''
    assert get_header_filename('attachment; filename="spec.fits"') == 'spec.fits'
    assert get_header_filename("attachment; filename*=UTF-8''spec%20file.fits") == 'spec file.fits'


def test_get_url_filename():
    assert get_url_filename('https://example.org/a/b/spectrum.fits') == 'spectrum.fits'
    assert get_url_filename('https://example.org/download/viewcontent') == 'viewcontent'


def test_unique_filename():
    existing = {'spec.fits', 'spec_1.fits'}
    assert unique_filename('new.fits', existing) == 'new.fits'
    assert unique_filename('spec.fits', existing) == 'spec_2.fits'


def test_csv_response():
    df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y, z']})
    out = csv_response(df, 'Sources')
    assert out.mimetype == 'text/csv'
    assert '_Sources.csv' in out.headers['Content-Disposition']
    assert out.get_data(as_text=True).splitlines() == ['a,b', '1,x', '2,"y, z"']


def test_write_multi_files():
    d_results = {
        'Sources': pd.DataFrame({'source': ['A']}),
        'Photometry': pd.DataFrame({'target': ['A'], 'band': ['2MASS.J']}),
    }
    zip_mem = write_multi_files(d_results)
    assert isinstance(zip_mem, BytesIO)
    with ZipFile(zip_mem, 'r') as zf:
        assert set(zf.namelist()) == {'Sources.csv', 'Photometry.csv'}

def test_propagate_coordinates():
    inventory = {'Sources': [{'source': 'A', 'ra': 10., 'dec': 0., 'epoch': None}],
                 'ProperMotions': [{'mu_ra': 0., 'mu_dec': 3600., 'adopted': True}]}  # 3.6 arcsec/yr north
    out = propagate_coordinates(inventory, '2010')
    assert out['from_epoch'] == 2000. and out['epoch'] == 2010.
    assert out['dec'] == pytest.approx(36 / 3600, abs=1e-6)  # 10 yr
    assert out['ra'] == pytest.approx(10.)
    assert not out['used_parallax'] and not out['used_radial_velocity']
    with pytest.raises(ValueError, match='No proper motion'):
        propagate_coordinates({'Sources': inventory['Sources']}, '2010')


def test_reference_handle(db, db_cs):
    pubs = db.query(db.Publications).pandas()
    with_bibcode = pubs[pubs.bibcode.fillna('') != ''].reference.iloc[0]
    out = reference_handle(pd.DataFrame({'reference': [with_bibcode, 'notareference']}), db_cs)
    assert out.reference[0].startswith('<a href="https://ui.adsabs.harvard.edu/abs/')
    assert out.reference[1] == 'notareference'


def test_parse_photometry_multi_source():
    df = pd.DataFrame({'source': ['A', 'A', 'A', 'B'], 'band': ['2MASS.J', '2MASS.J', 'WISE.W1', 'WISE.W1'],
                       'magnitude': [12., 13., 11., 10.]})
    out = parse_photometry(df, np.array(['2MASS.J', 'WISE.W1', '2MASS.H']), True)
    assert list(out.columns) == ['target', '2MASS.J', 'WISE.W1', '2MASS.H']
    assert out.loc[0, '2MASS.J'] == 12.  # first measurement of a band
    assert np.isnan(out.loc[1, '2MASS.J']) and np.isnan(out.loc[0, '2MASS.H'])


def test_propagate_coordinates_date():
    inventory = {'Sources': [{'source': 'A', 'ra': 10., 'dec': 0., 'epoch': 2000.}],
                 'ProperMotions': [{'mu_ra': 0., 'mu_dec': 3600., 'adopted': True}]}
    by_date = propagate_coordinates(inventory, '2010-01-01T12:00')
    assert by_date['epoch'] == pytest.approx(2010.0, abs=1e-3) and by_date['date'] == '2010-01-01 12:00'
    # J2010.0 is hours from that date: 3.6 arcsec/yr moves the source ~1e-6 deg in that time
    assert by_date['dec'] == pytest.approx(propagate_coordinates(inventory, '2010.0')['dec'], abs=1e-5)
    with pytest.raises(ValueError, match='decimal year'):
        propagate_coordinates(inventory, 'next tuesday')
