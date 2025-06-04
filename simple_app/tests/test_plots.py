"""
Testing the plot functions
"""
# local packages
from ..plots import *
from .test_utils import *


@pytest.fixture(scope='session')
def test_main_plots():
    night_sky_theme, js_callbacks = main_plots()
    assert type(night_sky_theme) == Theme
    assert isinstance(js_callbacks, JSCallbacks)
    return night_sky_theme, js_callbacks


def test_multi_plot_bokeh(db, test_get_all_sources, test_get_all_photometry, test_get_all_parallaxes,
                          test_get_all_spectral_types, test_main_plots):
    assert db
    all_photometry, all_bands = test_get_all_photometry
    all_results, all_results_full = test_get_all_sources
    all_parallaxes = test_get_all_parallaxes
    all_spectral_types = test_get_all_spectral_types
    night_sky_theme, js_callbacks = test_main_plots
    script, div = multi_plot_bokeh(all_results_full, all_bands, all_photometry, all_parallaxes, all_spectral_types,
                                   js_callbacks, night_sky_theme)
    assert all([type(s) == str for s in (script, div)])
    return


def test_spectra_plot(db, test_main_plots):
    assert db
    night_sky_theme, js_callbacks = test_main_plots
    good_query = '2MASS J00192626+4614078'
    bad_query = 'thisisabadquery'
    good_script, good_div = spectra_plot(good_query, db_cs, night_sky_theme, js_callbacks)[:2]
    bad_script, bad_div = spectra_plot(bad_query, db_cs, night_sky_theme, js_callbacks)[:2]
    assert all([type(s) == str for s in (good_script, good_div)])
    assert all([s is None for s in (bad_script, bad_div)])
    return


def test_camd_plot(db, test_main_plots, test_get_all_photometry, test_get_all_sources,
                   test_get_all_parallaxes, test_get_all_spectral_types, test_get_filters):
    assert db
    all_photometry, all_bands = test_get_all_photometry
    photometric_filters = test_get_filters
    args = argparse.ArgumentParser()
    args = args.parse_args([])
    args.debug = False
    good_query = '2MASS J00192626+4614078'
    bad_query = 'thisisabadquery'
    night_sky_theme, js_callbacks = test_main_plots
    d_result: dict = db.inventory(good_query)
    good_everything = Inventory(d_result, db_cs)
    d_result = db.inventory(bad_query)
    bad_everything = Inventory(d_result, db_cs)
    all_results, all_resultsfull = test_get_all_sources
    all_parallaxes = test_get_all_parallaxes
    all_spectral_types = test_get_all_spectral_types
    good_script, good_div = camd_plot(good_query, good_everything, all_bands, all_resultsfull,
                                      all_parallaxes, all_spectral_types, photometric_filters,
                                      all_photometry, js_callbacks, night_sky_theme, db_cs)
    bad_script, bad_div = camd_plot(bad_query, bad_everything, all_bands, all_resultsfull,
                                    all_parallaxes, all_spectral_types, photometric_filters,
                                    all_photometry, js_callbacks, night_sky_theme, db_cs)
    assert all([type(s) == str for s in (good_script, good_div)])
    assert all([s is None for s in (bad_script, bad_div)])
    return
