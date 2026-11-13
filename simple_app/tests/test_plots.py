"""
Testing the plot functions; fixtures are in conftest.py
"""
from ..plots import *

GOOD_QUERY = '2MASS J00192626+4614078'
BAD_QUERY = 'thisisabadquery'


def test_main_plots(plot_settings):
    night_sky_theme, js_callbacks = plot_settings
    assert isinstance(night_sky_theme, Theme)
    assert isinstance(js_callbacks, JSCallbacks)
    assert js_callbacks.dropdown_x_js  # callbacks parsed out of simple_callbacks.js


def test_multi_plot_bokeh(sources, photometry, parallaxes, spectral_types, modeled_parameters, plot_settings):
    all_photometry, all_bands = photometry
    night_sky_theme, js_callbacks = plot_settings
    script, div = multi_plot_bokeh(sources[1], all_bands, all_photometry, parallaxes, spectral_types,
                                   modeled_parameters, js_callbacks, night_sky_theme)
    assert isinstance(script, str) and isinstance(div, str)
    assert 'Physical Parameters' in script


def test_camd_plot(db, db_cs, plot_settings, photometry, sources, parallaxes, spectral_types, filters):
    all_photometry, all_bands = photometry
    night_sky_theme, js_callbacks = plot_settings
    for query, expect_plot in ((GOOD_QUERY, True), (BAD_QUERY, False)):
        everything = Inventory(db.inventory(query), db_cs)
        script, div = camd_plot(query, everything, all_bands, sources[1], parallaxes, spectral_types, filters,
                                all_photometry, js_callbacks, night_sky_theme, db_cs)
        assert isinstance(script, str) == expect_plot
        assert isinstance(div, str) == expect_plot


def test_name_simplifier():
    assert name_simplifier('GAIA3.G-2MASS.J-WISE.W1') == 'G - J - W1'


def test_multi_plot_axes_are_colours(sources, photometry, parallaxes, spectral_types, modeled_parameters,
                                     plot_settings):
    # band names such as PAN-STARRS/PS1.i contain '-' but are not colours
    all_photometry, all_bands = photometry
    night_sky_theme, js_callbacks = plot_settings
    script = multi_plot_bokeh(sources[1], all_bands, all_photometry, parallaxes, spectral_types,
                              modeled_parameters, js_callbacks, night_sky_theme)[0]
    assert 'PAN-STARRS' not in str(re.findall(r'"options":\[.*?\]\]', script))
    assert '"options":[["GAIA3.G-GAIA3.Grp"' in script
