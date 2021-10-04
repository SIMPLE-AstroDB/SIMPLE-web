"""
File containing the 'workhorse' functions generating the various plots seen on the website
"""
# external packages
import numpy as np
import pandas as pd
from astropy.table import Table
from bokeh.embed import components
from bokeh.layouts import row, column  # bokeh displaying nicely
from bokeh.models import ColumnDataSource, Range1d, CustomJS,\
    Select, Toggle, TapTool, OpenURL, HoverTool  # bokeh models
from bokeh.palettes import Colorblind8
from bokeh.plotting import figure  # bokeh plotting
from bokeh.themes import built_in_themes, Theme
from specutils import Spectrum1D
# internal packages
import sys
# local packages
sys.path.append('.')
from simple_app.utils import *
from simple_app.simple_callbacks import JSCallbacks


def specplot(query: str, db_file: str, nightskytheme: Theme):
    """
    Creates the bokeh representation of the plot

    Parameters
    ----------
    query: str
        The object that has been searched for
    db_file: str
        The connection string of the database
    nightskytheme: Theme
        The bokeh theme

    Returns
    -------
    script: str
        script for creating the bokeh plot
    div: str
        the html to be inserted in dom
    """
    def normalise(fluxarr: np.ndarray) -> np.ndarray:
        fluxarr = (fluxarr - np.nanmin(fluxarr)) /\
                  (np.nanmax(fluxarr) - np.nanmin(fluxarr))
        return fluxarr

    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    tspec: Table = db.query(db.Spectra).\
        filter(db.Spectra.c.source == query).\
        table(spectra=['spectrum'])  # query the database for the spectra
    if not len(tspec):  # if there aren't any spectra, return nothing
        return None, None
    p = figure(title='Spectra', plot_height=500,
               active_scroll='wheel_zoom', active_drag='box_zoom',
               tools='pan,wheel_zoom,box_zoom,reset', toolbar_location='left',
               sizing_mode='stretch_width')  # init figure
    p.xaxis.axis_label_text_font_size = '1.5em'
    p.yaxis.axis_label_text_font_size = '1.5em'
    p.xaxis.major_label_text_font_size = '1.5em'
    p.yaxis.major_label_text_font_size = '1.5em'
    p.title.text_font_size = '2em'
    normfact, ld = None, 'solid'
    for i, spec in enumerate(tspec):  # over all spectra
        spectrum: Spectrum1D = spec['spectrum']  # spectrum as an object
        wave: np.ndarray = spectrum.spectral_axis.value  # unpack wavelengths
        flux: np.ndarray = spectrum.flux.value  # unpack fluxes
        label = f'{spec["telescope"]}-{spec["instrument"]}: {spec["observation_date"].date()}'  # legend label
        if not i:  # first spectra
            p.xaxis.axis_label = 'Wavelength [μm]'  # units for wavelength on x axis
            p.yaxis.axis_label = 'Normalised Flux'  # units for wavelength on y axis
            flux = normalise(flux)  # normalise the flux by the sum
        if j := i > len(Colorblind8):  # loop around colours if we have more than 8 spectra, and start line dashing
            j = 0
            ld = 'dashed'
        p.line(x=wave, y=flux, legend_label=label, line_color=Colorblind8[j], line_dash=ld)  # create line plot
    p.legend.click_policy = 'hide'  # hide the graph if clicked on
    p.legend.label_text_font_size = '1.5em'
    script, div = components(p, theme=nightskytheme)  # convert bokeh plot into script and div for html us
    return script, div


def multiplotbokeh(all_results_full: pd.DataFrame, all_bands: np.ndarray,
                   all_photo: pd.DataFrame, all_plx: pd.DataFrame, jscallbacks: JSCallbacks, nightskytheme: Theme):
    """
    The workhorse generating the multiple plots view page

    Parameters
    ----------
    all_results_full: pd.DataFrame
        Every object and its basic information
    all_bands: np.ndarray
        All the photometric bands for colour-colour
    all_photo: pd.DataFrame
        All the photometry
    all_plx: pd.DataFrame
        All the parallaxes
    jscallbacks: JSCallbacks
        The javascript callbacks for bokeh
    nightskytheme: Theme
        The bokeh theme

    Returns
    -------
    script: str
        script for creating the bokeh plot
    div: str
        the html to be inserted in dom
    """
    all_results_mostfull = results_concat(all_results_full, all_photo, all_plx, all_bands)
    all_results_mostfull.dropna(axis=1, how='all', inplace=True)
    fullcds = ColumnDataSource(all_results_mostfull)  # convert to CDS
    tooltips = [('Target', '@source')]  # tooltips for hover tool
    # sky plot
    thishover = HoverTool(names=['circle', ], tooltips=tooltips)  # hovertool
    thistap = TapTool(names=['circle', ])  # taptool
    psky = figure(title='Sky Plot', plot_height=500,
                  active_scroll='wheel_zoom', active_drag='box_zoom',
                  tools='pan,wheel_zoom,box_zoom,box_select,reset',
                  sizing_mode='stretch_width', x_range=[-180, 180], y_range=[-90, 90])  # bokeh figure
    psky.add_tools(thishover)  # add hover tool to plot
    psky.add_tools(thistap)  # add tap tool to plot
    psky.ellipse(x=0, y=0, width=360, height=180, color='lightgrey', name='background')  # background ellipse
    psky.circle(source=fullcds, x='raproj', y='decproj', size=6, name='circle')
    psky.xaxis.axis_label_text_font_size = '1.5em'
    psky.yaxis.axis_label_text_font_size = '1.5em'
    psky.xaxis.major_label_text_font_size = '1.5em'
    psky.yaxis.major_label_text_font_size = '1.5em'
    psky.title.text_font_size = '2em'
    thistap.callback = OpenURL(url='/solo_result/@source')  # open new page on target when source tapped
    # colour-colour
    pcc = figure(title='Colour-Colour', plot_height=500,
                 active_scroll='wheel_zoom', active_drag='box_zoom',
                 tools='pan,wheel_zoom,box_zoom,box_select,hover,tap,reset', tooltips=tooltips,
                 sizing_mode='stretch_width')  # bokeh figure
    colbands = [col for col in all_results_mostfull.columns if '-' in col]
    just_colours = all_results_mostfull.loc[:, colbands].copy()
    xfullname = just_colours.columns[0]
    yfullname = just_colours.columns[1]
    xvisname = xfullname.replace('-', ' - ')
    yvisname = yfullname.replace('-', ' - ')
    fullplot = pcc.circle(x=xfullname, y=yfullname, source=fullcds, size=5)  # plot all objects
    pcc.x_range = Range1d(all_results_mostfull[xfullname].min(), all_results_mostfull[xfullname].max())  # x
    pcc.y_range = Range1d(all_results_mostfull[yfullname].min(), all_results_mostfull[yfullname].max())  # y
    pcc.xaxis.axis_label = xvisname  # x label
    pcc.yaxis.axis_label = yvisname  # y label
    pcc.xaxis.axis_label_text_font_size = '1.5em'
    pcc.yaxis.axis_label_text_font_size = '1.5em'
    pcc.xaxis.major_label_text_font_size = '1.5em'
    pcc.yaxis.major_label_text_font_size = '1.5em'
    pcc.title.text_font_size = '2em'
    taptool = pcc.select(type=TapTool)  # tapping
    taptool.callback = OpenURL(url='/solo_result/@source')  # open new page on target when source tapped
    buttonxflip = Toggle(label='X Flip')
    buttonxflip.js_on_click(CustomJS(code=jscallbacks.button_flip, args={'axrange': pcc.x_range}))
    buttonyflip = Toggle(label='Y Flip')
    buttonyflip.js_on_click(CustomJS(code=jscallbacks.button_flip, args={'axrange': pcc.y_range}))
    axis_names = [col.replace('-', ' - ') for col in just_colours.columns]  # convert nicely
    dropmenu = [*zip(just_colours.columns, axis_names), ]  # zip up into menu
    dropdownx = Select(options=dropmenu, value=xfullname)  # x axis select
    dropdownx.js_on_change('value', CustomJS(code=jscallbacks.dropdownx_js,
                                             args={'fullplot': fullplot,
                                                   'fulldata': fullcds.data, 'xbut': buttonxflip,
                                                   'xaxis': pcc.xaxis[0], 'xrange': pcc.x_range}))
    dropdowny = Select(options=dropmenu, value=yfullname)  # y axis select
    dropdowny.js_on_change('value', CustomJS(code=jscallbacks.dropdowny_js,
                                             args={'fullplot': fullplot,
                                                   'fulldata': fullcds.data, 'ybut': buttonyflip,
                                                   'yaxis': pcc.yaxis[0], 'yrange': pcc.y_range}))
    # colour absolute magnitude diagram
    just_mags: pd.DataFrame = all_results_mostfull[all_bands]
    absmagnames = ["M_" + col for col in just_mags.columns]
    dropmenumag = [*zip(absmagnames, absmagnames)]
    pcamd = figure(title='Colour-Absolute Magnitude Diagram', plot_height=500,
                   active_scroll='wheel_zoom', active_drag='box_zoom',
                   tools='pan,wheel_zoom,box_zoom,box_select,hover,tap,reset', tooltips=tooltips,
                   sizing_mode='stretch_width')  # bokeh figure
    yfullname = absmagnames[0]
    fullmagplot = pcamd.circle(x=xfullname, y=yfullname, source=fullcds, size=5)  # plot all objects
    pcamd.x_range = Range1d(all_results_mostfull[xfullname].min(), all_results_mostfull[xfullname].max())  # x
    pcamd.y_range = Range1d(all_results_mostfull[yfullname].max(), all_results_mostfull[yfullname].min())  # y limits
    pcamd.xaxis.axis_label = xvisname  # x label
    pcamd.yaxis.axis_label = yfullname  # y label
    pcamd.xaxis.axis_label_text_font_size = '1.5em'
    pcamd.yaxis.axis_label_text_font_size = '1.5em'
    pcamd.xaxis.major_label_text_font_size = '1.5em'
    pcamd.yaxis.major_label_text_font_size = '1.5em'
    pcamd.title.text_font_size = '2em'
    taptoolmag = pcamd.select(type=TapTool)  # tapping
    taptoolmag.callback = OpenURL(url='/solo_result/@source')  # open new page on target when source tapped
    buttonmagxflip = Toggle(label='X Flip')
    buttonmagxflip.js_on_click(CustomJS(code=jscallbacks.button_flip, args={'axrange': pcamd.x_range}))
    buttonmagyflip = Toggle(label='Y Flip')
    buttonmagyflip.js_on_click(CustomJS(code=jscallbacks.button_flip, args={'axrange': pcamd.y_range}))
    dropdownmagx = Select(options=dropmenu, value=xfullname)  # x axis
    dropdownmagx.js_on_change('value', CustomJS(code=jscallbacks.dropdownx_js,
                                                args={'fullplot': fullmagplot,
                                                      'fulldata': fullcds.data, 'xbut': buttonmagxflip,
                                                      'xaxis': pcamd.xaxis[0], 'xrange': pcamd.x_range}))
    dropdownmagy = Select(options=dropmenumag, value=yfullname)  # y axis
    dropdownmagy.js_on_change('value', CustomJS(code=jscallbacks.dropdowny_js,
                                                args={'fullplot': fullmagplot,
                                                      'fulldata': fullcds.data, 'ybut': buttonmagyflip,
                                                      'yaxis': pcamd.yaxis[0], 'yrange': pcamd.y_range}))
    plots = column(psky,
                   row(column(pcc,
                              row(dropdownx, dropdowny,
                                  sizing_mode='stretch_width'),
                              row(buttonxflip, buttonyflip,
                                  sizing_mode='stretch_width'),
                              sizing_mode='scale_width'),
                       column(pcamd,
                              row(dropdownmagx, dropdownmagy,
                                  sizing_mode='stretch_width'),
                              row(buttonmagxflip, buttonmagyflip,
                                  sizing_mode='stretch_width'),
                              sizing_mode='scale_width'),
                       sizing_mode='scale_width'),
                   sizing_mode='scale_width')
    script, div = components(plots, theme=nightskytheme)
    return script, div


def camdplot(query: str, everything: Inventory, all_bands: np.ndarray,
             all_results_full: pd.DataFrame, all_plx: pd.DataFrame, photfilters: pd.DataFrame,
             all_photo: pd.DataFrame, jscallbacks: JSCallbacks, nightskytheme: Theme):
    """
    Creates CAMD plot as JSON object

    Parameters
    ----------
    query: str
        The object that has been searched for
    everything: Inventory
        The class representation wrapping db.inventory
    all_bands: np.ndarray
        All of the photometric bands for colour-colour
    all_results_full: pd.DataFrame
        Every object and its basic information
    all_plx: pd.DataFrame
        All of the parallaxes
    photfilters: pd.DataFrame
        All of the filters to check
    all_photo: pd.DataFrame
        All of the photometry
    jscallbacks: JSCallbacks
        The javascript callbacks for bokeh
    nightskytheme: Theme
        The theme for bokeh

    Returns
    -------
    script: str
        script for creating the bokeh plot
    div: str
        the html to be inserted in dom
    """
    tooltips = [('Target', '@target')]
    p = figure(plot_height=500,
               active_scroll='wheel_zoom', active_drag='box_zoom',
               tools='pan,wheel_zoom,box_zoom,hover,tap,reset', tooltips=tooltips,
               sizing_mode='stretch_width')  # bokeh figure
    try:
        thisphoto: pd.DataFrame = everything.listconcat('Photometry', False)  # the photometry for this object
    except KeyError:  # no photometry for this object
        return None, None
    thisphoto = parse_photometry(thisphoto, all_bands)
    thisbands: np.ndarray = np.unique(thisphoto.columns)  # the columns
    thisphoto: pd.DataFrame = find_colours(thisphoto, thisbands, photfilters)  # get the colours
    thisphoto['target'] = query
    try:
        thisplx: pd.DataFrame = everything.listconcat('Parallaxes', False)  # try to grab parallaxes
    except KeyError:  # don't worry if they're not there
        pass
    else:  # if they are though...
        thisphoto['parallax'] = thisplx['parallax'].iloc[0]  # grab the first parallax (adopted might be better if key)
        thisphoto = absmags(thisphoto, thisbands)  # get abs mags
    thisphoto.dropna(axis=1, how='all', inplace=True)
    colbands = [col for col in thisphoto.columns if any([colcheck in col for colcheck in ('-', 'M_')])]
    just_colours = thisphoto.loc[:, colbands].copy()  # cut dataframe to just colour and abs mags
    xfullname = just_colours.columns[0]
    yfullname = just_colours.columns[1]
    xvisname = xfullname.replace('-', ' - ')
    yvisname = yfullname.replace('-', ' - ')
    thiscds = ColumnDataSource(data=thisphoto)  # this object cds
    thisplot = p.circle(x=xfullname, y=yfullname, source=thiscds,
                        color='blue', size=10)  # plot for this object
    all_results_mostfull = results_concat(all_results_full, all_photo, all_plx, thisbands)
    all_results_mostfull.dropna(axis=1, how='all', inplace=True)
    cdsfull = ColumnDataSource(data=all_results_mostfull)  # bokeh cds object
    fullplot = p.circle_x(x=xfullname, y=yfullname, source=cdsfull,
                          color='gray', alpha=0.5, size=5)  # plot all objects
    fullplot.level = 'underlay'  # put full plot underneath this plot
    p.x_range = Range1d(all_results_mostfull[xfullname].min(), all_results_mostfull[xfullname].max())  # x limits
    p.y_range = Range1d(all_results_mostfull[yfullname].min(), all_results_mostfull[yfullname].max())  # y limits
    p.xaxis.axis_label = xvisname  # x label
    p.yaxis.axis_label = yvisname  # y label
    p.xaxis.axis_label_text_font_size = '1.5em'
    p.yaxis.axis_label_text_font_size = '1.5em'
    p.xaxis.major_label_text_font_size = '1.5em'
    p.yaxis.major_label_text_font_size = '1.5em'
    taptool = p.select(type=TapTool)  # tapping
    taptool.callback = OpenURL(url='/solo_result/@target')  # open new page on target when source tapped
    buttonxflip = Toggle(label='X Flip')
    buttonxflip.js_on_click(CustomJS(code=jscallbacks.button_flip, args={'axrange': p.x_range}))
    buttonyflip = Toggle(label='Y Flip')
    buttonyflip.js_on_click(CustomJS(code=jscallbacks.button_flip, args={'axrange': p.y_range}))
    axis_names = [col.replace('-', ' - ') for col in just_colours.columns]  # convert nicely
    dropmenu = [*zip(just_colours.columns, axis_names), ]  # zip up into menu
    dropdownx = Select(options=dropmenu, value=xfullname)  # x axis select
    dropdownx.js_on_change('value', CustomJS(code=jscallbacks.dropdownx_js,
                                             args={'fullplot': fullplot, 'thisplot': thisplot,
                                                   'fulldata': cdsfull.data, 'xbut': buttonxflip,
                                                   'xaxis': p.xaxis[0], 'xrange': p.x_range}))
    dropdowny = Select(options=dropmenu, value=yfullname)  # y axis select
    dropdowny.js_on_change('value', CustomJS(code=jscallbacks.dropdowny_js,
                                             args={'fullplot': fullplot, 'thisplot': thisplot,
                                                   'fulldata': cdsfull.data, 'ybut': buttonyflip,
                                                   'yaxis': p.yaxis[0], 'yrange': p.y_range}))
    plots = column(p, row(dropdownx,
                          dropdowny,
                          buttonxflip,
                          buttonyflip,
                          sizing_mode='scale_width'),
                   sizing_mode='scale_width')
    script, div = components(plots, theme=nightskytheme)
    return script, div


def mainplots():
    _nightskytheme = built_in_themes['night_sky']  # darker theme for bokeh
    _jscallbacks = JSCallbacks()  # grab the callbacks for bokeh interactivity
    return _nightskytheme, _jscallbacks


if __name__ == '__main__':
    ARGS, DB_FILE, PHOTOMETRIC_FILTERS, ALL_RESULTS, ALL_RESULTS_FULL, ALL_PHOTO, ALL_BANDS, ALL_PLX = mainutils()
    NIGHTSKYTHEME, JSCALLBACKS = mainplots()
