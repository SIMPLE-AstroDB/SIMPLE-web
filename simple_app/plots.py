"""
File containing the 'workhorse' functions generating the various plots seen on the website
"""
# external packages
from astropy.table import Table
from bokeh.embed import components
from bokeh.layouts import row, column  # bokeh displaying nicely
from bokeh.models import ColumnDataSource, Range1d, CustomJS,\
    Select, Toggle, TapTool, OpenURL, HoverTool  # bokeh models
from bokeh.palettes import Colorblind8
from bokeh.plotting import figure  # bokeh plotting
from bokeh.themes import built_in_themes
from specutils import Spectrum1D
# local packages
from utils import *


def specplot(query: str):
    """
    Creates the bokeh representation of the plot

    Parameters
    ----------
    query: str
        The object that has been searched for

    Returns
    -------
    script
        script for creating the bokeh plot
    div
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
    # TODO: Overplotting standards
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


def multiplotbokeh():
    """
    The workhorse generating the multiple plots view page

    Returns
    -------
    script
        script for creating the bokeh plot
    div
        the html to be inserted in dom
    """
    raproj, decproj = coordinate_project()  # project coordinates to galactic
    all_results_full['raproj'] = raproj  # ra
    all_results_full['decproj'] = decproj  # dec
    all_results_full_cut: pd.DataFrame = all_results_full[['source', 'raproj', 'decproj']]  # cut dataframe
    all_results_mostfull: pd.DataFrame = pd.merge(all_results_full_cut, all_photo,
                                                  left_on='source', right_on='target', how='left')
    all_results_mostfull = pd.merge(all_results_mostfull, all_plx, on='source', how='left')
    all_results_mostfull = absmags(all_results_mostfull)  # find the absolute mags
    fullcds = ColumnDataSource(all_results_mostfull)  # convert to CDS
    bands = [band.split("_")[1] for band in all_bands]  # nice band names
    vals = [f'@{band}' for band in all_bands]  # the values in CDS
    tooltips = [('Target', '@source'), *zip(bands, vals), ('Ref', '@ref')]  # tooltips for hover tool
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
    fullplot = pcc.circle(x='WISE_W1_WISE_W2', y='WISE_W3_WISE_W4', source=fullcds, size=5)  # plot all objects
    pcc.x_range = Range1d(all_results_mostfull.WISE_W1_WISE_W2.min(), all_results_mostfull.WISE_W1_WISE_W2.max())  # x
    pcc.y_range = Range1d(all_results_mostfull.WISE_W3_WISE_W4.min(), all_results_mostfull.WISE_W3_WISE_W4.max())  # y
    pcc.xaxis.axis_label = 'W1 - W2'  # x label
    pcc.yaxis.axis_label = 'W3 - W4'  # y label
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
    just_colours: pd.DataFrame = all_photo.drop(columns=np.hstack([all_bands, ['ref', 'target']]))  # cols
    axis_names = [f'{col.split("_")[1]} - {col.split("_")[3]}' for col in just_colours.columns]  # convert nicely
    dropmenu = [*zip(just_colours.columns, axis_names), ]  # zip up into menu
    dropdownx = Select(options=dropmenu, value='WISE_W1_WISE_W2')  # x axis select
    dropdownx.js_on_change('value', CustomJS(code=jscallbacks.dropdownx_js,
                                             args={'fullplot': fullplot,
                                                   'fulldata': fullcds.data, 'xbut': buttonxflip,
                                                   'xaxis': pcc.xaxis[0], 'xrange': pcc.x_range}))
    dropdowny = Select(options=dropmenu, value='WISE_W3_WISE_W4')  # y axis select
    dropdowny.js_on_change('value', CustomJS(code=jscallbacks.dropdowny_js,
                                             args={'fullplot': fullplot,
                                                   'fulldata': fullcds.data, 'ybut': buttonyflip,
                                                   'yaxis': pcc.yaxis[0], 'yrange': pcc.y_range}))
    # colour absolute magnitude diagram
    just_mags: pd.DataFrame = all_photo[all_bands]
    magaxisnames = [col.split("_")[1] for col in just_mags.columns]
    absmagnames = ["M_" + col for col in just_mags.columns]
    dropmenumag = [*zip(absmagnames, magaxisnames)]
    pcamd = figure(title='Colour-Absolute Magnitude Diagram', plot_height=500,
                   active_scroll='wheel_zoom', active_drag='box_zoom',
                   tools='pan,wheel_zoom,box_zoom,box_select,hover,tap,reset', tooltips=tooltips,
                   sizing_mode='stretch_width')  # bokeh figure
    fullmagplot = pcamd.circle(x='WISE_W1_WISE_W2', y='M_WISE_W1', source=fullcds, size=5)  # plot all objects
    pcamd.x_range = Range1d(all_results_mostfull.WISE_W1_WISE_W2.min(), all_results_mostfull.WISE_W1_WISE_W2.max())  # x
    pcamd.y_range = Range1d(all_results_mostfull.M_WISE_W1.max(), all_results_mostfull.M_WISE_W1.min())  # y limits
    pcamd.xaxis.axis_label = 'W1 - W2'  # x label
    pcamd.yaxis.axis_label = 'W1'  # y label
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
    dropdownmagx = Select(options=dropmenu, value='WISE_W1_WISE_W2')  # x axis
    dropdownmagx.js_on_change('value', CustomJS(code=jscallbacks.dropdownx_js,
                                                args={'fullplot': fullmagplot,
                                                      'fulldata': fullcds.data, 'xbut': buttonmagxflip,
                                                      'xaxis': pcamd.xaxis[0], 'xrange': pcamd.x_range}))
    dropdownmagy = Select(options=dropmenumag, value='M_WISE_W1')  # y axis
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


def camdplot(query: str, everything: Inventory):
    """
    Creates CAMD plot as JSON object

    Parameters
    ----------
    query: str
        The object that has been searched for
    everything: Inventory
        The class representation wrapping db.inventory

    Returns
    -------
    script
        script for creating the bokeh plot
    div
        the html to be inserted in dom
    """
    # TODO: Add CAMD diagram when we have data to test this on (i.e. parallaxes + photometry for same object)
    bands = [band.split("_")[1] for band in all_bands]  # nice band names
    vals = [f'@{band}' for band in all_bands]  # the values in CDS
    tooltips = [('Target', '@target'), *zip(bands, vals), ('Ref', '@ref')]  # tooltips for hover tool
    p = figure(title='Colour-Colour', plot_height=500,
               active_scroll='wheel_zoom', active_drag='box_zoom',
               tools='pan,wheel_zoom,box_zoom,hover,tap,reset', tooltips=tooltips,
               sizing_mode='stretch_width')  # bokeh figure
    try:
        thisphoto: pd.DataFrame = everything.listconcat('Photometry', False)  # the photometry for this object
    except KeyError:  # no photometry for this object
        return None, None
    newphoto: dict = parse_photometry(thisphoto, all_bands)  # transpose photometric table
    newphoto['target'] = [query, ] * len(newphoto['ref'])  # the targetname
    thisphoto = pd.DataFrame(newphoto)  # turn into dataframe
    thisbands: np.ndarray = np.unique(thisphoto.columns)  # the columns
    thisbands = thisbands[np.isin(thisbands, all_bands)]  # the bands for this object
    thisphoto: pd.DataFrame = find_colours(thisphoto, thisbands)  # get the colours
    # FIXME: This'll break in the circumstance that there is only one band (shouldn't happen)
    xfullname = '_'.join(thisbands[0:2])  # x axis colour
    xvisname = thisbands[0].split('_')[-1] + ' - ' + thisbands[1].split('_')[-1]  # x axis name
    try:
        yfullname = '_'.join(thisbands[2:4])  # y axis colour
        yvisname = thisbands[2].split('_')[-1] + ' - ' + thisbands[3].split('_')[-1]  # y axis name
    except IndexError:
        yfullname = '_'.join(thisbands[0:2])  # y axis name if only one colour
        yvisname = thisbands[0].split('_')[-1] + ' - ' + thisbands[1].split('_')[-1]  # y axis name
    thiscds = ColumnDataSource(data=thisphoto)  # this object cds
    thisplot = p.circle(x=xfullname, y=yfullname, source=thiscds,
                        color='blue', size=10)  # plot for this object
    cdsfull = ColumnDataSource(data=all_photo)  # bokeh cds object
    fullplot = p.circle_x(x=xfullname, y=yfullname, source=cdsfull,
                          color='gray', alpha=0.5, size=5)  # plot all objects
    fullplot.level = 'underlay'  # put full plot underneath this plot
    p.x_range = Range1d(all_photo[xfullname].min(), all_photo[xfullname].max())  # x limits
    p.y_range = Range1d(all_photo[yfullname].min(), all_photo[yfullname].max())  # y limits
    p.xaxis.axis_label = xvisname  # x label
    p.yaxis.axis_label = yvisname  # y label
    p.xaxis.axis_label_text_font_size = '1.5em'
    p.yaxis.axis_label_text_font_size = '1.5em'
    p.xaxis.major_label_text_font_size = '1.5em'
    p.yaxis.major_label_text_font_size = '1.5em'
    p.title.text_font_size = '2em'
    taptool = p.select(type=TapTool)  # tapping
    taptool.callback = OpenURL(url='/solo_result/@target')  # open new page on target when source tapped
    buttonxflip = Toggle(label='X Flip')
    buttonxflip.js_on_click(CustomJS(code=jscallbacks.button_flip, args={'axrange': p.x_range}))
    buttonyflip = Toggle(label='Y Flip')
    buttonyflip.js_on_click(CustomJS(code=jscallbacks.button_flip, args={'axrange': p.y_range}))
    if thisbands is all_bands:
        whichdf = all_photo
    else:
        whichdf = thisphoto
    just_colours: pd.DataFrame = whichdf.drop(columns=np.hstack([thisbands, ['ref', 'target']]))  # only the colours
    axis_names = [f'{col.split("_")[1]} - {col.split("_")[3]}' for col in just_colours.columns]  # convert nicely
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


nightskytheme = built_in_themes['night_sky']  # nicer looking bokeh