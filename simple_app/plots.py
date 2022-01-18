"""
File containing the 'workhorse' functions generating the various plots seen on the website
"""
import sys
# local packages

sys.path.append('.')
from simple_app.utils import *

# feature labels taken from splat
FEATURE_LABELS = {
    'h2o': {'altname': [], 'label': r'H₂O', 'type': 'band',
            'wavelengths': [[0.925, 0.95], [1.08, 1.20], [1.325, 1.550],
                            [1.72, 2.14]]},
    'ch4': {'altname': [], 'label': r'CH₄', 'type': 'band',
            'wavelengths': [[1.1, 1.24], [1.28, 1.44], [1.6, 1.76],
                            [2.2, 2.35]]},
    'co': {'altname': [], 'label': r'CO', 'type': 'band', 'wavelengths': [[2.29, 2.39]]},
    'tio': {'altname': [], 'label': r'TiO', 'type': 'band',
            'wavelengths': [[0.6569, 0.6852], [0.705, 0.727], [0.76, 0.80],
                            [0.825, 0.831], [0.845, 0.86]]},
    'vo': {'altname': [], 'label': r'VO', 'type': 'band', 'wavelengths': [[1.04, 1.08]]},
    'young vo': {'altname': [], 'label': r'VO', 'type': 'band', 'wavelengths': [[1.17, 1.20]]},
    'cah': {'altname': [], 'label': r'CaH', 'type': 'band',
            'wavelengths': [[0.6346, 0.639], [0.675, 0.705]]},
    'crh': {'altname': [], 'label': r'CrH', 'type': 'band', 'wavelengths': [[0.8611, 0.8681]]},
    'feh': {'altname': [], 'label': r'FeH', 'type': 'band',
            'wavelengths': [[0.8692, 0.875], [0.98, 1.03], [1.19, 1.25],
                            [1.57, 1.64]]},
    'h2': {'altname': ['cia h2'], 'label': r'H₂', 'type': 'band', 'wavelengths': [[1.5, 2.4]]},
    'sb': {'altname': ['binary', 'lt binary', 'spectral binary'], 'label': r'*', 'type': 'band',
           'wavelengths': [[1.6, 1.64]]},
    'h': {'altname': ['hi', 'h1'], 'label': r'H I', 'type': 'line',
          'wavelengths': [[1.004, 1.005], [1.093, 1.094], [1.281, 1.282],
                          [1.944, 1.945], [2.166, 2.166]]},
    'na': {'altname': ['nai', 'na1'], 'label': r'Na I', 'type': 'line',
           'wavelengths': [[0.8186, 0.8195], [1.136, 1.137], [2.206, 2.209]]},
    'cs': {'altname': ['csi', 'cs1'], 'label': r'Cs I', 'type': 'line',
           'wavelengths': [[0.8521, 0.8521], [0.8943, 0.8943]]},
    'rb': {'altname': ['rbi', 'rb1'], 'label': r'Rb I', 'type': 'line',
           'wavelengths': [[0.78, 0.78], [0.7948, 0.7948]]},
    'mg': {'altname': ['mgi', 'mg1'], 'label': r'Mg I', 'type': 'line',
           'wavelengths': [[1.7113336, 1.7113336], [1.5745017, 1.5770150],
                           [1.4881595, 1.4881847, 1.5029098, 1.5044356], [1.1831422, 1.2086969]]},
    'ca': {'altname': ['cai', 'ca1'], 'label': r'Ca I', 'type': 'line',
           'wavelengths': [[0.6573, 0.6573], [2.263110, 2.265741],
                           [1.978219, 1.985852, 1.986764], [1.931447, 1.945830, 1.951105]]},
    'caii': {'altname': ['ca2'], 'label': r'Ca II', 'type': 'line',
             'wavelengths': [[1.184224, 1.195301], [0.985746, 0.993409]]},
    'al': {'altname': ['ali', 'al1'], 'label': r'Al I', 'type': 'line',
           'wavelengths': [[1.672351, 1.675511], [1.3127006, 1.3154345]]},
    'fe': {'altname': ['fei', 'fe1'], 'label': r'Fe I', 'type': 'line',
           'wavelengths': [[1.5081407, 1.5494570], [1.25604314, 1.28832892],
                           [1.14254467, 1.15967616, 1.16107501, 1.16414462, 1.16931726, 1.18860965, 1.18873357,
                            1.19763233]]},
    'k': {'altname': ['ki', 'k1'], 'label': r'K I', 'type': 'line',
          'wavelengths': [[0.7699, 0.7665], [1.169, 1.177], [1.244, 1.252]]},
}


def specplot(query: str, db_file: str,
             nightskytheme: Theme, jscallbacks: JSCallbacks) -> Tuple[Optional[str], Optional[str],
                                                                      Optional[int], Optional[str]]:
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
    jscallbacks: JSCallbacks
        An instance containing the javascript as strings

    Returns
    -------
    script: str
        script for creating the bokeh plot
    div: str
        the html to be inserted in dom
    nfail: int
        the number of failed spectra to be loaded
    failstr: str
        the failed spectra
    """
    def normalise() -> np.ndarray:
        """
        Normalises the flux using the wave & flux variables in the surrounding scope
        """
        wavestart, waveend = wave[0], wave[-1]  # start and end points of wavelength array
        minwave, maxwave = 0.81, 0.82  # normalisation region bounds
        objminwave, objmaxwave = np.clip([minwave, maxwave], wavestart, waveend)  # clipping bounds on wavelength
        if np.isclose(objminwave, waveend):  # if clipped by end of wavelength
            objminwave -= 0.01  # shift minimum down
        if np.isclose(objmaxwave, wavestart):  # if clipped by start of wavelength
            objmaxwave += 0.01  # shift maximum up
        fluxreg = flux[(wave >= objminwave) & (wave <= objmaxwave)]  # cut flux to region
        if len(fluxreg):
            fluxmed = np.nanmedian(fluxreg)
        else:
            fluxmed = np.nanmedian(flux)  # for spectra with wavelength steps < 100A
        return flux / fluxmed

    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    tspec: Table = db.query(db.Spectra).\
        filter(db.Spectra.c.source == query).\
        table(spectra=['spectrum'])  # query the database for the spectra
    nfail, failstrlist = 0, []
    if not len(tspec):  # if there aren't any spectra, return nothing
        return None, None, None, None
    p = figure(title='Spectra', plot_height=500,
               active_scroll='wheel_zoom', active_drag='box_zoom',
               tools='pan,wheel_zoom,box_zoom,save,reset', toolbar_location='left',
               sizing_mode='stretch_width')  # init figure
    p.xaxis.axis_label_text_font_size = '1.5em'
    p.yaxis.axis_label_text_font_size = '1.5em'
    p.xaxis.major_label_text_font_size = '1.5em'
    p.yaxis.major_label_text_font_size = '1.5em'
    p.title.text_font_size = '2em'
    p.xaxis.axis_label = 'Wavelength [μm]'  # units for wavelength on x axis
    p.yaxis.axis_label = 'Normalised Flux'  # units for wavelength on y axis
    normfact, ld = None, 'solid'
    i = 0
    cdslist, lineplots = [], []
    normminwave, normmaxwave = 0.81, 0.82
    fluxmin, fluxmax = np.inf, -np.inf
    for spec in tspec:  # over all spectra
        spectrum: Spectrum1D = spec['spectrum']  # spectrum as an object
        try:
            wave: np.ndarray = spectrum.spectral_axis.to(u.micron).value  # unpack wavelengths
        except (u.UnitConversionError, AttributeError):  # check astrodbkit2 has loaded spectra
            nfail += 1
            if spec["mode"] is None:
                failstrlist.append(f'{spec["telescope"]}/{spec["instrument"]} '
                                   f' ({spec["reference"]})')
            else:
                failstrlist.append(f'{spec["telescope"]}/{spec["instrument"]}/{spec["mode"]}'
                                   f' ({spec["reference"]})')
            continue
        flux: np.ndarray = spectrum.flux.value  # unpack fluxes
        label = f'{spec["telescope"]}-{spec["instrument"]}: {spec["observation_date"].date()}'  # legend label
        normminwave = wave[0] if wave[0] < normminwave else normminwave
        normmaxwave = wave[-1] if wave[-1] > normmaxwave else normmaxwave
        normflux = normalise()  # normalise the flux by the sum
        fluxmin = np.min(normflux) if np.min(normflux) < fluxmin else fluxmin
        fluxmax = np.max(normflux) if np.max(normflux) > fluxmax else fluxmax
        cds = ColumnDataSource(data=dict(wave=wave, flux=flux, normflux=normflux))
        cdslist.append(cds)
        if j := i > len(Colorblind8):  # loop around colours if we have more than 8 spectra, and start line dashing
            j = 0
            ld = 'dashed'
        lineplot = p.line(x='wave', y='normflux', source=cds, legend_label=label,
                          line_color=Colorblind8[j], line_dash=ld)  # create line plot
        lineplots.append(lineplot)
        i += 1
    failstr = 'The spectra ' + ', '.join(failstrlist) + ' could not be plotted.'
    if not i:
        return None, None, nfail, failstr
    bounds = [normminwave, normmaxwave, fluxmin, fluxmax]
    p.add_tools(HoverTool(tooltips=[('Wave', '@wave'), ('Flux', '@flux')], renderers=lineplots))
    featuresall = {'L Dwarf Features': ['k', 'na', 'feh', 'tio', 'co', 'h2o', 'h2'],
                   'T Dwarf Features': ['k', 'ch4', 'h2o', 'h2'],
                   'Youth Features': ['vo', ],
                   'Binary Features': ['sb', ]}
    p.legend.click_policy = 'hide'  # hide the graph if clicked on
    p.legend.label_text_font_size = '1.5em'
    spmin = Span(location=0.81, dimension='height', line_color='white', line_dash='dashed')
    spmax = Span(location=0.82, dimension='height', line_color='white', line_dash='dashed')
    spslide = RangeSlider(start=normminwave, end=normmaxwave, value=(0.81, 0.82), step=0.01, title='Normalisation')
    p.js_on_event('reset', CustomJS(args=dict(spslide=spslide), code=jscallbacks.reset_slider))
    spslide.js_on_change('value', CustomJS(args=dict(spmin=spmin, spmax=spmax, cdslist=cdslist),
                                           code=jscallbacks.normslider))
    for sp in (spmin, spmax):
        p.add_layout(sp)
    # all this features stuff is heavily taken from splat
    yoff = 0.02 * (bounds[3] - bounds[2])  # label offset
    for featurename, features in featuresall.items():
        for ftr in features:
            for ii, waverng in enumerate(FEATURE_LABELS[ftr]['wavelengths']):
                if np.nanmin(waverng) > bounds[0] and np.nanmax(waverng) < bounds[1]:
                    y = 1
                    if FEATURE_LABELS[ftr]['type'] == 'band':
                        p.line(waverng, [y + 5 * yoff] * 2, color='white', legend_label=featurename, visible=False)
                        lfeat = p.line([waverng[0]] * 2, [y + 4 * yoff, y + 5 * yoff], color='white',
                                       legend_label=featurename, visible=False)
                        t = Label(x=np.mean(waverng), y=y + 5.5 * yoff, text=FEATURE_LABELS[ftr]['label'],
                                  text_color='white', visible=False)
                    else:
                        lfeat = None
                        for w in waverng:
                            lfeat = p.line([w] * 2, [y, y + yoff], color='white', line_dash='dotted',
                                           legend_label=featurename, visible=False)
                        t = Label(x=np.mean(waverng), y=y + 1.5 * yoff, text=FEATURE_LABELS[ftr]['label'],
                                  text_color='white', visible=False)
                    p.add_layout(t)
                    lfeat.js_on_change('visible', CustomJS(args=dict(t=t),
                                                           code="""t.visible = cb_obj.visible;"""))
    scriptdiv = components(column(p, spslide, sizing_mode='stretch_width'),
                           theme=nightskytheme)  # convert bokeh plot into script and div
    script: str = scriptdiv[0]
    div: str = scriptdiv[1]
    return script, div, nfail, failstr


def multiplotbokeh(all_results_full: pd.DataFrame, all_bands: np.ndarray,
                   all_photo: pd.DataFrame, all_plx: pd.DataFrame,
                   jscallbacks: JSCallbacks, nightskytheme: Theme) -> Tuple[str, str]:
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
             all_photo: pd.DataFrame, jscallbacks: JSCallbacks, nightskytheme: Theme) -> Tuple[Optional[str],
                                                                                               Optional[str]]:
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
    """
    Control module, called to grab the specific instances relating to plotting.
    """
    _nightskytheme = built_in_themes['night_sky']  # darker theme for bokeh
    _jscallbacks = JSCallbacks()  # grab the callbacks for bokeh interactivity
    return _nightskytheme, _jscallbacks


if __name__ == '__main__':
    ARGS, DB_FILE, PHOTOMETRIC_FILTERS, ALL_RESULTS, ALL_RESULTS_FULL, ALL_PHOTO, ALL_BANDS, ALL_PLX = mainutils()
    NIGHTSKYTHEME, JSCALLBACKS = mainplots()
