"""
File containing the 'workhorse' functions generating the various plots seen on the website
"""
# local packages
from .utils import *

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


def bokeh_formatter(p: figure) -> figure:
    """
    Performs some basic formatting

    Parameters
    ----------
    p
        The figure object

    Returns
    -------
    p
        The formatted figure
    """
    p.xaxis.axis_label_text_font_size = '1.5em'
    p.yaxis.axis_label_text_font_size = '1.5em'
    p.xaxis.major_label_text_font_size = '1.5em'
    p.yaxis.major_label_text_font_size = '1.5em'
    p.title.text_font_size = '2em'
    return p


def name_simplifier(s: str) -> str:
    """
    Simplifies name of magnitude into something nicer to read

    Parameters
    ----------
    s
        Input string

    Returns
    -------
    s
        Simplified input string
    """
    s = s.replace('-', ' - ').replace('GAIA3.', '').replace('2MASS.', '').replace('WISE.', '')
    return s


def spectra_plot(query: str, db_file: str, night_sky_theme: Theme,
                 js_callbacks: JSCallbacks) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """
    Creates the bokeh representation of the spectra plot

    Parameters
    ----------
    query: str
        The object that has been searched for
    db_file: str
        The connection string of the database
    night_sky_theme: Theme
        The bokeh theme
    js_callbacks: JSCallbacks
        An instance containing the javascript as strings

    Returns
    -------
    script: str
        script for creating the bokeh plot
    div: str
        the html to be inserted in dom
    n_fail: int
        the number of failed spectra to be loaded
    fail_string: str
        the failed spectra
    """
    def normalise() -> np.ndarray:
        """
        Normalises the flux using the wave & flux variables in the surrounding scope
        """
        # clipping between the start and end points of the normalisation region
        wave_start, wave_end = wave[0], wave[-1]
        min_wave, max_wave = 0.81, 0.82
        object_min_wave, object_max_wave = np.clip([min_wave, max_wave], wave_start, wave_end)

        if np.isclose(object_min_wave, wave_end):
            object_min_wave -= 0.01
        if np.isclose(object_max_wave, wave_start):
            object_max_wave += 0.01

        # normalising flux by median in wavelength regime
        fluxreg = flux[(wave >= object_min_wave) & (wave <= object_max_wave)]

        if len(fluxreg):
            fluxmed = np.nanmedian(fluxreg)

        else:
            fluxmed = np.nanmedian(flux)

        if not np.isclose(fluxmed, 0, atol=1e-30):
            return flux / fluxmed
        return flux  # unable to normalise by first 0.01um

    # query the database for the spectra
    db = SimpleDB(db_file)  # open database
    t_spectra: Table = db.query(db.Spectra).\
        filter(db.Spectra.c.source == query).\
        table()  # do not use spectra=['access_url'] here, it will try to read the spectra as Spectrum objects

    # initialise plot
    n_fail, fail_string_list = 0, []
    if not len(t_spectra):
        return None, None, None, None
    p = figure(title='Spectra', outer_height=500,
               active_scroll='wheel_zoom', active_drag='box_zoom',
               tools='pan,wheel_zoom,box_zoom,save,reset', toolbar_location='left',
               sizing_mode='stretch_width')
    p = bokeh_formatter(p)
    p.xaxis.axis_label = 'Wavelength [μm]'
    p.yaxis.axis_label = 'Normalised Flux'
    i = 0
    cds_list, line_plots = [], []
    normalised_min_wave, normalised_max_wave = 0.81, 0.82
    flux_min, flux_max = np.inf, -np.inf

    # checking each spectra in table
    for spec in t_spectra:
        try:
            # Manually convert the spectrum to a Spectrum object
            spectrum = Spectrum.read(spec['access_url'], cache=True)
        except Exception as e:
            print(f"Unable to read {spec['access_url']} as Spectrum. Error: {e}")
            continue

        # checking spectrum has good units and not only NaNs or 0s
        try:
            wave: np.ndarray = spectrum.spectral_axis.to(u.micron).value
            flux: np.ndarray = spectrum.flux.value
            nan_check: np.ndarray = ~np.isnan(flux) & ~np.isnan(wave)
            zero_check: np.ndarray = ~np.isclose(flux, 0, atol=1e-30)
            nanzero_check = nan_check & zero_check
            wave = wave[nanzero_check]
            flux = flux[nanzero_check]
            if not len(wave):
                raise ValueError

        # handle any objects which failed checks
        except (u.UnitConversionError, AttributeError, ValueError):
            n_fail += 1

            if spec["mode"] == 'Missing':
                fail_string_list.append(f'{spec["telescope"]}/{spec["instrument"]} '
                                        f' ({spec["reference"]})')

            else:
                fail_string_list.append(f'{spec["telescope"]}/{spec["instrument"]}/{spec["mode"]}'
                                        f' ({spec["reference"]})')
            continue

        # otherwise, label and normalise
        label = f'{spec["telescope"]}-{spec["instrument"]}: {spec["observation_date"].date()}'
        normalised_min_wave = wave[0] if wave[0] < normalised_min_wave else normalised_min_wave
        normalised_max_wave = wave[-1] if wave[-1] > normalised_max_wave else normalised_max_wave
        normalised_flux = normalise()
        flux_min = np.min(normalised_flux) if np.min(normalised_flux) < flux_min else flux_min
        flux_max = np.max(normalised_flux) if np.max(normalised_flux) > flux_max else flux_max

        # add to bokeh object
        cds = ColumnDataSource(data=dict(wave=wave, flux=flux, normalised_flux=normalised_flux))
        cds_list.append(cds)

        # cycle the palette, dashed once it is exhausted
        line_dash = 'solid' if i < len(Colorblind8) else 'dashed'
        lineplot = p.line(x='wave', y='normalised_flux', source=cds, legend_label=label,
                          line_color=Colorblind8[i % len(Colorblind8)], line_dash=line_dash, line_width=2)
        line_plots.append(lineplot)
        i += 1

    # handle the case when no spectra could be plotted
    fail_string = 'The spectra ' + ', '.join(fail_string_list) + ' could not be plotted.'
    if not i:
        return None, None, n_fail, fail_string

    # additional plot functionality
    bounds = [normalised_min_wave, normalised_max_wave, flux_min, flux_max]
    p.add_tools(HoverTool(tooltips=[('Wave', '@wave'), ('Flux', '@flux')], renderers=line_plots))
    featuresall = {'L Dwarf Features': ['k', 'na', 'feh', 'tio', 'co', 'h2o'],
                   'T Dwarf Features': ['k', 'ch4', 'h2o'],
                   'Youth Features': ['vo', ],
                   'Binary Features': ['sb', ]}
    p.legend.click_policy = 'hide'
    p.legend.label_text_font_size = '1.5em'
    spectra_min = Span(location=0.81, dimension='height', line_color='white', line_dash='dashed')
    spectra_max = Span(location=0.82, dimension='height', line_color='white', line_dash='dashed')
    spectra_slide = RangeSlider(start=normalised_min_wave, end=normalised_max_wave, value=(0.81, 0.82),
                                step=0.01, title='Normalisation', sizing_mode='stretch_width')
    p.js_on_event('reset', CustomJS(args=dict(spectra_slide=spectra_slide), code=js_callbacks.reset_slider))
    spectra_slide.js_on_change('value',
                               CustomJS(args=dict(spectra_min=spectra_min, spectra_max=spectra_max, cds_list=cds_list),
                                        code=js_callbacks.normalisation_slider))

    for sp in (spectra_min, spectra_max):
        p.add_layout(sp)

    # spectral features, heavily taken from splat
    yoff = 0.02 * (bounds[3] - bounds[2])
    toglist = []

    for feature_name, features in featuresall.items():
        feature_toggle = Toggle(label=feature_name, width=200)

        for feature in features:

            for ii, wavelength_range in enumerate(FEATURE_LABELS[feature]['wavelengths']):

                if np.nanmin(wavelength_range) > bounds[0] and np.nanmax(wavelength_range) < bounds[1]:
                    y = 1

                    if FEATURE_LABELS[feature]['type'] == 'band':
                        p.line(wavelength_range, [y + 5 * yoff] * 2, color='white', visible=False)
                        line_feature = p.line([wavelength_range[0]] * 2, [y + 4 * yoff, y + 5 * yoff],
                                              color='white', visible=False)
                        t = Label(x=np.mean(wavelength_range), y=y + 5.5 * yoff, text=FEATURE_LABELS[feature]['label'],
                                  text_color='white', visible=False)

                    else:
                        line_feature = None

                        for w in wavelength_range:
                            line_feature = p.line([w] * 2, [y, y + yoff], color='white',
                                                  line_dash='dotted', visible=False)

                        t = Label(x=np.mean(wavelength_range), y=y + 1.5 * yoff, text=FEATURE_LABELS[feature]['label'],
                                  text_color='white', visible=False)
                    p.add_layout(t)
                    feature_toggle.js_link('active', line_feature, 'visible')
                    line_feature.js_on_change('visible', CustomJS(args=dict(t=t),
                                                                  code="""t.visible = cb_obj.visible;"""))
        toglist.append(feature_toggle)

    # unpacking bokeh plots into html content
    scriptdiv = components(column(row(p, column(*toglist, max_width=200), sizing_mode='stretch_width'),
                                  spectra_slide, sizing_mode='stretch_width'),
                           theme=night_sky_theme)
    script: str = scriptdiv[0]
    div: str = scriptdiv[1]
    return script, div, n_fail, fail_string


def multi_plot_bokeh(all_results: pd.DataFrame, all_bands: np.ndarray,
                     all_photometry: pd.DataFrame, all_parallaxes: pd.DataFrame, all_spectral_types: pd.DataFrame,
                     all_modeled_parameters: pd.DataFrame, js_callbacks: JSCallbacks,
                     night_sky_theme: Theme) -> Tuple[str, str]:
    """
    The workhorse generating the multiple plots view page

    Parameters
    ----------
    all_results: pd.DataFrame
        Every object and its basic information
    all_bands: np.ndarray
        All the photometric bands for colour-colour
    all_photometry: pd.DataFrame
        All the photometry
    all_parallaxes: pd.DataFrame
        All the parallaxes
    all_spectral_types: pd.DataFrame
        All spectral types
    all_modeled_parameters: pd.DataFrame
        Modeled physical parameters, one column per parameter
    js_callbacks: JSCallbacks
        The javascript callbacks for bokeh
    night_sky_theme: Theme
        The bokeh theme

    Returns
    -------
    script: str
        script for creating the bokeh plot
    div: str
        the html to be inserted in dom
    """
    def sky_plot() -> figure:
        """
        Creates the sky plot for projected position
        """
        # sky plot
        _p_sky = figure(title='Sky Plot', outer_height=500,
                        active_scroll='wheel_zoom', active_drag='box_zoom',
                        tools='pan,wheel_zoom,box_zoom,box_select,reset',
                        sizing_mode='stretch_width', x_range=Range1d(-180, 180), y_range=Range1d(-90, 90))

        # background for skyplot
        _p_sky.ellipse(x=0, y=0, width=360, height=180, color='#444444', name='background')

        # scatter plot for sky plot
        circle = _p_sky.scatter(marker='circle', source=full_cds, x='ra_projected', y='dec_projected',
                                size=6, name='circle', color='ghostwhite')

        # bokeh tools for sky plot
        this_hover = HoverTool(renderers=[circle, ], tooltips=tooltips)
        this_tap = TapTool(renderers=[circle, ])
        _p_sky.add_tools(this_hover)
        _p_sky.add_tools(this_tap)
        _p_sky = bokeh_formatter(_p_sky)
        this_tap.callback = OpenURL(url='/load_solo/@source')
        return _p_sky

    def dropdown_scatter(title: str, x_name: str, y_name: str, x_menu: List[Tuple[str, str]],
                         y_menu: List[Tuple[str, str]], y_flip: bool = False) -> column:
        """
        Scatter plot of any two columns, coloured by spectral type, with axis drop downs and flip buttons
        """
        p = figure(title=title, outer_height=500, active_scroll='wheel_zoom', active_drag='box_zoom',
                   tools='pan,wheel_zoom,box_zoom,box_select,hover,tap,reset', tooltips=tooltips,
                   sizing_mode='stretch_width')
        y_limits = all_results_full[y_name].min(), all_results_full[y_name].max()
        p.x_range = Range1d(all_results_full[x_name].min(), all_results_full[x_name].max())
        p.y_range = Range1d(*(y_limits[::-1] if y_flip else y_limits))
        p.xaxis.axis_label = name_simplifier(x_name)
        p.yaxis.axis_label = name_simplifier(y_name)
        p = bokeh_formatter(p)
        scatter = p.scatter(marker='circle', x=x_name, y=y_name, source=full_cds, size=6, color=cmap)
        p.add_layout(ColorBar(color_mapper=cmap['transform'], label_standoff=12,
                              ticker=FixedTicker(ticks=np.arange(60, 100, 10), minor_ticks=np.arange(60, 100, 5)),
                              major_label_overrides={60: 'M', 70: 'L', 80: 'T', 90: 'Y'},
                              major_label_text_font_size='1.5em'), 'right')
        p.select(type=TapTool).callback = OpenURL(url='/load_solo/@source')  # open target page when tapped

        button_x_flip = Toggle(label='X Flip')
        button_x_flip.js_on_click(CustomJS(code=js_callbacks.button_flip, args={'ax_range': p.x_range}))
        button_y_flip = Toggle(label='Y Flip', active=y_flip)
        button_y_flip.js_on_click(CustomJS(code=js_callbacks.button_flip, args={'ax_range': p.y_range}))
        dropdown_x = Select(options=x_menu, value=x_name)
        dropdown_x.js_on_change('value', CustomJS(code=js_callbacks.dropdown_x_js,
                                                  args={'full_plot': scatter, 'full_data': full_cds.data,
                                                        'x_button': button_x_flip, 'x_axis': p.xaxis[0],
                                                        'x_range': p.x_range}))
        dropdown_y = Select(options=y_menu, value=y_name)
        dropdown_y.js_on_change('value', CustomJS(code=js_callbacks.dropdown_y_js,
                                                  args={'full_plot': scatter, 'full_data': full_cds.data,
                                                        'y_button': button_y_flip, 'y_axis': p.yaxis[0],
                                                        'y_range': p.y_range}))
        p.js_on_event('reset', CustomJS(args=dict(dropdown_x=dropdown_x, dropdown_y=dropdown_y,
                                                  x_value=x_name, y_value=y_name),
                                        code=js_callbacks.reset_dropdown))
        return column(p, row(dropdown_x, dropdown_y, sizing_mode='stretch_width'),
                      row(button_x_flip, button_y_flip, sizing_mode='stretch_width'), sizing_mode='scale_width')

    def menu(columns) -> List[Tuple[str, str]]:
        """
        Drop down options of the given columns which have at least two values to plot
        """
        return [(col, name_simplifier(col)) for col in columns if all_results_full[col].count() > 1]

    # gather all necessary data including parallaxes, spectral types and bands
    all_results_full = results_concat(all_results, all_photometry, all_parallaxes, all_spectral_types, all_bands)
    all_results_full = all_results_full.merge(all_modeled_parameters, on='source', how='left')
    all_results_full.dropna(axis=1, how='all', inplace=True)
    all_bands = all_bands[np.isin(all_bands, all_results_full.columns)]
    all_bands = np.array([band for band in COLOUR_BANDS if band in all_bands])

    # bokeh tools initialisation
    full_cds = ColumnDataSource(all_results_full)
    cmap = linear_cmap('sptnum', Turbo256, 60, 100)
    tooltips = [('Target', '@source')]

    # create sky plot
    p_sky = sky_plot()

    # colour-colour plot
    colour_menu = menu([col for col in COLOURS if col in all_results_full.columns])
    p_colour_colour = dropdown_scatter('Colour-Colour', colour_menu[0][0], colour_menu[1][0], colour_menu, colour_menu)

    # colour absolute magnitude diagram
    mag_menu = menu([f'M_{band}' for band in ABSOLUTE_MAGNITUDE_BANDS if band in all_bands])
    p_camd = dropdown_scatter('Colour-Absolute Magnitude Diagram', colour_menu[0][0], mag_menu[1][0],
                              colour_menu, mag_menu, y_flip=True)

    # physical parameters against each other, absolute magnitudes and colours
    physical_menu = menu(all_modeled_parameters.columns.drop('source')) + mag_menu + colour_menu
    p_physical = dropdown_scatter('Physical Parameters', 'mass [M_jup]', 'M_2MASS.J', physical_menu, physical_menu,
                                  y_flip=True)

    # constructing bokeh layout and outputting to html
    plots = column(p_sky, row(p_colour_colour, p_camd, sizing_mode='scale_width'),
                   row(p_physical, sizing_mode='scale_width'), sizing_mode='scale_width')
    script, div = components(plots, theme=night_sky_theme)
    return script, div


def camd_plot(query: str, everything: Inventory, all_bands: np.ndarray, all_results: pd.DataFrame,
              all_parallaxes: pd.DataFrame, all_spectral_types: pd.DataFrame, photometric_filters: pd.DataFrame,
              all_photometry: pd.DataFrame, js_callbacks: JSCallbacks, night_sky_theme: Theme, db_file: str) -> Tuple[Optional[str],
                                                                                                        Optional[str]]:
    """
    Creates CAMD plot into html

    Parameters
    ----------
    query: str
        The object that has been searched for
    everything: Inventory
        The class representation wrapping db.inventory
    all_bands: np.ndarray
        All of the photometric bands for colour-colour
    all_results: pd.DataFrame
        Every object and its basic information
    all_parallaxes: pd.DataFrame
        All of the parallaxes
    all_spectral_types: pd.DataFrame
        All spectral types
    photometric_filters: pd.DataFrame
        All of the filters to check
    all_photometry: pd.DataFrame
        All of the photometry
    js_callbacks: JSCallbacks
        The javascript callbacks for bokeh
    night_sky_theme: Theme
        The theme for bokeh
    db_file: str
        The connection string to the database

    Returns
    -------
    script: str
        script for creating the bokeh plot
    div: str
        the html to be inserted directly
    """
    # initialise plot
    tooltips = [('Target', '@target')]
    cmap = linear_cmap('sptnum', Turbo256, 60, 100)

    # retrieve photometry for given object
    try:
        this_photometry: pd.DataFrame = everything.list_concat('Photometry', db_file, False)

        if len(this_photometry) < 4:
            raise KeyError('Not enough photometric entries')

    # give up if not enough photometry present
    except KeyError:
        return None, None

    # look for spectral type
    try:
        this_spectral_type: pd.DataFrame = everything.list_concat('SpectralTypes', db_file, False)

    except KeyError:
        this_spectral_type = pd.DataFrame.from_dict(dict(spectral_type_code=[np.nan, ], adopted=[np.nan, ]))

    # use adopted spectral type if present
    this_spectral_type['adopted'] = this_spectral_type.adopted.astype(bool).fillna(False)
    if not this_spectral_type.adopted.any():
        this_spectral_type.loc[0, 'adopted'] = True

    # get all photometry for this object in a useful format, including bands
    this_photometry = parse_photometry(this_photometry, all_bands)
    this_bands: np.ndarray = np.unique(this_photometry.columns)
    this_photometry: pd.DataFrame = find_colours(this_photometry, this_bands, photometric_filters)
    this_photometry['target'] = query
    this_photometry['sptnum'] = this_spectral_type.loc[this_spectral_type.adopted].spectral_type_code.iloc[0]

    # attempt to retrieve parallaxes to process absolute magnitudes
    try:
        this_parallaxes: pd.DataFrame = everything.list_concat('Parallaxes', db_file, False)

    except KeyError:
        pass

    # if parallax present
    else:

        # use adopted parallax, if present
        this_parallaxes['adopted'] = this_parallaxes.adopted.astype(bool).fillna(False)
        if not this_parallaxes.adopted.any():
            this_parallaxes.loc[0, 'adopted'] = True
        this_photometry['parallax'] = this_parallaxes.loc[this_parallaxes.adopted].parallax.iloc[0]
        this_photometry = absolute_magnitudes(this_photometry, this_bands)  # get abs mags

    this_photometry.dropna(axis=1, how='all', inplace=True)

    # comparing with the full sample
    all_results_full = results_concat(all_results, all_photometry, all_parallaxes, all_spectral_types, this_bands)
    all_results_full.dropna(axis=1, how='all', inplace=True)

    # preparing the available magnitudes and colours
    colour_bands = np.array([col for col in (*COLOURS, *(f'M_{band}' for band in ABSOLUTE_MAGNITUDE_BANDS))
                             if col in this_photometry.columns and col in all_results_full.columns])

    bad_cols = []
    for col in colour_bands:

        if not all_results_full[col].count() > 1:
            bad_cols.append(col)

    colour_bands = colour_bands[~np.isin(colour_bands, bad_cols)]
    just_colours = this_photometry.loc[:, colour_bands].copy()

    if len(just_colours.columns) < 2:
        return None, None

    x_full_name = just_colours.columns[0]
    y_full_name = just_colours.columns[1]
    x_shown_name = name_simplifier(x_full_name)
    y_shown_name = name_simplifier(y_full_name)
    axis_names = [name_simplifier(col) for col in just_colours.columns]
    dropdown_menu = [*zip(just_colours.columns, axis_names), ]

    # initialise plot
    p = figure(outer_height=500,
               active_scroll='wheel_zoom', active_drag='box_zoom',
               tools='pan,wheel_zoom,box_zoom,hover,tap,reset', tooltips=tooltips,
               sizing_mode='stretch_width')
    p.x_range = Range1d(all_results_full[x_full_name].min(), all_results_full[x_full_name].max())
    p.y_range = Range1d(all_results_full[y_full_name].min(), all_results_full[y_full_name].max())
    p.xaxis.axis_label = x_shown_name
    p.yaxis.axis_label = y_shown_name
    p = bokeh_formatter(p)

    # scatter plot for given object
    this_cds = ColumnDataSource(data=this_photometry)
    this_plot = p.scatter(marker='square', x=x_full_name, y=y_full_name, source=this_cds,
                          color=cmap, size=20)  # plot for this object

    # scatter plot for all data
    cds_full = ColumnDataSource(data=all_results_full)  # bokeh cds object
    full_plot = p.scatter(marker='circle', x=x_full_name, y=y_full_name, source=cds_full,
                          color=cmap, alpha=0.5, size=6)  # plot all objects
    full_plot.level = 'underlay'  # put full plot underneath this plot

    # colour bar
    cbar = ColorBar(color_mapper=cmap['transform'], label_standoff=12,
                    ticker=FixedTicker(ticks=np.arange(60, 100, 10), minor_ticks=np.arange(60, 100, 5)),
                    major_label_overrides={60: 'M', 70: 'L', 80: 'T', 90: 'Y'},
                    major_label_text_font_size='1.5em')
    p.add_layout(cbar, 'right')

    # bokeh tools
    tap_tool = p.select(type=TapTool)  # tapping
    tap_tool.callback = OpenURL(url='/load_solo/@target')
    button_x_flip = Toggle(label='X Flip')
    button_x_flip.js_on_click(CustomJS(code=js_callbacks.button_flip, args={'ax_range': p.x_range}))
    button_y_flip = Toggle(label='Y Flip')
    button_y_flip.js_on_click(CustomJS(code=js_callbacks.button_flip, args={'ax_range': p.y_range}))
    dropdown_x = Select(options=dropdown_menu, value=x_full_name)  # x axis select
    dropdown_x.js_on_change('value', CustomJS(code=js_callbacks.dropdown_x_js,
                                              args={'full_plot': full_plot, 'this_plot': this_plot,
                                                    'full_data': cds_full.data, 'x_button': button_x_flip,
                                                    'x_axis': p.xaxis[0], 'x_range': p.x_range}))
    dropdown_y = Select(options=dropdown_menu, value=y_full_name)  # y axis select
    dropdown_y.js_on_change('value', CustomJS(code=js_callbacks.dropdown_y_js,
                                              args={'full_plot': full_plot, 'this_plot': this_plot,
                                                    'full_data': cds_full.data, 'y_button': button_y_flip,
                                                    'y_axis': p.yaxis[0], 'y_range': p.y_range}))
    p.js_on_event('reset', CustomJS(args=dict(dropdown_x=dropdown_x, dropdown_y=dropdown_y),
                                    code=js_callbacks.reset_dropdown))

    # creating bokeh layout and html
    plots = column(p, row(dropdown_x,
                          dropdown_y,
                          button_x_flip,
                          button_y_flip,
                          sizing_mode='scale_width'),
                   sizing_mode='scale_width')
    script, div = components(plots, theme=night_sky_theme)
    return script, div


def main_plots():
    """
    Control module, called to grab the specific instances relating to plotting.
    """
    _night_sky_theme = built_in_themes['night_sky']  # darker theme for bokeh
    _js_callbacks = JSCallbacks()  # grab the callbacks for bokeh interactivity
    return _night_sky_theme, _js_callbacks


