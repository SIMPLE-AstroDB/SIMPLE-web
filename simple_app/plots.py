"""
File containing the 'workhorse' functions generating the various plots seen on the website
"""
# local packages
from .utils import *

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


def multi_plot_bokeh(all_concat: pd.DataFrame, all_bands: np.ndarray, all_modeled_parameters: pd.DataFrame,
                     js_callbacks: JSCallbacks, night_sky_theme: Theme) -> Tuple[str, str]:
    """
    The workhorse generating the multiple plots view page

    Parameters
    ----------
    all_concat: pd.DataFrame
        Every object with its photometry, parallax and spectral type, from results_concat()
    all_bands: np.ndarray
        All the photometric bands for colour-colour
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
    all_results_full = all_concat.merge(all_modeled_parameters, on='source', how='left')
    all_results_full = all_results_full.dropna(axis=1, how='all')
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


def camd_plot(query: str, everything: Inventory, all_bands: np.ndarray, all_concat: pd.DataFrame,
              photometric_filters: pd.DataFrame, js_callbacks: JSCallbacks, night_sky_theme: Theme,
              db_file: str) -> Tuple[Optional[str], Optional[str]]:
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
    all_concat: pd.DataFrame
        Every object with its photometry, parallax and spectral type, from results_concat()
    photometric_filters: pd.DataFrame
        All of the filters to check
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

    this_photometry = this_photometry.dropna(axis=1, how='all')

    # comparing with the full sample
    all_results_full = all_concat.dropna(axis=1, how='all')

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


