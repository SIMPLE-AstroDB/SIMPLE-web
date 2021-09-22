"""
This is the main script to be run from the directory root, it will start the Flask application running which one can
then connect to.
"""
# external packages
from astrodbkit2.astrodb import Database, REFERENCE_TABLES  # used for pulling out database and querying
from astropy.coordinates import SkyCoord
from astropy.table import Table
from bokeh.palettes import Colorblind8
from bokeh.embed import components
from bokeh.layouts import row, column  # bokeh displaying nicely
from bokeh.models import ColumnDataSource, Range1d, CustomJS,\
    Select, Toggle, TapTool, OpenURL, HoverTool  # bokeh models
from bokeh.plotting import figure, curdoc  # bokeh plotting
from bokeh.resources import CDN
from bokeh.themes import built_in_themes
from flask import Flask, render_template, jsonify  # website functionality
from flask_cors import CORS  # cross origin fix (aladin mostly)
from flask_wtf import FlaskForm  # web forms
from markdown2 import markdown  # using markdown formatting
import numpy as np  # numerical python
import pandas as pd  # running dataframes
from specutils import Spectrum1D
from wtforms import StringField, SubmitField  # web forms
# internal packages
import argparse  # system arguments
import os  # operating system
from typing import Union, List  # type hinting
from urllib.parse import quote  # handling strings into url friendly form
# local packages
from simple_callbacks import JSCallbacks

# initialise
app_simple = Flask(__name__)  # start flask app
app_simple.config['SECRET_KEY'] = os.urandom(32)  # need to generate csrf token as basic security for Flask
CORS(app_simple)  # makes CORS work (aladin notably)
nightskytheme = built_in_themes['night_sky']  # nicer looking bokeh


def sysargs():
    """
    These are the system arguments given after calling this python script

    Returns
    -------
    _args
        The different argument parameters, can be grabbed via their long names (e.g. _args.host)
    """
    _args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _args.add_argument('-i', '--host', default='127.0.0.1',
                       help='Local IP Address to host server, default 127.0.0.1')
    _args.add_argument('-p', '--port', default=8000,
                       help='Local port number to host server through, default 8000', type=int)
    _args.add_argument('-d', '--debug', help='Run Flask in debug mode?', default=False, action='store_true')
    _args.add_argument('-f', '--file', default='SIMPLE.db',
                       help='Database file path relative to current directory, default SIMPLE.db')
    _args = _args.parse_args()
    return _args


class SimpleDB(Database):  # this keeps pycharm happy about unresolved references
    """
    Wrapper class for astrodbkit2.Database specific to SIMPLE
    """
    Sources = None  # initialise class attribute
    Photometry = None
    Parallaxes = None
    Spectra = None


class Inventory:
    """
    For use in the solo result page where the inventory of an object is queried, grabs also the RA & Dec
    """
    ra: float = 0
    dec: float = 0

    def __init__(self, resultdict: dict):
        """
        Constructor method for Inventory

        Parameters
        ----------
        resultdict: dict
            The dictionary of all the key: values in a given object inventory
        """
        self.results: dict = resultdict  # given inventory for a target
        for key in self.results:  # over every key in inventory
            if args.debug:
                print(key)
            if key in REFERENCE_TABLES:  # ignore the reference table ones
                continue
            lowkey: str = key.lower()  # lower case of the key
            mkdown_output: str = self.listconcat(key)  # get in markdown the dataframe value for given key
            setattr(self, lowkey, mkdown_output)  # set the key attribute with the dataframe for given key
        try:
            srcs: pd.DataFrame = self.listconcat('Sources', rtnmk=False)  # open the Sources result
            self.ra, self.dec = srcs.ra[0], srcs.dec[0]
        except (KeyError, AttributeError):
            pass
        return

    def listconcat(self, key: str, rtnmk: bool = True) -> Union[pd.DataFrame, str]:
        """
        Concatenates the list for a given key

        Parameters
        ----------
        key: str
            The key corresponding to the inventory
        rtnmk: bool
            Switch for whether to return either a markdown string or a dataframe
        """
        obj: List[dict] = self.results[key]  # the value for the given key
        df: pd.DataFrame = pd.concat([pd.DataFrame(objrow, index=[i])  # create dataframe from found dict
                                      for i, objrow in enumerate(obj)], ignore_index=True)  # every dict in the list
        if rtnmk and key == 'Spectra':
            df.drop(columns=[col for col in df.columns if any([substr in col for substr in ('wave', 'flux')])],
                    inplace=True)
            df = df.loc[:, 'telescope':].copy()
        if rtnmk:  # return markdown boolean
            df.rename(columns={s: s.replace('_', ' ') for s in df.columns}, inplace=True)  # renaming columns
            return markdown(df.to_html(index=False,
                                       classes='table table-dark table-bordered table-striped'))  # html then markdown
        return df  # otherwise return dataframe as is


class SearchForm(FlaskForm):
    """
    Searchbar class
    """
    search = StringField('Search for an object:', id='autocomplete')  # searchbar
    submit = SubmitField('Query', id='querybutton')  # clicker button to send request


def all_sources():
    """
    Queries the full table to get all the sources

    Returns
    -------
    allresults
        Just the main IDs
    fullresults
        The full dataframe of all the sources
    """
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    fullresults: pd.DataFrame = db.query(db.Sources).pandas()
    allresults: list = fullresults['source'].tolist()  # gets all the main IDs in the database
    return allresults, fullresults


def find_colours(photodf: pd.DataFrame, allbands: np.ndarray):
    """
    Find all the colours using available photometry

    Parameters
    ----------
    photodf: pd.DataFrame
        The dataframe with all photometry in
    allbands: np.ndarray
        All the photometric bands

    Returns
    -------
    photodf: pd.DataFrame
        The dataframe with all photometry and colours in
    """
    for i, band in enumerate(allbands):  # loop over all bands TODO: sort by wavelength?
        j = 1  # start count
        while j < 20:
            if i + j == len(allbands):  # last band
                break
            nextband: str = allbands[i + j]  # next band
            j += 1
            try:
                photodf[f'{band}_{nextband}'] = photodf[band] - photodf[nextband]  # colour
            except KeyError:
                continue
    return photodf


def parse_photometry(photodf: pd.DataFrame,  allbands: np.ndarray, multisource: bool = False) -> dict:
    """
    Parses the photometry dataframe handling multiple references for same magnitude

    Parameters
    ----------
    photodf: pd.DataFrame
        The dataframe with all photometry in
    allbands: np.ndarray
        All the photometric bands
    multisource: bool
        Switch whether to iterate over initial dataframe with multiple sources

    Returns
    -------
    newphoto: dict
        Dictionary of effectively transposed photometry
    """
    def one_source_iter(onephotodf: pd.DataFrame):
        """
        Parses the photometry dataframe handling multiple references for same magnitude for one object

        Parameters
        ----------
        onephotodf: pd.DataFrame
            The dataframe with all the photometry in it

        Returns
        -------
        thisnewphot: dict
            Dictionary of transposed photometry
        arrsize: int
            The number of rows in the dictionary
        """
        refgrp = onephotodf.groupby('reference')  # all references grouped
        arrsize: int = len(refgrp)  # the number of rows
        thisnewphot = {band: [None, ] * arrsize for band in onephotodf.band.unique()}  # initial dictionary
        thisnewphot['ref'] = [None, ] * arrsize  # references
        for i, (ref, refval) in enumerate(refgrp):  # over all references
            for band, bandval in refval.groupby('band'):  # over all bands
                thisnewphot[band][i] = bandval.iloc[0].magnitude  # given magnitude (0 index of length 1 dataframe)
            thisnewphot['ref'][i] = ref  # reference for these mags
        return thisnewphot, arrsize

    if not multisource:
        newphoto = one_source_iter(photodf)[0]
    else:
        newphoto: dict = {band: [] for band in np.hstack([allbands, ['ref', 'target']])}  # empty dict
        for target, targetdf in photodf.groupby('source'):
            specificphoto, grplen = one_source_iter(targetdf)  # get the dictionary for this object photometry
            targetname = [target, ] * grplen  # list of the target name
            for key in newphoto.keys():  # over all keys
                key: str = key
                if key == 'target':
                    continue
                try:
                    newphoto[key].extend(specificphoto[key])  # extend the list for given key
                except KeyError:  # if that key wasn't present for the object
                    newphoto[key].extend([None, ] * grplen)  # use None as filler
            newphoto['target'].extend(targetname)  # add target to table
    newphotocp: dict = newphoto.copy()
    for key in newphotocp:
        key: str = key
        if key in ('ref', 'target'):  # other than these columns
            continue
        newkey: str = key.replace('.', '_')  # swap dot for underscore
        newphoto[newkey] = newphoto[key].copy()
        del newphoto[key]
    return newphoto


def all_photometry():
    """
    Get all the photometric data from the database to be used in later CMD as background

    Returns
    -------
    allphoto: pd.DataFrame
        All the photometry in a dataframe
    allbands: np.ndarray
        The unique passbands to create dropdowns by
    """
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    allphoto: pd.DataFrame = db.query(db.Photometry).pandas()  # get all photometry
    allbands: np.ndarray = allphoto.band.unique()  # the unique bands
    outphoto: dict = parse_photometry(allphoto, allbands, True)  # transpose photometric table
    allbands = np.array([band.replace('.', '_') for band in allbands])
    allphoto = pd.DataFrame(outphoto)  # use rearranged dataframe
    allphoto = find_colours(allphoto, allbands)  # get the colours
    return allphoto, allbands


def all_parallaxes():
    """
    Get the parallaxes from the database for every object

    Returns
    -------
    allplx: pd.DataFrame
        The dataframe of all the parallaxes
    """
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    allplx: pd.DataFrame = db.query(db.Parallaxes).pandas()  # get all photometry
    allplx = allplx[['source', 'parallax']]
    return allplx


def absmags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all the absolute magnitudes in a given dataframe

    Parameters
    ----------
    df
        The input dataframe
    Returns
    -------
    df
        The output dataframe with absolute mags calculated
    """
    def pogsonlaw(m: Union[float, np.ndarray], dist: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Distance modulus equation

        Parameters
        ----------
        m
            The apparent magnitude
        dist
            The distance in pc
        Returns
        -------
        _
            Absolute magnitude
        """
        return m - 5 * np.log10(dist) + 5

    df['dist'] = np.divide(1000, df['parallax'])
    for mag in all_bands:
        abs_mag = "M_" + mag
        df[abs_mag] = pogsonlaw(df[mag], df['dist'])
    return df


def coordinate_project():
    """
    Projects RA and Dec coordinates onto Mollweide grid

    Returns
    -------
    raproj: np.ndarray
        The projected RA coordinates
    decproj: np.ndarray
        The projected DEC coordinates
    """
    def fnewton_solve(thetan: float, phi: float, acc: float = 1e-4):
        """
        Solves the numerical transformation to project coordinate

        Parameters
        ----------
        thetan
            theta in radians
        phi
            phi in raidans
        acc
            Accuracy of calculation

        Returns
        -------
        thetan
            theta in radians
        """
        thetanp1 = thetan - (2 * thetan + np.sin(2 * thetan) - np.pi * np.sin(phi)) / (2 + 2 * np.cos(2 * thetan))
        if np.isnan(thetanp1):  # at pi/2
            return phi
        elif np.abs(thetanp1 - thetan) / np.abs(thetan) < acc:  # less than desired accuracy
            return thetanp1
        else:
            return fnewton_solve(thetanp1, phi)

    @np.vectorize
    def project_mollweide(ra: Union[np.ndarray, float], dec: Union[np.ndarray, float]):
        """
        Mollweide projection of the co-ordinates, see https://en.wikipedia.org/wiki/Mollweide_projection

        Parameters
        ----------
        ra
            Longitudes (RA in degrees)
        dec
            Latitudes (Dec in degrees)

        Returns
        -------
        x
            Projected RA
        y
            Projected DEC
        """
        r = np.pi / 2 / np.sqrt(2)
        theta = fnewton_solve(dec, dec)  # project
        x = r * (2 * np.sqrt(2)) / np.pi * ra * np.cos(theta)
        y = r * np.sqrt(2) * np.sin(theta)
        x, y = np.rad2deg([x, y])  # back to degrees
        return x, y

    ravalues: np.ndarray = all_results_full.ra.values  # all ra values
    decvalues: np.ndarray = all_results_full.dec.values  # all dec values
    allcoords = SkyCoord(ravalues, decvalues, unit='deg', frame='icrs')  # make astropy skycoord object
    ravalues = allcoords.galactic.l.value  # convert to galactic
    decvalues = allcoords.galactic.b.value  # convert to galactic
    ravalues -= 180  # shift position
    ravalues = np.array([np.abs(180 - raval) if raval >= 0 else -np.abs(raval + 180) for raval in ravalues])
    ravalues, decvalues = np.deg2rad([ravalues, decvalues])  # convert to radians
    raproj, decproj = project_mollweide(ravalues, decvalues)  # project to Mollweide
    return raproj, decproj


# website pathing
@app_simple.route('/')
@app_simple.route('/index', methods=['GET', 'POST'])
def index_page():
    """
    The main splash page
    """
    source_count = len(all_results)  # count the number of sources
    return render_template('index_simple.html', source_count=source_count)


@app_simple.route('/search', methods=['GET', 'POST'])
def search():
    """
    The searchbar page
    """
    form = SearchForm()  # searchbar
    query = form.search.data  # the content in searchbar
    if query is None:
        query = ''
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    results = db.search_object(query, fmt='astropy')  # get the results for that object
    results: Union[pd.DataFrame, None] = results.to_pandas()  # convert to pandas from astropy table
    sourcelinks: list = []  # empty list
    if len(results):
        for src in results.source.values:  # over every source in table
            urllnk = quote(src)  # convert object name to url safe
            srclnk = f'<a href="/solo_result/{urllnk}" target="_blank">{src}</a>'  # construct hyperlink
            sourcelinks.append(srclnk)  # add that to list
        results['source'] = sourcelinks  # update dataframe with the linked ones
        query = query.upper()  # convert contents of search bar to all upper case
        results: str = markdown(results.to_html(index=False, escape=False, max_rows=10,
                                                classes='table table-dark table-bordered table-striped'))
    else:
        results = None
    return render_template('search.html', form=form,
                           results=results, query=query)  # if everything not okay, return existing page as is


@app_simple.route('/solo_result/<query>')
def solo_result(query: str):
    """
    The result page for just one object when the query matches but one object

    Parameters
    ----------
    query: str
        The query -- full match to a main ID
    """
    curdoc().template_variables['query'] = query  # add query to bokeh curdoc
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    resultdict: dict = db.inventory(query)  # get everything about that object
    query = query.upper()  # convert query to all upper case
    everything = Inventory(resultdict)  # parsing the inventory into markdown
    scriptcmd, divcmd = camdplot(query, everything)
    scriptspectra, divspectra = specplot(query)
    return render_template('solo_result.html', resources=CDN.render(), scriptcmd=scriptcmd, divcmd=divcmd,
                           scriptspectra=scriptspectra, divspectra=divspectra,
                           query=query, resultdict=resultdict, everything=everything)


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


@app_simple.route('/multiplot')
def multiplotpage():
    """
    The page for all the plots
    """
    scriptmulti, divmulti = multiplotbokeh()
    return render_template('multiplot.html', scriptmulti=scriptmulti, divmulti=divmulti, resources=CDN.render())


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


@app_simple.route('/autocomplete', methods=['GET'])
def autocomplete():
    """
    Autocompleting function, id linked to the jquery which does the heavy lifting
    """
    return jsonify(alljsonlist=all_results)  # wraps all of the object names as a list, into a .json for server use


@app_simple.route('/feedback')
def feedback_page():
    """
    Page for directing users to github SIMPLE-web page
    """
    return render_template('feedback.html')


@app_simple.route('/schema')
def schema_page():
    """
    Page for directing users to github SIMPLE-db page
    """
    return render_template('schema.html')


if __name__ == '__main__':
    args = sysargs()  # get all system arguments
    db_file = f'sqlite:///{args.file}'  # the database file
    jscallbacks = JSCallbacks()
    all_results, all_results_full = all_sources()  # find all the objects once
    all_photo, all_bands = all_photometry()  # get all the photometry
    all_plx = all_parallaxes()
    app_simple.run(host=args.host, port=args.port, debug=args.debug)  # generate the application on server side
