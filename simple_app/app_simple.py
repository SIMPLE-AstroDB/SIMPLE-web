"""
This is the main script to be run from the directory root, it will start the Flask application running which one can
then connect to.
"""
# external packages
from astrodbkit2.astrodb import Database, REFERENCE_TABLES  # used for pulling out database and querying
from astropy.table import Table  # tabulating
from bokeh.embed import json_item
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Range1d, CustomJS, Select, Toggle
from bokeh.plotting import figure, curdoc
from flask import Flask, render_template, request, redirect, url_for, jsonify  # website functionality
from flask_cors import CORS  # cross origin fix (aladin mostly)
from flask_wtf import FlaskForm  # web forms
from markdown2 import markdown  # using markdown formatting
import numpy as np  # numerical python
import pandas as pd  # running dataframes
from wtforms import StringField, SubmitField  # web forms
from wtforms.validators import DataRequired, StopValidation  # validating web forms
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
        if rtnmk:  # return markdown boolean
            return markdown(df.to_html(index=False))  # wrap the dataframe into html then markdown
        return df  # otherwise return dataframe as is


class CheckResultsLength(object):
    """
    Validation class for use in the searchbar
    """
    def __call__(self, form, field):
        """
        Runs when class called

        Parameters
        ----------
        form
            The form object
        field
            Current values in the form
        """
        db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
        results = db.search_object(field.data, fmt='astropy')  # search by what is currently in searchbar
        if not len(results):  # if that search is empty
            field.errors[:] = []  # clear existing errors
            raise StopValidation(field.gettext('No results'))  # stop validating and return error


class SearchForm(FlaskForm):
    """
    Searchbar class
    """
    search = StringField('', [DataRequired(), CheckResultsLength()], id='autocomplete')  # searchbar
    submit = SubmitField('Query')  # clicker button to send request


def all_sources():
    """
    Queries the full table to get all the sources

    Returns
    -------
    allresults
        Just the main IDs
    """
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    allresults: list = db.query(db.Sources).table()['source'].tolist()  # gets all the main IDs in the database
    return allresults


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
            photodf[f'{band}_{nextband}'] = photodf[band] - photodf[nextband]  # colour
            j += 1
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
    allphoto = pd.DataFrame(outphoto)  # use rearranged dataframe
    allphoto = find_colours(allphoto, allbands)  # get the colours
    return allphoto, allbands


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
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    results = db.search_object(query, fmt='astropy')  # get the results for that object
    if request.method == 'POST' and form.validate_on_submit():  # if everything okay with the search
        if len(results) > 1:  # if more than one result
            return redirect((url_for('search_results', query=query)))  # return table of all results
        return redirect((url_for('solo_result', query=query)))  # otherwise return page for that one object
    return render_template('search.html', form=form)  # if everything not okay, return existing page as is


@app_simple.route('/search_results/<query>')
def search_results(query: str):
    """
    The tabulated page for all the sources matching a given query string

    Parameters
    ----------
    query: str
        The query -- partial match of main IDs
    """
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    results: Table = db.search_object(query, fmt='astropy')  # get all results for that object
    results: pd.DataFrame = results.to_pandas()  # convert to pandas from astropy table
    sourcelinks: list = []  # empty list
    for src in results.source.values:  # over every source in table
        urllnk = quote(src)  # convert object name to url safe
        srclnk = f'<a href="/solo_result/{urllnk}" target="_blank">{src}</a>'  # construct hyperlink
        sourcelinks.append(srclnk)  # add that to list
    results['source'] = sourcelinks  # update dataframe with the linked ones
    query = query.upper()  # convert contents of search bar to all upper case
    results: str = markdown(results.to_html(index=False, escape=False))  # convert results into markdown
    return render_template('search_results.html', query=query, results=results)


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
    return render_template('solo_result.html',
                           query=query, resultdict=resultdict, everything=everything)


@app_simple.route('/camdplot')
def camdplot():
    """
    Creates CAMD plot as JSON object

    Returns
    -------
    plot
        JSON object for page to use
    """
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    query: str = curdoc().template_variables['query']  # get the query (on page)
    resultdict: dict = db.inventory(query)  # get everything about that object
    everything = Inventory(resultdict)  # parsing the inventory
    bands = [band.split("_")[1] for band in all_bands]  # nice band names
    vals = [f'@{band}' for band in all_bands]  # the values in CDS
    tooltips = [('Target', '@target'), *zip(bands, vals), ('Ref', '@ref')]  # tooltips for hover tool
    p = figure(title='CAMD', plot_width=800, plot_height=400, active_scroll='wheel_zoom', active_drag='box_zoom',
               tools='pan,wheel_zoom,box_zoom,hover,tap,reset', tooltips=tooltips,
               sizing_mode='stretch_width')
    cdsfull = ColumnDataSource(data=all_photo)  # bokeh cds object
    fullplot = p.circle_x(x='WISE_W1_WISE_W2', y='WISE_W3_WISE_W4', source=cdsfull,
                          color='gray', alpha=0.5, size=5)  # plot all objects
    try:
        thisphoto: pd.DataFrame = everything.listconcat('Photometry', False)  # the photometry for this object
    except KeyError:  # no photometry for this object
        thisplot = None
        pass
    else:
        newphoto: dict = parse_photometry(thisphoto, all_bands)  # transpose photometric table
        newphoto['target'] = [query, ] * len(newphoto['ref'])
        thisphoto: pd.DataFrame = find_colours(pd.DataFrame(newphoto), all_bands)  # get the colours
        thiscds = ColumnDataSource(data=thisphoto)  # this object cds
        thisplot = p.circle(x='WISE_W1_WISE_W2', y='WISE_W3_WISE_W4', source=thiscds,
                            color='blue', size=10)  # plot for this object
    p.x_range = Range1d(all_photo.WISE_W1_WISE_W2.min(), all_photo.WISE_W1_WISE_W2.max())  # x limits
    p.y_range = Range1d(all_photo.WISE_W3_WISE_W4.min(), all_photo.WISE_W3_WISE_W4.max())  # y limits
    p.xaxis.axis_label = 'W1 - W2'  # x label
    p.yaxis.axis_label = 'W3 - W4'  # y label
    buttonxflip = Toggle(label='X Flip')
    buttonxflip.js_on_click(CustomJS(code=jscallbacks.button_flip, args={'axrange': p.x_range}))
    buttonyflip = Toggle(label='Y Flip')
    buttonyflip.js_on_click(CustomJS(code=jscallbacks.button_flip, args={'axrange': p.y_range}))
    just_colours: pd.DataFrame = all_photo.drop(columns=np.hstack([all_bands, ['ref', 'target']]))  # only the colours
    axis_names = [f'{col.split("_")[1]} - {col.split("_")[3]}' for col in just_colours.columns]  # convert nicely
    dropmenu = [*zip(just_colours.columns, axis_names), ]  # zip up into menu
    dropdownx = Select(title='X Axis', options=dropmenu, value='WISE_W1_WISE_W2', width=200, height=50)  # x axis select
    dropdownx.js_on_change('value', CustomJS(code=jscallbacks.dropdownx_js,
                                             args={'fullplot': fullplot, 'thisplot': thisplot,
                                                   'fulldata': cdsfull.data, 'xbut': buttonxflip,
                                                   'xaxis': p.xaxis[0], 'xrange': p.x_range}))
    dropdowny = Select(title='Y Axis', options=dropmenu, value='WISE_W3_WISE_W4', width=200, height=50)  # y axis select
    dropdowny.js_on_change('value', CustomJS(code=jscallbacks.dropdowny_js,
                                             args={'fullplot': fullplot, 'thisplot': thisplot,
                                                   'fulldata': cdsfull.data, 'ybut': buttonyflip,
                                                   'yaxis': p.yaxis[0], 'yrange': p.y_range}))
    plot = jsonify(json_item(row(p, column(dropdownx,
                                           dropdowny,
                                           row(buttonxflip, buttonyflip, sizing_mode='scale_width'),
                                           sizing_mode='fixed', width=200, height=200),
                                 sizing_mode='scale_width'), 'camdplot'))  # bokeh object in json
    return plot


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
    all_results = all_sources()  # find all the objects once
    all_photo, all_bands = all_photometry()  # get all the photometry
    app_simple.run(host=args.host, port=args.port, debug=args.debug)  # generate the application on server side
