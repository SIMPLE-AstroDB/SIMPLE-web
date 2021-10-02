"""
This is the main script to be run from the directory root, it will start the Flask application running which one can
then connect to.
"""
# external packages
from bokeh.plotting import curdoc  # bokeh plotting
from bokeh.resources import CDN
from flask import Flask, render_template, jsonify  # website functionality
from flask_cors import CORS  # cross origin fix (aladin mostly)
# internal packages
import os  # operating system
from urllib.parse import quote  # handling strings into url friendly form
# local packages
from plots import *
from utils import *

# initialise
app_simple = Flask(__name__)  # start flask app
app_simple.config['SECRET_KEY'] = os.urandom(32)  # need to generate csrf token as basic security for Flask
CORS(app_simple)  # makes CORS work (aladin notably)


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
    # FIXME: FYI David, doing the to_pandas method because giving fmt='pandas' produces df without col names
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
    everything = Inventory(resultdict, args)  # parsing the inventory into markdown
    scriptcmd, divcmd = camdplot(query, everything, all_bands, all_results_full, all_plx,
                                 all_photo, jscallbacks, nightskytheme)
    scriptspectra, divspectra = specplot(query, db_file, nightskytheme)
    return render_template('solo_result.html', resources=CDN.render(), scriptcmd=scriptcmd, divcmd=divcmd,
                           scriptspectra=scriptspectra, divspectra=divspectra,
                           query=query, resultdict=resultdict, everything=everything)


@app_simple.route('/multiplot')
def multiplotpage():
    """
    The page for all the plots
    """
    scriptmulti, divmulti = multiplotbokeh(all_results_full, all_bands, all_photo, all_plx, jscallbacks, nightskytheme)
    return render_template('multiplot.html', scriptmulti=scriptmulti, divmulti=divmulti, resources=CDN.render())


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
    args, db_file, all_results, all_results_full, all_photo, all_bands, all_plx = mainutils()
    nightskytheme, jscallbacks = mainplots()
    app_simple.run(host=args.host, port=args.port, debug=args.debug)  # generate the application on server side
