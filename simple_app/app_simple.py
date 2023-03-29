"""
This is the main script to be run from the directory root, it will start the Flask application running which one can
then connect to.
"""
import sys
sys.path.append('simple_root/simple_app')
# local packages
from plots import *
from utils import *

# initialise
app_simple = Flask(__name__)  # start flask app
app_simple.config['SECRET_KEY'] = os.urandom(32)  # need to generate csrf token as basic security for Flask
app_simple.config['UPLOAD_FOLDER'] = 'tmp/'
CORS(app_simple)  # makes CORS work (aladin notably)


# website pathing
@app_simple.route('/')
@app_simple.route('/index', methods=['GET', 'POST'])
def index_page():
    """
    The main splash page
    """
    source_count = len(all_results)  # count the number of sources
    form = BasicSearchForm()  # main searchbar
    return render_template('index_simple.html', source_count=source_count, form=form, version_str=version_str)


@app_simple.route('/search', methods=['GET', 'POST'])
def search():
    """
    The searchbar page
    """
    form = SearchForm()  # main searchbar
    if (refquery := form.refsearch.data) is None:  # content in references searchbar
        refquery = ''
    if (query := form.search.data) is None:  # content in main searchbar
        query = ''
    curdoc().template_variables['query'] = query  # add query to bokeh curdoc
    curdoc().template_variables['refquery'] = refquery  # add query to bokeh curdoc
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    try:
        results: Optional[pd.DataFrame] = db.search_object(query, fmt='pandas')  # get the results for that object
        if not len(results):
            raise IndexError('Empty dataframe from search')
    except (IndexError, OperationalError):
        stringed_results: Optional[str] = None
        return render_template('search.html', form=form, refquery=refquery,
                               results=stringed_results, query=query, version_str=version_str)
    except IndexError:
        results = pd.DataFrame()
    refresults: Optional[dict] = db.search_string(refquery, fmt='pandas', verbose=False)  # search all the strings
    try:
        refsources: pd.DataFrame = refresults['Sources']
    except KeyError:
        stringed_results = None
    else:
        filtered_results: Optional[pd.DataFrame] = results.merge(refsources, on='source', suffixes=(None, 'extra'))
        filtered_results.drop(columns=list(filtered_results.filter(regex='extra')), inplace=True)
        stringed_results = onedfquery(filtered_results)
    return render_template('search.html', form=form, refquery=refquery, version_str=version_str,
                           results=stringed_results, query=query)  # if everything not okay, return existing page as is


@app_simple.route('/coordquery', methods=['GET', 'POST'])
def coordquery():
    """
    Wrapping the query by coordinate function
    """
    form = CoordQueryForm(db_file=db_file)
    if form.validate_on_submit():
        if (query := form.query.data) is None:  # content in main searchbar
            query = ''
        curdoc().template_variables['query'] = query  # add query to bokeh curdoc
        db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
        ra, dec, radius = multi_param_str_parse(query)
        ra, dec, unit = ra_dec_unit_parse(ra, dec)
        c = SkyCoord(ra=ra, dec=dec, unit=unit)
        results: pd.DataFrame = db.query_region(c, fmt='pandas', radius=radius)  # query
        stringed_results = onedfquery(results)
        return render_template('coordquery.html', form=form, query=query, results=stringed_results,
                               version_str=version_str)
    else:
        return render_template('coordquery.html', form=form, results=None, query='', version_str=version_str)


@app_simple.route('/fulltextsearch', methods=['GET', 'POST'])
def fulltextsearch():
    """
    Wrapping the search string function to search through all tables and return them
    """
    form = LooseSearchForm()  # main searchbar
    limmaxrows = False
    if (query := form.search.data) is None:  # content in main searchbar
        query = ''
        limmaxrows = True
    curdoc().template_variables['query'] = query  # add query to bokeh curdoc
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    results: Dict[str, pd.DataFrame] = db.search_string(query, fmt='pandas', verbose=False)  # search
    resultsout = multidfquery(results, limmaxrows)
    return render_template('fulltextsearch.html', form=form, version_str=version_str,
                           results=resultsout, query=query)  # if everything not okay, return existing page


@app_simple.route('/load_fulltext')
def load_fulltext():
    """
    Loading full text search page
    """
    return render_template('load_fulltext.html', version_str=version_str)


@app_simple.route('/raw_query', methods=['GET', 'POST'])
def raw_query():
    """
    Page for raw sql query, returning all tables
    """
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    form = SQLForm(db_file=db_file)  # main query form
    if form.validate_on_submit():
        if (query := form.sqlfield.data) is None:  # content in main searchbar
            query = ''
        curdoc().template_variables['query'] = query  # add query to bokeh curdoc
        try:
            results: Optional[pd.DataFrame] = db.sql_query(query, fmt='pandas')
        except (ResourceClosedError, OperationalError, IndexError, SqliteWarning, BadSQLError):
            results = pd.DataFrame()
        stringed_results = onedfquery(results)
        return render_template('rawquery.html', form=form, results=stringed_results, version_str=version_str)
    else:
        return render_template('rawquery.html', form=form, results=None, query='', version_str=version_str)


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
    everything = Inventory(resultdict, args)  # parsing the inventory into markdown
    scriptcmd, divcmd = camdplot(query, everything, all_bands, all_results_full, all_plx, all_spts, photfilters,
                                 all_photo, jscallbacks, nightskytheme)
    scriptspectra, divspectra, nfail, failstr = specplot(query, db_file, nightskytheme, jscallbacks)
    query = query.upper()  # convert query to all upper case
    return render_template('solo_result.html', resources=CDN.render(), scriptcmd=scriptcmd, divcmd=divcmd,
                           scriptspectra=scriptspectra, divspectra=divspectra, nfail=nfail, failstr=failstr,
                           query=query, resultdict=resultdict, everything=everything, version_str=version_str)


@app_simple.route('/load_solo/<query>')
def load_solopage(query: str):
    """
    Loading solo result page
    """
    return render_template('load_solo.html', query=query, version_str=version_str)


@app_simple.route('/multiplot')
def multiplotpage():
    """
    The page for all the plots
    """
    scriptmulti, divmulti = multiplotbokeh(all_results_full, all_bands, all_photo, all_plx, all_spts,
                                           jscallbacks, nightskytheme)
    return render_template('multiplot.html', scriptmulti=scriptmulti, divmulti=divmulti, resources=CDN.render(),
                           version_str=version_str)


@app_simple.route('/load_multiplot')
def load_multiplot():
    """
    Loading multiplot page
    """
    return render_template('load_multiplot.html', version_str=version_str)


@app_simple.route('/autocomplete', methods=['GET'])
def autocomplete():
    """
    Autocompleting function, id linked to the jquery which does the heavy lifting
    """
    return jsonify(alljsonlist=all_results)  # wraps all of the object names as a list, into a .json for server use


@app_simple.errorhandler(HTTPException)
def bad_request(e):
    """
    Handling bad HTTP requests such as a 404 error

    Parameters
    ----------
    e
        The HTTP status code
    """
    return render_template('bad_request.html', e=e), 500


@app_simple.route('/write/<key>.csv', methods=['GET', 'POST'])
def create_file_for_download(key: str):
    """
    Creates and downloads the shown dataframe from solo results

    Parameters
    ----------
    key: str
        The dataframe string
    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    resultdict: dict = db.inventory(query)  # get everything about that object
    everything = Inventory(resultdict, args, rtnmk=False)
    if key in resultdict:
        results: pd.DataFrame = getattr(everything, key.lower())
        response = Response(write_file(results), mimetype='text/csv')
        response = control_response(response, key)
        return response
    return None


@app_simple.route('/write_soloall', methods=['GET', 'POST'])
def create_files_for_solodownload():
    """
    Creates and downloads all dataframes from solo results
    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    resultdict: dict = db.inventory(query)  # get everything about that object
    resultdictnew: Dict[str, pd.DataFrame] = {}
    for key, obj in resultdict.items():
        df: pd.DataFrame = pd.concat([pd.DataFrame(objrow, index=[i])  # create dataframe from found dict
                                      for i, objrow in enumerate(obj)], ignore_index=True)  # every dict in the list
        resultdictnew[key] = df
    response = Response(write_multifiles(resultdictnew), mimetype='application/zip')
    response = control_response(response, apptype='zip')
    return response


@app_simple.route('/write_spectra', methods=['GET', 'POST'])
def create_spectrafile_for_download():
    """
    Downloads the spectra files and zips together
    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    resultdict: dict = db.inventory(query)  # get everything about that object
    everything = Inventory(resultdict, args, rtnmk=False)
    results: pd.DataFrame = getattr(everything, 'spectra')
    response = Response(write_fitsfiles(results.spectrum.values), mimetype='application/zip')
    response = control_response(response, apptype='zip')
    return response


@app_simple.route('/write_filt', methods=['GET', 'POST'])
def create_file_for_filtdownload():
    """
    Creates and downloads the shown dataframe when in filtered search
    """
    query = curdoc().template_variables['query']
    refquery = curdoc().template_variables['refquery']
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    results: Optional[pd.DataFrame] = db.search_object(query, fmt='pandas')  # get the results for that object
    refresults: Optional[dict] = db.search_string(refquery, fmt='pandas', verbose=False)  # search all the strings
    refsources: pd.DataFrame = refresults['Sources']
    filtered_results: Optional[pd.DataFrame] = results.merge(refsources, on='source', suffixes=(None, 'extra'))
    filtered_results.drop(columns=list(filtered_results.filter(regex='extra')), inplace=True)
    response = Response(write_file(filtered_results), mimetype='text/csv')
    response = control_response(response)
    return response


@app_simple.route('/write_coord', methods=['GET', 'POST'])
def create_file_for_coorddownload():
    """
    Creates and downloads the shown dataframe when in coordinate search
    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    ra, dec, radius = multi_param_str_parse(query)
    ra, dec, unit = ra_dec_unit_parse(ra, dec)
    c = SkyCoord(ra=ra, dec=dec, unit=unit)
    results: pd.DataFrame = db.query_region(c, fmt='pandas', radius=radius)  # query
    response = Response(write_file(results), mimetype='text/csv')
    response = control_response(response)
    return response


@app_simple.route('/write_full/<key>', methods=['GET', 'POST'])
def create_file_for_fulldownload(key: str):
    """
    Creates and downloads the shown dataframe when in unrestrained search

    Parameters
    ----------
    key: str
        The dataframe string
    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    resultdict: Dict[str, pd.DataFrame] = db.search_string(query, fmt='pandas', verbose=False)  # search
    if key in resultdict:
        results: pd.DataFrame = resultdict[key]
        response = Response(write_file(results), mimetype='text/csv')
        response = control_response(response, key)
        return response
    return None


@app_simple.route('/write_all', methods=['GET', 'POST'])
def create_files_for_multidownload():
    """
    Creates and downloads all dataframes from full results

    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    resultdict: Dict[str, pd.DataFrame] = db.search_string(query, fmt='pandas', verbose=False)  # search
    response = Response(write_multifiles(resultdict), mimetype='application/zip')
    response = control_response(response, apptype='zip')
    return response


@app_simple.route('/write_sql', methods=['GET', 'POST'])
def create_file_for_sqldownload():
    """
    Creates and downloads the shown dataframe when in sql query
    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    results: Optional[pd.DataFrame] = db.sql_query(query, fmt='pandas')
    response = Response(write_file(results), mimetype='text/csv')
    response = control_response(response)
    return response


args, db_file, photfilters, all_results, all_results_full, version_str,\
    all_photo, all_bands, all_plx, all_spts = mainutils()
nightskytheme, jscallbacks = mainplots()

if __name__ == '__main__':
    app_simple.run(host=args.host, port=args.port, debug=args.debug)  # generate the application on server side
