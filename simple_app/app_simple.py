"""
This is the main script to be run from the directory root, it will start the Flask application running which one can
then connect to.
"""
# local packages
from .plots import *

# initialise
app_simple = Flask(__name__)  # start flask app
app_simple.config['SECRET_KEY'] = os.urandom(32)  # need to generate csrf token as basic security for Flask
app_simple.config['UPLOAD_FOLDER'] = 'tmp/'
CORS(app_simple)  # makes CORS work (aladin notably)


# website pathing
@app_simple.route('/')
@app_simple.route('/index', methods=['GET'])
def index_page():
    """
    The main splash/home page
    """
    # basic searchbar and source count
    source_count = len(all_results)
    form = BasicSearchForm(request.args)
    return render_template('index_simple.html', source_count=source_count, form=form, version_str=version_str)


@app_simple.route('/about')
def about():
    """
    The about page
    """
    return render_template('about.html')


@app_simple.route('/search', methods=['GET'])
def search():
    """
    The searchbar page
    """
    # initialisation, check contents of searchbars
    form = SearchForm(request.args)  # main searchbar
    if (ref_query := form.ref_search.data) is None:  # content in references searchbar
        ref_query = ''
    if (query := form.search.data) is None:  # content in main searchbar
        query = ''

    curdoc().template_variables['query'] = query  # add query to bokeh curdoc
    curdoc().template_variables['ref_query'] = ref_query  # add query to bokeh curdoc
    db = SimpleDB(db_file)  # open database

    # object search function
    try:
        results: Optional[pd.DataFrame] = db.search_object(query, fmt='pandas')  # get the results for that object

        if not len(results):
            raise IndexError('Empty dataframe from search')

    except (IndexError, OperationalError):
        stringed_results: Optional[str] = None

        # if search failed, return page as is
        return render_template('search.html', form=form, ref_query=ref_query,
                               results=stringed_results, query=query, version_str=version_str)

    # full text search function
    ref_results: Optional[Dict[str, pd.DataFrame]] = db.search_string(ref_query, fmt='pandas', verbose=False)
    try:
        ref_sources: pd.DataFrame = ref_results['Sources']

    # if Sources table not in returned dictionary, catch it
    except KeyError:
        stringed_results = None

    # otherwise, filter the Sources table by whatever is in the full text search
    else:
        filtered_results: Optional[pd.DataFrame] = results.merge(ref_sources, on='source', suffixes=(None, 'extra'))
        filtered_results.drop(columns=list(filtered_results.filter(regex='extra')), inplace=True)
        stringed_results = one_df_query(filtered_results)

    # return page with filtered Sources present
    return render_template('search.html', form=form, ref_query=ref_query, version_str=version_str,
                           results=stringed_results, query=query)  # if everything not okay, return existing page as is


@app_simple.route('/coordinate_query', methods=['GET', 'POST'])
def coordinate_query():
    """
    Wrapping the query by coordinate function
    """
    form = CoordQueryForm(db_file=db_file)

    # validating the form and returning results from query
    if form.validate_on_submit():

        if (query := form.query.data) is None:
            query = ''

        curdoc().template_variables['query'] = query
        db = SimpleDB(db_file)

        # parse query into ra, dec, radius
        ra, dec, radius = form.multi_param_str_parse(query)
        ra, dec, unit = form.ra_dec_unit_parse(ra, dec)
        c = SkyCoord(ra=ra, dec=dec, unit=unit)

        # submit query
        results: pd.DataFrame = db.query_region(c, fmt='pandas', radius=radius)  # query
        results = reference_handle(results, db_file)
        stringed_results = one_df_query(results)
        return render_template('coordinate_query.html', form=form, query=query, results=stringed_results,
                               version_str=version_str)

    else:
        return render_template('coordinate_query.html', form=form, results=None, query='', version_str=version_str)


@app_simple.route('/full_text_search', methods=['GET'])
def full_text_search():
    """
    Wrapping the search string function to search through all tables and return them
    """
    form = LooseSearchForm(request.args)
    limmaxrows = False

    if (query := form.search.data) is None:
        query = ''
        limmaxrows = True

    curdoc().template_variables['query'] = query
    db = SimpleDB(db_file)

    # search through the tables using the given query
    results: Dict[str, pd.DataFrame] = db.search_string(query, fmt='pandas', verbose=False)
    resultsout = multi_df_query(results, db_file, limmaxrows)

    return render_template('full_text_search.html', form=form, version_str=version_str,
                           results=resultsout, query=query)


@app_simple.route('/load_full_text')
def load_full_text():
    """
    Loading full text search page
    """
    return render_template('load_fulltext.html', version_str=version_str)


@app_simple.route('/raw_query', methods=['GET', 'POST'])
def raw_query():
    """
    Page for raw sql query, returning all tables
    """
    db = SimpleDB(db_file)  # open database
    form = SQLForm(db_file=db_file)  # main query form

    # checks that the SQL is valid, then submits form
    if form.validate_on_submit():

        if (query := form.sqlfield.data) is None:
            query = ''

        curdoc().template_variables['query'] = query

        # attempt to query the database
        try:
            results: Optional[pd.DataFrame] = db.sql_query(query, fmt='pandas')

        # catch any broken queries (should not activate as will be caught by validation)
        except (ResourceClosedError, OperationalError, IndexError, SqliteWarning, BadSQLError):
            results = pd.DataFrame()

        results = reference_handle(results, db_file, True)
        stringed_results = one_df_query(results)
        return render_template('raw_query.html', form=form, results=stringed_results, version_str=version_str)

    else:
        return render_template('raw_query.html', form=form, results=None, query='', version_str=version_str)


@app_simple.route('/solo_result/<query>')
def solo_result(query: str):
    """
    The result page for just one object when the query matches but one object

    Parameters
    ----------
    query: str
        The query -- full match to a main ID
    """
    curdoc().template_variables['query'] = query
    db = SimpleDB(db_file)

    # search database for given object
    try:
        resultdict: dict = db.inventory(query)
        if not len(resultdict):
            raise KeyError
    except KeyError:
        abort(404, f'"{query}" does match any result in SIMPLE!')
        return
    everything = Inventory(resultdict, db_file)

    # create camd and spectra plots
    scriptcmd, divcmd = camd_plot(query, everything, all_bands, all_results_full, all_parallaxes, all_spectral_types,
                                  photometric_filters, all_photometry, js_callbacks, night_sky_theme, db_file)
    scriptspectra, divspectra, nfail, failstr = spectra_plot(query, db_file, night_sky_theme, js_callbacks)

    query = query.upper()
    return render_template('solo_result.html', resources=CDN.render(), scriptcmd=scriptcmd, divcmd=divcmd,
                           scriptspectra=scriptspectra, divspectra=divspectra, nfail=nfail, failstr=failstr,
                           query=query, resultdict=resultdict, everything=everything, version_str=version_str)


@app_simple.route('/load_solo/<query>')
def load_solo_page(query: str):
    """
    Loading solo result page
    """
    return render_template('load_solo.html', query=query, version_str=version_str)


@app_simple.route('/multi_plot')
def multi_plot_page():
    """
    The page for all the plots
    """
    scriptmulti, divmulti = multi_plot_bokeh(all_results_full, all_bands, all_photometry, all_parallaxes,
                                             all_spectral_types, js_callbacks, night_sky_theme)
    return render_template('multi_plot.html', scriptmulti=scriptmulti, divmulti=divmulti, resources=CDN.render(),
                           version_str=version_str)


@app_simple.route('/load_multi_plot')
def load_multi_plot():
    """
    Loading multiplot page
    """
    return render_template('load_multi_plot.html', version_str=version_str)


@app_simple.route('/autocomplete', methods=['GET'])
def autocomplete():
    """
    Autocompleting function, id linked to the jquery which does the heavy lifting
    """
    return jsonify(alljsonlist=all_results)


@app_simple.errorhandler(HTTPException)
def bad_request(e):
    """
    Handling bad HTTP requests such as a 404 error

    Parameters
    ----------
    e
        The HTTP status code
    """
    # handling 404 file not found errors
    if e.code == 404:
        # all different website routes and requested route
        all_routes = ['about', 'search', 'full_text_search', 'load_full_text', 'coordinate_query', 'raw_query',
                      'multi_plot', 'load_multi_plot']
        requested_route = request.path.strip('/')

        # get best match of path and redirect
        best_match = get_close_matches(requested_route, all_routes, 1)
        if best_match:
            return redirect(url_for(best_match[0]))

    # any other error codes or no good match found
    return render_template('bad_request.html', e=e), e.code


@app_simple.route('/write/<key>.csv', methods=['GET'])
def create_file_for_download(key: str):
    """
    Creates and downloads the shown dataframe from solo results

    Parameters
    ----------
    key: str
        The dataframe string
    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file)

    # search for a given object and a given key
    resultdict: dict = db.inventory(query)
    everything = Inventory(resultdict, db_file, return_markdown=False)

    # writes table to csv
    if key in resultdict:
        results: pd.DataFrame = getattr(everything, key.lower())
        response = Response(write_file(results), mimetype='text/csv')
        response = control_response(response, key)
        return response

    abort(400, 'Could not write table')


@app_simple.route('/write_solo_all', methods=['GET'])
def create_files_for_solo_download():
    """
    Creates and downloads all dataframes from solo results
    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file)

    # search for a given object
    resultdict: dict = db.inventory(query)
    resultdictnew: Dict[str, pd.DataFrame] = {}

    for key, obj in resultdict.items():
        df: pd.DataFrame = pd.concat([pd.DataFrame(objrow, index=[i])  # create dataframe from found dict
                                      for i, objrow in enumerate(obj)], ignore_index=True)  # every dict in the list
        resultdictnew[key] = df

    # write all tables to zipped csvs
    response = Response(write_multi_files(resultdictnew), mimetype='application/zip')
    response = control_response(response, app_type='zip')
    return response


@app_simple.route('/write_spectra', methods=['GET'])
def create_spectra_files_for_download():
    """
    Downloads the spectra files and zips together
    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file)

    # search for a given object and specifically its spectra
    resultdict: dict = db.inventory(query)
    everything = Inventory(resultdict, db_file, return_markdown=False)
    results: pd.DataFrame = getattr(everything, 'spectra')

    # write all spectra for object to zipped file
    zipped = write_spec_files(results.access_url.values)
    if zipped is not None:
        response = Response(zipped, mimetype='application/zip')
        response = control_response(response, app_type='zip')
        return response

    abort(400, 'Could not download fits')


@app_simple.route('/write_filt', methods=['GET'])
def create_file_for_filtered_download():
    """
    Creates and downloads the shown dataframe when in filtered search
    """
    query = curdoc().template_variables['query']
    refquery = curdoc().template_variables['ref_query']
    db = SimpleDB(db_file)

    # search for a given object and a full text search at the same time
    results: Optional[pd.DataFrame] = db.search_object(query, fmt='pandas')
    ref_results: Optional[dict] = db.search_string(refquery, fmt='pandas', verbose=False)
    ref_sources: pd.DataFrame = ref_results['Sources']

    # filter the search by the reference search
    filtered_results: Optional[pd.DataFrame] = results.merge(ref_sources, on='source', suffixes=(None, 'extra'))
    filtered_results.drop(columns=list(filtered_results.filter(regex='extra')), inplace=True)

    # write to a csv
    response = Response(write_file(filtered_results), mimetype='text/csv')
    response = control_response(response)
    return response


@app_simple.route('/write_coord', methods=['GET', 'POST'])
def create_file_for_coordinate_download():
    """
    Creates and downloads the shown dataframe when in coordinate search
    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file)

    # query the database for given coordinates and parse those coordinates
    form = CoordQueryForm(db_file=db_file)
    ra, dec, radius = form.multi_param_str_parse(query)
    ra, dec, unit = form.ra_dec_unit_parse(ra, dec)
    c = SkyCoord(ra=ra, dec=dec, unit=unit)
    results: pd.DataFrame = db.query_region(c, fmt='pandas', radius=radius)

    # write results to a csv
    response = Response(write_file(results), mimetype='text/csv')
    response = control_response(response)
    return response


@app_simple.route('/write_full/<key>', methods=['GET'])
def create_file_for_full_download(key: str):
    """
    Creates and downloads the shown dataframe when in unrestrained search

    Parameters
    ----------
    key: str
        The dataframe string
    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file)

    # search database with a free-text search and specific table
    resultdict: Dict[str, pd.DataFrame] = db.search_string(query, fmt='pandas', verbose=False)

    # write to csv
    if key in resultdict:
        results: pd.DataFrame = resultdict[key]
        response = Response(write_file(results), mimetype='text/csv')
        response = control_response(response, key)
        return response

    abort(400, 'Could not write table')


@app_simple.route('/write_all', methods=['GET'])
def create_files_for_multi_download():
    """
    Creates and downloads all dataframes from full results
    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file)

    # search with full-text search
    resultdict: Dict[str, pd.DataFrame] = db.search_string(query, fmt='pandas', verbose=False)

    # write all returned tables to zipped file of csvs
    response = Response(write_multi_files(resultdict), mimetype='application/zip')
    response = control_response(response, app_type='zip')
    return response


@app_simple.route('/write_sql', methods=['GET', 'POST'])
def create_file_for_sql_download():
    """
    Creates and downloads the shown dataframe when in sql query
    """
    query = curdoc().template_variables['query']
    db = SimpleDB(db_file)

    # query database via sql
    results: Optional[pd.DataFrame] = db.sql_query(query, fmt='pandas')

    # write results to a csv
    response = Response(write_file(results), mimetype='text/csv')
    response = control_response(response)
    return response


args, db_file, photometric_filters, all_results, all_results_full, version_str, \
    all_photometry, all_bands, all_parallaxes, all_spectral_types = main_utils()
night_sky_theme, js_callbacks = main_plots()

if __name__ == '__main__':
    app_simple.run(host=args.host, port=args.port, debug=args.debug)  # generate the application on server side
