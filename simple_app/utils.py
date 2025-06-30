"""
The static functions for various calculations and required parameters
"""
from .simports import *


def sys_args():
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
    _args.add_argument('-f', '--file', default='SIMPLE.sqlite',
                       help='Database file path relative to current directory, default SIMPLE.sqlite')
    _args = _args.parse_args()
    return _args


class SimpleDB(Database):  # this keeps pycharm happy about unresolved references
    """
    Wrapper class for astrodbkit2.Database specific to SIMPLE
    """
    # all class attributes are initialised here
    Sources = None
    Photometry = None
    Parallaxes = None
    Spectra = None
    PhotometryFilters = None
    Versions = None
    SpectralTypes = None
    CompanionRelationships = None

    def __init__(self, connection_string):
        super().__init__(connection_string,
                         reference_tables=REFERENCE_TABLES,
                         connection_arguments={'check_same_thread': False})


class Inventory:
    """
    For use in the solo result page where the inventory of an object is queried, grabs also the RA & Dec
    """
    ra: float = 0
    dec: float = 0

    def __init__(self, d_result: Dict[str, List[Dict[str, List[Union[str, float, int]]]]], **kwargs):
        """
        Constructor method for Inventory

        Parameters
        ----------
        d_result: Dict[str, List[Dict[str, List[Union[str, float, int]]]
            The dictionary of all the key: values in a given object inventory
        """
        self.results: Dict[str, List[Dict[str, List[Union[str, float, int]]]]] = d_result

        # look through each table in inventory
        for key in self.results:  # over every key in inventory

            # ignore reference tables like PhotometryFilters
            if key in REFERENCE_TABLES:  # ignore the reference table ones
                continue

            # convert the table result to Markdown
            low_key: str = key.lower()
            markdown_output: str = self.list_concat(key, **kwargs)
            setattr(self, low_key, markdown_output)

        # retrieve ra and dec from the Sources table, if present
        try:
            sources: pd.DataFrame = self.list_concat('Sources', return_markdown=False)
            self.ra, self.dec = sources.ra[0], sources.dec[0]
        except (KeyError, AttributeError):
            pass
        return

    @staticmethod
    def spectra_handle(df: pd.DataFrame, drop_source: bool = True, multi_obj: bool = False):
        """
        Handles spectra, converting files to links

        Parameters
        ----------
        df: pd.DataFrame
            The table for the spectra
        drop_source: bool
            Switch to keep source in the dataframe
        multi_obj: bool
            Switch on multiple objects being looked at or just one individual object

        Returns
        -------
        df: pd.DataFrame
            The edited table
        """

        # convert links to spectra files from plaintext to hyperlinks
        url_links = []
        for source in df.access_url.values:
            source_link = f'<a href="{source}" target="_blank">Link</a>'
            url_links.append(source_link)

        # removing excess columns which aren't pretty on the website
        df.drop(columns=[col for col in df.columns if
                         any([sub_string in col for sub_string in ('wave', 'flux', 'original')])],
                inplace=True)
        drop_cols = ['access_url', 'local_spectrum', 'regime']
        if drop_source:
            drop_cols.append('source')
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

        # editing dataframe with nicely formatted columns
        if multi_obj:
            href_path = 'write_multi_spectra'
        else:
            href_path = 'write_spectra'
        df[f'<a href="/{href_path}" target="_blank">download</a>'] = url_links
        df['observation_date'] = df['observation_date'].dt.date
        return df

    def list_concat(self, key: str, return_markdown: bool = True) -> Union[pd.DataFrame, str]:
        """
        Concatenates the list for a given key

        Parameters
        ----------
        key: str
            The key corresponding to the inventory
        return_markdown: bool
            Switch for whether to return either a markdown string or a dataframe

        Returns
        -------
        df: Union[pd.DataFrame, str]
            Either the dataframe for a given key or the markdown parsed string
        """
        # construct dataframe for a given key corresponding to a table
        obj = self.results[key]
        df_list = []

        # check each row for NaNs, only append to list if not wholly NaNs
        for i, obj_row in enumerate(obj):
            df_row = pd.DataFrame(obj_row, index=[i]).dropna(axis=0, how='all')

            if len(df_row):
                df_list.append(df_row)

        # create a concatenated dataframe of all the rows in a given table
        if len(df_list):
            df: pd.DataFrame = pd.concat(df_list, ignore_index=True)
        else:
            df = pd.DataFrame(columns=list(obj[0].keys()))

        # switch whether to have a Markdown version of the table, or a normal DataFrame
        if return_markdown:
            if key == 'Spectra':
                df = self.spectra_handle(df)
            df.rename(columns={s: s.replace('_', ' ') for s in df.columns if 'download' not in s}, inplace=True)
            return markdown(df.to_html(index=False, escape=False,
                                       classes='table table-dark table-bordered table-striped'))
        return df


class CSRFOverride(FlaskForm):
    """
    Overriding the CSRF protection on forms we want to GET instead of POST
    """
    class Meta:
        csrf = False


class SearchForm(CSRFOverride):
    """
    Basic search, combined with full text filtering
    """
    search = StringField('Search for an object:', id='mainsearchfield')  # searchbar
    ref_search = StringField('Filter by full text search:', id='refsearchfield')  # full text search
    submit = SubmitField('Search', id='querybutton')  # clicker button to send request


class BasicSearchForm(CSRFOverride):
    """
    Most basic searchbar
    """
    search = StringField('Search for an object:', id='mainsearchfield')  # searchbar
    submit = SubmitField('Search', id='querybutton')  # clicker button to send request


class CoordQueryForm(CSRFOverride):
    """
    Class for searching by coordinate
    """
    query = StringField('Query by coordinate within radius:', id='mainsearchfield')  # searchbar
    submit = SubmitField('Query', id='querybutton')  # clicker button to send request

    def __init__(self, *args, **kwargs: str):
        super(CoordQueryForm, self).__init__(*args, **kwargs)
        self.db_file: str = kwargs['db_file']
        return

    @staticmethod
    def multi_param_str_parse(s: str) -> Optional[Tuple[str, str, float]]:
        """
        Parses a string to split into two or three parameters separated by N spaces

        Parameters
        ----------
        s: str
            Input string

        Returns
        -------
        a
            First output
        b
            Second output
        c
            Third output (optional)
        """
        # split up the string by empty space
        try:
            query_split = s.lower().split()
            query_length = len(query_split)

            # check length is 2 or 3
            if query_length < 2 or query_length > 3:
                raise ValueError

            # check last input is numeric
            elif query_length == 3:
                _ = float(query_split[2])

        # main catch if inputs can't be parsed as expected
        except ValueError:
            return '', '', 10.

        # unpack the query
        if query_length == 3:
            a, b, c = query_split
            c = float(c)
        else:
            a, b = query_split
            c = 10.
        return a, b, c

    @staticmethod
    def ra_dec_unit_parse(ra: str, dec: str) -> Tuple[Union[str, float], Union[str, float], str]:
        """
        Parses ra and dec values into either string (hms) or float (deg)

        Parameters
        ----------
        ra: str
            RA from string
        dec: str
            Dec from string

        Returns
        -------
        ra
            RA either string or float
        dec
            Dec either string or float
        unit
            Unit as a string
        """
        # try to make ra and dec numeric
        try:
            ra = float(ra)
            dec = float(dec)

        # in the case of hms, dms, set unit as so
        except ValueError:
            unit = 'hourangle,deg'

        # degree coordinates are fine though
        else:
            unit = 'deg'

        return ra, dec, unit

    def validate_query(self, field):
        """
        Validates the query to be understandable ra, dec

        Parameters
        ----------
        field
            The data within the searchbar

        """
        # split search bar into 3 components (ra, dec, radius)
        db = SimpleDB(self.db_file)  # open database
        ra, dec, radius = self.multi_param_str_parse(field.data)

        if not ra:  # i.e. empty string, bad parse
            raise ValidationError('Input must be two or three inputs separated by " "')

        # convert given values into understandable values (astropy units)
        ra, dec, unit = self.ra_dec_unit_parse(ra, dec)
        try:
            c = SkyCoord(ra=ra, dec=dec, unit=unit)

        except ValueError:
            raise ValidationError('Cannot parse coordinates, check astropy SkyCoord documentation')

        # attempt to query the region around given coordinates
        try:
            _ = db.query_region(c, fmt='pandas', radius=radius)
        except Exception as e:
            raise ValidationError(f'Uncaught Error -- {e}')


class LooseSearchForm(CSRFOverride):
    """
    Searching by full text
    """
    search = StringField('Search by full text:', id='mainsearchfield')  # searchbar
    submit = SubmitField('Search', id='querybutton')  # clicker button to send request


class BadSQLError(Exception):
    """
    Anything not starting with select in sql query
    """


class SQLForm(FlaskForm):
    """
    Searchbox class
    """
    sqlfield = TextAreaField('Enter SQL query here:', id='rawsqlarea', render_kw={'rows': '4'})
    submit = SubmitField('Query', id='querybutton')

    def __init__(self, *args, **kwargs: str):
        super(SQLForm, self).__init__(*args, **kwargs)
        self.db_file: str = kwargs['db_file']
        return

    def validate_sqlfield(self, field):
        """
        Validating SQL queries before they can be submitted
        Parameters
        ----------
        field
            The data within the query form
        """
        forbidden = ('update', 'drop', 'truncate', 'grant', 'commit', 'create', 'replace', 'alter', 'insert', 'delete')
        db = SimpleDB(self.db_file)  # open database

        # check query field has data within
        if (query := field.data) is None or query.strip() == '':  # content in main searchbar
            raise ValidationError('Empty field')

        # check for anything which would be problematic for sql to deal with
        try:
            query_low: str = query.lower()

            # select and from required
            if not query_low.startswith('select') or 'from' not in query_low:
                raise BadSQLError('Queries must start with "select" and contain "from".')

            # checking joins are done correctly
            if 'join' in query_low and not any([sub_string in query_low for sub_string in ('using', 'on')]):
                raise BadSQLError('When performing a join, you must provide either "using" or "on".')

            # look for malicious commands (David)
            if any([sub_string in query_low for sub_string in forbidden]):
                raise BadSQLError('Forbidden keyword detected.')

            # only then, attempt query
            _: Optional[pd.DataFrame] = db.sql_query(query, fmt='pandas')

        # catch these expected errors
        except (ResourceClosedError, OperationalError, IndexError, SqliteWarning, BadSQLError, ProgrammingError) as e:
            raise ValidationError('Invalid SQL: ' + str(e))

        # any unexpected errors
        except Exception as e:
            raise ValidationError('Uncaught Error: ' + str(e))


class JSCallbacks:
    """
    Converts javascript callbacks into python triple quoted strings
    """
    # initialised some empty strings to be filled with js functions
    dropdown_x_js = ''
    dropdown_y_js = ''
    button_flip = ''
    normalisation_slider = ''
    reset_slider = ''
    reset_dropdown = ''

    def __init__(self):
        """
        Loads simple_callbacks and unpacks the js functions within, to the python variables into instance attributes
        """
        # open js functions script
        js_func_names = ('dropdown_x_js', 'dropdown_y_js', 'button_flip', 'normalisation_slider',
                         'reset_slider', 'reset_dropdown')
        with open('simple_app/simple_callbacks.js', 'r') as func_call:
            which_var = ''
            out_string = """"""

            # reading through file as plaintext
            for line in func_call:

                # check each function name
                for func_name in js_func_names:

                    # ensure correct instance attribute is being written to
                    if func_name in line:
                        which_var = func_name
                        out_string = """"""
                        break

                # all functions in js script need to end with '}', this defines our attribute end
                else:

                    # set the instance attribute which has been written through previous lines and reset
                    if line.startswith('}'):
                        setattr(self, which_var, out_string)
                        which_var = ''
                        out_string = """"""
                        continue

                    # if the line is not defining a function start or end, write the line to the instance attribute
                    out_string = '\n'.join([out_string, line.strip('\n')])


def get_all_sources(db_file: str):
    """
    Queries the full table to get all the sources

    Parameters
    ----------
    db_file: str
        The connection string to the database

    Returns
    -------
    all_results
        Just the main IDs
    full_results
        The full dataframe of all the sources
    """
    db = SimpleDB(db_file)

    # get the full Sources table and just the main id list
    full_results: pd.DataFrame = db.query(db.Sources).pandas()
    all_results: list = full_results['source'].tolist()
    return all_results, full_results


def get_version(db_file: str) -> str:
    """
    Get the version and affiliated data

    Parameters
    ----------
    db_file
        The string pointing to the database file

    Returns
    -------
    vstr
        The stringified version formatted
    """
    db = SimpleDB(db_file)

    # query Versions table and extract active version before pretty-printing
    v: pd.DataFrame = db.query(db.Versions).pandas()
    v_active: pd.Series = v.iloc[-2]  # -1 is "latest" or main, hence -2
    v_str = f'Version {v_active.version}, updated last: {pd.Timestamp(v_active.end_date).strftime("%d %b %Y")}'
    return v_str


def find_colours(photometry_df: pd.DataFrame, all_bands: np.ndarray, photometric_filters: pd.DataFrame):
    """
    Find all the colours using available photometry

    Parameters
    ----------
    photometry_df: pd.DataFrame
        The dataframe with all photometry in
    all_bands: np.ndarray
        All the photometric bands
    photometric_filters: pd.DataFrame
        The filters

    Returns
    -------
    photometry_df: pd.DataFrame
        The dataframe with all photometry and colours in
    """
    def band_validate(checking_band: str):
        """
        Validates the given band that it is in our PhotometryFilters

        Parameters
        ----------
        checking_band
            Band to validate

        Returns
        -------
        checking_band_true
            Validated band
        """
        # check the true band name (ignoring if duplicated)
        checking_band_true = checking_band
        if '(' in checking_band:
            checking_band_true = checking_band[:checking_band.find('(')]

        # fail if checking_band is not in our PhotometryFilters
        if checking_band_true not in photometric_filters.columns:
            raise KeyError(f'{checking_band_true} not yet a supported filter')
        return checking_band_true

    wanted_mags = {'GAIA3.G', 'GAIA3.Grp', '2MASS.J', '2MASS.H', '2MASS.Ks', 'WISE.W1', 'WISE.W2'}
    wanted_cols = {'GAIA3.G-GAIA3.Grp', 'GAIA3.G-2MASS.J', '2MASS.J-2MASS.Ks', '2MASS.H-2MASS.Ks', 'WISE.W1-WISE.W2'}
    all_bands = np.array(list(wanted_mags.intersection(all_bands)))

    # looking at each band given in turn
    d_cols: Dict[str, np.ndarray] = {}
    for band in all_bands:

        # validate band
        band_true = band_validate(band)

        # looking at all other bands
        for next_band in all_bands:  # over all bands

            # don't make a colour of same band
            if band == next_band:
                continue
            # only want certain colours defined above
            elif f'{band}-{next_band}' not in wanted_cols:
                continue

            # validate band
            next_band_true = band_validate(next_band)

            # make sure we are only constructing blue-red colours
            if photometric_filters.at['effective_wavelength', band_true] >= \
                    photometric_filters.at['effective_wavelength', next_band_true]:
                continue

            # construct colour
            try:
                d_cols[f'{band}-{next_band}'] = photometry_df[band] - photometry_df[next_band]

            # in the case of duplicate bands (multiple measurements for one object)
            except KeyError:
                d_cols[f'{band}-{next_band}'] = photometry_df[band_true] - photometry_df[next_band_true]

    # rearrange colours as extra columns in the input photometry dataframe
    photometry_df = pd.concat([photometry_df, pd.DataFrame.from_dict(d_cols)], axis=1)
    return photometry_df


def one_source_iter(one_photometry_df: pd.DataFrame):
    """
    Parses the photometry dataframe handling multiple references for same magnitude for one object

    Parameters
    ----------
    one_photometry_df: pd.DataFrame
        The dataframe with all the photometry in it

    Returns
    -------
    this_new_phot: pd.DataFrame
        DataFrame of transposed photometry
    """

    def replacer(val: int) -> str:
        """
        Swapping an integer value for a string denoting the value

        Parameters
        ----------
        val: int
            The input number

        Returns
        -------
        _: str
            The formatted string of the value
        """
        if not val:
            return ''
        return f'({val})'

    # create dataframe of bands against only magnitudes, with different bands as columns
    one_photometry_df.set_index('band', inplace=True)
    this_new_phot: pd.DataFrame = one_photometry_df.loc[:, ['magnitude']].T

    # replace any duplicate columns with numeric, e.g. WISE W1(1), WISE W1(2)
    s = pd.Series(this_new_phot.columns)
    scc = s.groupby(s).cumcount()
    this_new_phot.columns += scc.map(replacer)
    return this_new_phot


# noinspection PyTypeChecker
def parse_photometry(photometry_df: pd.DataFrame, all_bands: np.ndarray, multi_source: bool = False) -> pd.DataFrame:
    """
    Parses the photometry dataframe handling multiple references for same magnitude

    Parameters
    ----------
    photometry_df: pd.DataFrame
        The dataframe with all photometry in
    all_bands: np.ndarray
        All the photometric bands
    multi_source: bool
        Switch whether to iterate over initial dataframe with multiple sources

    Returns
    -------
    df_new_photometry: pd.DataFrame
        DataFrame of effectively transposed photometry
    """
    # handle if looking at multiple objects at once or not
    if not multi_source:
        df_new_photometry = one_source_iter(photometry_df)

    # for multiple objects
    else:
        # initialise a DataFrame grouped by each target
        df_group_photometry = photometry_df.groupby('source')

        sources_list: List[Dict[str, object]] = []  # initialize empty list of sources

        # process photometry of each source
        with mp.Pool(processes=mp.cpu_count() - 1 or 1) as pool:
            results = pool.map(one_source_iter, [df for _, df in df_group_photometry])

        for (target, _), data in tqdm(zip(df_group_photometry, results), total=len(df_group_photometry),
                                      desc='Photometry'):

            row_data = {'target': target}

            for band in all_bands:

                try:
                    row_data[band] = data.loc['magnitude', band]
                except KeyError:
                    row_data[band] = None

            sources_list.append(row_data)

        df_new_photometry = pd.DataFrame(sources_list)

    return df_new_photometry


def get_all_photometry(db_file: str, photometric_filters: pd.DataFrame):
    """
    Get all of the photometric data from the database to be used in later CMD as background

    Parameters
    ----------
    db_file: str
        The connection string to the database
    photometric_filters: pd.DataFrame
        The dataframe of the filters

    Returns
    -------
    all_photometry: pd.DataFrame
        All the photometry in a dataframe
    all_bands: np.ndarray
        The unique passbands to create dropdowns by
    """
    db = SimpleDB(db_file)

    # query all photometry and extract the unique bands
    all_photometry: pd.DataFrame = db.query(db.Photometry).pandas()
    all_bands: np.ndarray = all_photometry.band.unique()

    # process magnitudes and extract colours
    print('Processing photometry.')
    all_photometry: pd.DataFrame = parse_photometry(all_photometry, all_bands, True)
    all_photometry = find_colours(all_photometry, all_bands, photometric_filters)
    print('Done.')
    return all_photometry, all_bands


def get_all_parallaxes(db_file: str):
    """
    Get the parallaxes from the database for every object

    Returns
    -------
    all_parallax: pd.DataFrame
        The dataframe of all the parallaxes
    """
    db = SimpleDB(db_file)

    # query the database for the parallaxes and only take necessary columns
    all_parallaxes: pd.DataFrame = db.query(db.Parallaxes).pandas()
    all_parallaxes = all_parallaxes[['source', 'parallax', 'adopted']]
    return all_parallaxes


def get_all_spectral_types(db_file: str):
    """
    Get the parallaxes from the database for every object

    Returns
    -------
    all_spts: pd.DataFrame
        The dataframe of all the spectral type numbers
    """
    db = SimpleDB(db_file)

    # query the database for the spectral types and only take necessary columns
    all_spts: pd.DataFrame = db.query(db.SpectralTypes).pandas()
    all_spts = all_spts[['source', 'spectral_type_code', 'adopted']]
    all_spts.rename(columns={'spectral_type_code': 'sptnum'}, inplace=True)
    return all_spts


def absolute_magnitudes(df: pd.DataFrame, all_bands: np.ndarray) -> pd.DataFrame:
    """
    Calculate all the absolute magnitudes in a given dataframe

    Parameters
    ----------
    df: pd.DataFrame
        The input dataframe
    all_bands: np.ndarray
        The photometric bands

    Returns
    -------
    df: pd.DataFrame
        The output dataframe with absolute mags calculated
    """

    def pogson_law(m: Union[float, pd.Series]) -> Union[float, np.ndarray]:
        """
        Distance modulus equation. Calculates the absolute magnitude only for sources with a positive parallax,
        otherwise returns a NaN

        Parameters
        ----------
        m
            The apparent magnitude
        Returns
        -------
        _
            Absolute magnitude
        """
        mask = df.parallax > 0
        _abs_mag = np.full_like(m, fill_value=np.nan)
        _abs_mag[mask] = m[mask] + 5 * np.log10(df.parallax[mask]) - 10
        return _abs_mag

    wanted_mags = {'GAIA3.G', '2MASS.J', 'WISE.W1'}
    all_bands = np.array(list(wanted_mags.intersection(all_bands)))

    # create absolute magnitude for each apparent magnitude
    d_magnitudes: Dict[str, np.ndarray] = {}
    for band in all_bands:

        abs_mag = "M_" + band
        try:
            d_magnitudes[abs_mag] = pogson_law(df[band])

        # in the case of duplicated band names
        except KeyError:
            d_magnitudes[abs_mag] = pogson_law(df[band[:band.find('(')]])

    # convert to dataframe version and append to original dataframe
    df_absolute_magnitudes = pd.DataFrame.from_dict(d_magnitudes)
    if 'magnitude' in df.index:
        df_absolute_magnitudes.rename(index={0: 'magnitude'}, inplace=True)
    df = pd.concat([df, df_absolute_magnitudes], axis=1)
    return df


def results_concat(all_results: pd.DataFrame, all_photometry: pd.DataFrame,
                   all_plx: pd.DataFrame, all_spts: pd.DataFrame, all_bands: np.ndarray) -> pd.DataFrame:
    """
    Gets parallax, photometry and projected positions into one dataframe
    Parameters
    ----------
    all_results
        Basic data for all the objects
    all_photometry
        All the photometry
    all_plx
        All the parallaxes
    all_spts
        All spectral types
    all_bands
        All the photometric bands

    Returns
    -------
    all_results_full: pd.DataFrame
        Concatenated data for all the objects
    """
    # project ra and dec onto Mollweide representation
    ra_projected, dec_projected = coordinate_project(all_results)
    all_results['ra_projected'] = ra_projected
    all_results['dec_projected'] = dec_projected

    # combine the photometry, parallaxes, spectral types and source table into one dataframe
    all_results_cut: pd.DataFrame = all_results[['source', 'ra_projected', 'dec_projected']]
    all_results_full: pd.DataFrame = pd.merge(all_results_cut, all_photometry,
                                              left_on='source', right_on='target', how='left')
    all_results_full = pd.merge(all_results_full, all_plx, on='source')
    all_results_full = pd.merge(all_results_full, all_spts, on='source')
    all_results_full = absolute_magnitudes(all_results_full, all_bands)
    all_results_full.drop_duplicates('source', inplace=True)
    return all_results_full


def coordinate_project(all_results_full: pd.DataFrame):
    """
    Projects RA and Dec coordinates onto Mollweide grid

    Returns
    -------
    ra_projected: np.ndarray
        The projected RA coordinates
    dec_projected: np.ndarray
        The projected DEC coordinates
    """

    def f_newton_solve(theta: float, phi: float, acc: float = 1e-4) -> float:
        """
        Solves the numerical transformation to project coordinate, see
        https://mathworld.wolfram.com/MollweideProjection.html

        Parameters
        ----------
        theta: float
            theta in radians
        phi: float
            phi in raidans
        acc: float
            Accuracy of calculation

        Returns
        -------
        theta: float
            theta in radians
        """
        # projection equation
        theta_projected = theta - (2 * theta + np.sin(2 * theta) - np.pi * np.sin(phi)) / (2 + 2 * np.cos(2 * theta))

        # handle special case of 90 degrees
        if np.isnan(theta_projected):  # at pi/2
            return phi

        # check the accuracy of the projection
        elif np.abs(theta_projected - theta) / np.abs(theta) < acc:  # less than desired accuracy
            return theta_projected

        # otherwise, recurse the function until the accuracy diminishes
        else:
            return f_newton_solve(theta_projected, phi)

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
        # determine polar coordinates from radians
        r = np.pi / 2 / np.sqrt(2)
        theta = f_newton_solve(dec, dec)

        # convert to x and y then back to degrees
        x = r * (2 * np.sqrt(2)) / np.pi * ra * np.cos(theta)
        y = r * np.sqrt(2) * np.sin(theta)
        x, y = np.rad2deg([x, y])
        return x, y

    # gather all coordinates
    ra_values: np.ndarray = all_results_full.ra.values
    dec_values: np.ndarray = all_results_full.dec.values
    all_coords = SkyCoord(ra_values, dec_values, unit='deg', frame='icrs')

    # convert to galactic and shift longitudes
    ra_values = all_coords.galactic.l.value
    dec_values = all_coords.galactic.b.value
    ra_values -= 180
    ra_values = np.array([np.abs(180 - raval) if raval >= 0 else -np.abs(raval + 180) for raval in ra_values])

    # project to Mollweide
    ra_values, dec_values = np.deg2rad([ra_values, dec_values])
    ra_projected, dec_projected = project_mollweide(ra_values, dec_values)
    return ra_projected, dec_projected


def one_df_query(results: pd.DataFrame, table_id: Optional[str] = None, limit_max_rows: bool = False) -> Optional[str]:
    """
    Handling the output from a query that returns only one dataframe

    Parameters
    ----------
    results
        The dataframe of results for the query
    table_id
        The table id to be passed to html
    limit_max_rows
        Limit max rows switch

    Returns
    -------
    stringed_results
        Results converted into markdown including links where there is a source
    """
    # html id keyword
    if table_id is None:
        table_id = 'searchtable'

    # for any tables containing data
    if len(results):

        # make the source always hyperlink to the solo_result page
        if 'source' in results.columns:
            source_links = []

            for source in results.source.values:
                if not isinstance(source, str):
                    source = source[0]
                url_link = quote(source)
                source_link = f'<a href="/load_solo/{url_link}" target="_blank">{source}</a>'
                source_links.append(source_link)

            results['source'] = source_links

        # for very large tables (e.g. just searching for '2MASS'), limit max rows
        if limit_max_rows:
            stringed_results = markdown(results.to_html(index=False, escape=False, table_id=table_id, max_rows=50,
                                                        classes='table table-dark table-bordered table-striped'))
        # otherwise, display all data
        else:
            stringed_results = markdown(results.to_html(index=False, escape=False, table_id=table_id,
                                                        classes='table table-dark table-bordered table-striped'))
    # if there is no data in the table
    else:
        stringed_results = None
    return stringed_results


def multi_df_query(results: Dict[str, pd.DataFrame], limit_max_rows: bool = False) -> Dict[str, Optional[str]]:
    """
    Handling the output from a query which returns multiple dataframes

    Parameters
    ----------
    results
        The dictionary of dataframes
    limit_max_rows
        Limit max rows switch

    Returns
    -------
    d_results
        The dictionary of handled dataframes
    """
    d_results = {}

    if len(results):

        # make sources table go first if present
        if 'Sources' in results.keys():
            d_results['Sources'] = one_df_query(results.pop('Sources'), 'sourcestable', limit_max_rows)

        # wrapping the one_df_query method for each table
        for table_name, df in results.items():
            if table_name == 'Spectra':
                df = Inventory.spectra_handle(df, False, True)
            stringed_df = one_df_query(df, table_name.lower() + 'table', limit_max_rows)
            d_results[table_name] = stringed_df
    return d_results


def get_filters(db_file: str) -> pd.DataFrame:
    """
    Query the photometry filters table

    Parameters
    ----------
    db_file: str
        The connection string to the database

    Returns
    -------
    phot_filters: pd.DataFrame
        All of the filters, access as: phot_filters.loc['effective_wavelength', <bandname>]
    """
    db = SimpleDB(db_file)

    # query the database for all of the PhotometryFilters
    phot_filters: pd.DataFrame = db.query(db.PhotometryFilters).pandas().set_index('band').T
    return phot_filters


def control_response(response: Response, key: str = '', app_type: str = 'csv') -> Response:
    """
    Edits the headers of a flask response

    Parameters
    ----------
    response
        The response as streamed out
    key
        The key used in the query, to differentiate returned results
    app_type
        The type of application being returned

    Returns
    -------
    response
        The response with edited headers
    """
    # adding to filename to differentiate results
    if len(key):  # if something provided to append
        key = '_' + key

    # checking application/content/mime types to be returned
    if app_type == 'csv':
        content_type = 'text/csv'
    elif app_type == 'zip':
        content_type = 'application/zip'
    else:
        content_type = 'text/plain'

    # handling headers and filename
    suffix = '.' + app_type
    response.headers['Content-Type'] = f"{content_type}; charset=utf-8"
    now_time = strftime("%Y-%m-%d--%H-%M-%S", localtime())
    filename = 'simplequery-' + now_time + key + suffix
    response.headers['Content-Disposition'] = f"attachment; filename={filename}"
    return response


def write_file(results: pd.DataFrame) -> Generator:
    """
    Creates a csv file ready for download on a line by line basis

    Parameters
    ----------
    results: pd.DataFrame
        The dataframe to be written

    Yields
    -------
    _
        Each line of the outputted csv
    """
    yield f"{','.join(results.columns)}\n"

    for i, _row in results.iterrows():
        row_pack = [str(val) for val in _row.tolist()]
        yield f"{','.join(row_pack)}\n"


# noinspection PyTypeChecker
def write_multi_files(d_results: Dict[str, pd.DataFrame]) -> BytesIO:
    """
    Creates a zip file containing multiple csvs ready for download

    Parameters
    ----------
    d_results
        The collection of dataframes

    Returns
    -------
    zip_mem
        The zipped file in memory
    """
    d_csv: Dict[str, StringIO] = {}

    # creating each csv file corresponding to each datafarme
    for key, df in d_results.items():
        csv_data = StringIO()
        df.to_csv(csv_data, index=False)
        csv_data.seek(0)
        d_csv[key] = csv_data

    # creating a zipped file containing all csv files
    zip_mem = BytesIO()
    with ZipFile(zip_mem, 'w') as zipper:
        for key, csv_data in d_csv.items():
            zipper.writestr(f"{key}.csv", csv_data.getvalue())

    zip_mem.seek(0)
    return zip_mem


# noinspection PyTypeChecker
def write_spec_files(spec_files: List[str]) -> BytesIO:
    """
    Creates a zip file containing multiple spectra ready for download

    Parameters
    ----------
    spec_files
        The list of fits files for this object

    Returns
    -------
    zip_mem
        The zipped file in memory
    """
    d_spec: Dict[str, Union[BytesIO, StringIO]] = {}

    # processing each spectrum
    for i, spec_file in enumerate(spec_files):

        # for fits files
        if spec_file.endswith('fits'):
            file_mem = BytesIO()

            # opening with astropy
            try:
                with fits.open(spec_file) as hdu_list:
                    hdu_list.writeto(file_mem, output_verify='ignore')

            # if there is an issue with the spectra
            except (OSError, fits.verify.VerifyError):
                continue

        # for text files
        else:
            response = requests.get(spec_file)
            if response.status_code != 200:  # i.e. could not download
                continue
            file_mem = StringIO(response.text)

        file_mem.seek(0)
        d_spec[f"spectra_{i}_" + os.path.basename(spec_file)] = file_mem

    # if at least one spectra downloaded correctly
    if len(d_spec):

        # creating a zipped file containing all spectra files
        zip_mem = BytesIO()
        with ZipFile(zip_mem, 'w') as zipper:
            for key, spec_data in d_spec.items():
                zipper.writestr(f"{key}", spec_data.getvalue())

        zip_mem.seek(0)
        return zip_mem

    # if no spectra files extracted
    return None


def main_utils():
    """
    Control module called when grabbing all instances from utils scripts.
    """
    _args = sys_args()
    _db_file = f'sqlite:///{_args.file}'
    _phot_filters = get_filters(_db_file)
    _all_results, _all_results_full = get_all_sources(_db_file)
    _versionstr = get_version(_db_file)
    _all_photometry, _all_bands = get_all_photometry(_db_file, _phot_filters)
    _all_plx = get_all_parallaxes(_db_file)
    _all_spts = get_all_spectral_types(_db_file)
    return _args, _db_file, _phot_filters, _all_results, _all_results_full, _versionstr, \
        _all_photometry, _all_bands, _all_plx, _all_spts


REFERENCE_TABLES = [
    "Publications",
    "Telescopes",
    "Instruments",
    "PhotometryFilters",
    "Versions",
    "Parameters",
    "Regimes",
    "CompanionList"
]

if __name__ == '__main__':
    ARGS, DB_FILE, PHOTOMETRIC_FILTERS, ALL_RESULTS, ALL_RESULTS_FULL, VERSION_STR, \
        ALL_PHOTO, ALL_BANDS, ALL_PLX, ALL_SPTS = main_utils()
