"""
The static functions for various calculations and required parameters
"""
# external packages
from astrodbkit2.astrodb import Database, REFERENCE_TABLES  # used for pulling out database and querying
from astropy.coordinates import SkyCoord
from flask_wtf import FlaskForm  # web forms
from markdown2 import markdown  # using markdown formatting
import numpy as np  # numerical python
import pandas as pd  # running dataframes
from wtforms import StringField, SubmitField  # web forms
# internal packages
import argparse  # system arguments
from typing import Union, List  # type hinting


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

    def __init__(self, resultdict: dict, args):
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
        urlinks = []
        if rtnmk and key == 'Spectra':
            for src in df.spectrum.values:  # over every source in table
                srclnk = f'<a href="{src}" target="_blank">Link</a>'  # construct hyperlink
                urlinks.append(srclnk)  # add that to list
            df.drop(columns=[col for col in df.columns if any([substr in col for substr in ('wave', 'flux')])],
                    inplace=True)
            df = df.loc[:, 'telescope':].copy()
            df['download'] = urlinks
        if rtnmk:  # return markdown boolean
            df.rename(columns={s: s.replace('_', ' ') for s in df.columns}, inplace=True)  # renaming columns
            return markdown(df.to_html(index=False, escape=False,
                                       classes='table table-dark table-bordered table-striped'))  # html then markdown
        return df  # otherwise return dataframe as is


class SearchForm(FlaskForm):
    """
    Searchbar class
    """
    search = StringField('Search for an object:', id='autocomplete')  # searchbar
    submit = SubmitField('Query', id='querybutton')  # clicker button to send request


def all_sources(db_file):
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
                photodf[f'{band}-{nextband}'] = photodf[band] - photodf[nextband]  # colour
            except KeyError:
                continue
    return photodf


def parse_photometry(photodf: pd.DataFrame, allbands: np.ndarray, multisource: bool = False) -> dict:
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
    return newphoto


def all_photometry(db_file: str):
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
    # allbands = np.array([band.replace('.', '_') for band in allbands])
    allphoto = pd.DataFrame(outphoto)  # use rearranged dataframe
    allphoto = find_colours(allphoto, allbands)  # get the colours
    return allphoto, allbands


def all_parallaxes(db_file: str):
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


def absmags(df: pd.DataFrame, all_bands: np.ndarray) -> pd.DataFrame:
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


def results_concat(all_results_full: pd.DataFrame, all_photo: pd.DataFrame,
                   all_plx: pd.DataFrame, all_bands: np.ndarray) -> pd.DataFrame:
    """
    Gets parallax, photometry and projected positions into one dataframe
    Parameters
    ----------
    all_results_full
        Basic data for all the objects
    all_photo
        All the photometry
    all_plx
        All the parallaxes
    all_bands
        All the photometric bands

    Returns
    -------
    all_results_mostfull: pd.DataFrame
        Concatenated data for all the objects
    """
    raproj, decproj = coordinate_project(all_results_full)  # project coordinates to galactic
    all_results_full['raproj'] = raproj  # ra
    all_results_full['decproj'] = decproj  # dec
    all_results_full_cut: pd.DataFrame = all_results_full[['source', 'raproj', 'decproj']]  # cut dataframe
    all_results_mostfull: pd.DataFrame = pd.merge(all_results_full_cut, all_photo,
                                                  left_on='source', right_on='target', how='left')
    all_results_mostfull = pd.merge(all_results_mostfull, all_plx, on='source', how='left')
    all_results_mostfull = absmags(all_results_mostfull, all_bands)  # find the absolute mags
    return all_results_mostfull


def coordinate_project(all_results_full: pd.DataFrame):
    """
    Projects RA and Dec coordinates onto Mollweide grid

    Returns
    -------
    raproj: np.ndarray
        The projected RA coordinates
    decproj: np.ndarray
        The projected DEC coordinates
    """
    def fnewton_solve(thetan: float, phi: float, acc: float = 1e-4) -> float:
        """
        Solves the numerical transformation to project coordinate

        Parameters
        ----------
        thetan: float
            theta in radians
        phi: float
            phi in raidans
        acc: float
            Accuracy of calculation

        Returns
        -------
        thetan: float
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


def mainutils():
    _args = sysargs()  # get all system arguments
    _db_file = f'sqlite:///{_args.file}'  # the database file
    _all_results, _all_results_full = all_sources(_db_file)  # find all the objects once
    _all_photo, _all_bands = all_photometry(_db_file)  # get all the photometry
    _all_plx = all_parallaxes(_db_file)  # get all the parallaxes
    return _args, _db_file, _all_results, _all_results_full, _all_photo, _all_bands, _all_plx


if __name__ == '__main__':
    ARGS, DB_FILE, ALL_RESULTS, ALL_RESULTS_FULL, ALL_PHOTO, ALL_BANDS, ALL_PLX = mainutils()
