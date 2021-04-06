"""
This is the main script to be run from the directory root, it will start the Flask application running which one can
then connect to.
"""
# external packages
from astrodbkit2.astrodb import Database
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_wtf import FlaskForm
from markdown2 import markdown
import numpy as np
import pandas as pd
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, StopValidation
# internal packages
import argparse
import os

# initialise
app_simple = Flask(__name__)
app_simple.config['SECRET_KEY'] = os.urandom(32)  # need to generate csrf token as basic security for Flask


def sysargs():
    """
    These are the system arguments given after calling this python script

    Returns
    -------
    _args
        The different argument parameters, can be grabbed via their long names (e.g. _args.host)
    """
    _args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _args.add_argument('-i', '--host', default='127.0.0.1', help='Local IP Address to host server')
    _args.add_argument('-p', '--port', default=8000, help='Local port number to host server through', type=int)
    _args.add_argument('-d', '--debug', help='Run Flask in debug mode?', default=False, action='store_true')
    _args.add_argument('-f', '--file', help='Database file path relative to current directory', default='SIMPLE.db')
    _args = _args.parse_args()
    return _args


class SimpleDB(Database):  # this keeps pycharm happy about unresolved references
    Sources = None  # initialise class attribute


class CheckResultsLength(object):
    def __call__(self, form, field):
        db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
        results: pd.DataFrame = db.search_object(field.data, fmt='pandas')  # search by what is currently in searchbar
        if not len(results):  # if that search is empty
            field.errors[:] = []  # clear existing errors
            raise StopValidation(field.gettext('No results'))  # stop validating and return error


class SearchForm(FlaskForm):
    search = StringField('', [DataRequired(), CheckResultsLength()], id='autocomplete')  # searchbar
    submit = SubmitField('Query')  # clicker button to send request


def all_sources():
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    # FIXME: db.search_object(..fmt='pandas') not returning column names
    allresults = np.array(db.search_object(''))[:, 0].tolist()  # list all the sources by main name
    return allresults


# website pathing
@app_simple.route('/')
@app_simple.route('/index', methods=['GET', 'POST'])
def index_page():
    # open database object, use connection arguments to have different threads to calm sqlite
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})
    source_count = db.query(db.Sources).count()  # count the number of sources
    return render_template('index_simple.html', source_count=source_count)


@app_simple.route('/search', methods=['GET', 'POST'])
def search():
    form = SearchForm()  # searchbar
    query = form.search.data  # the content in searchbar
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    results: pd.DataFrame = db.search_object(query, fmt='pandas')  # get the results for that object
    if request.method == 'POST' and form.validate_on_submit():  # if everything okay with the search
        if len(results) > 1:  # if more than one result
            return redirect((url_for('search_results', query=query)))  # return table of all results
        return redirect((url_for('solo_result', query=query)))  # otherwise return page for that one object
    return render_template('search.html', form=form)  # if everything not okay, return existing page as is


@app_simple.route('/search_results/<query>')
def search_results(query: str):
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    results: pd.DataFrame = db.search_object(query, fmt='pandas')  # get all results for that object
    query = query.upper()  # convert contents of search bar to all upper case
    results: str = markdown(results.to_html())  # convert the results into markdown to display nice on page
    return render_template('search_results.html', query=query, results=results)


@app_simple.route('/solo_result/<query>')
def solo_result(query: str):
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})  # open database
    resultdict: dict = db.inventory(query)  # get everything about that object
    query = query.upper()  # convert query to all upper case
    return render_template('solo_result.html', query=query, resultdict=resultdict)


@app_simple.route('/autocomplete', methods=['GET'])
def autocomplete():
    return jsonify(alljsonlist=all_results)  # wraps all of the object names as a list, into a .json for server use


@app_simple.route('/feedback')
def feedback_page():
    return render_template('feedback.html')


@app_simple.route('/schema')
def schema_page():
    return render_template('schema.html')


if __name__ == '__main__':
    args = sysargs()  # get all system arguments
    db_file = f'sqlite:///{args.file}'  # the database file
    all_results = all_sources()  # find all the objects once
    app_simple.run(host=args.host, port=args.port, debug=args.debug)  # generate the application on server side
