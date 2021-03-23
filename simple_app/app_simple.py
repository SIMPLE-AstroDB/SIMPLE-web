from astrodbkit2.astrodb import Database
from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from markdown2 import markdown
import os
import pandas as pd
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, StopValidation

# initialise
app_simple = Flask(__name__)
app_simple.vars = dict()
app_simple.vars['query'] = ''
app_simple.vars['search'] = ''
app_simple.vars['specid'] = ''
app_simple.vars['source_id'] = ''
app_simple.config['SECRET_KEY'] = os.urandom(32)

db_file = 'sqlite:///../SIMPLE.db'
pd.set_option('max_colwidth', None)  # deprecation warning


class SimpleDB(Database):  # this keeps pycharm happy about unresolved references
    Sources = None


class CheckResultsLength(object):
    def __call__(self, form, field):
        db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})
        results: pd.DataFrame = db.search_object(field.data, fmt='pandas')
        if not len(results):
            field.errors[:] = []
            raise StopValidation(field.gettext('No results'))


class SearchForm(FlaskForm):
    search = StringField('', [DataRequired(), CheckResultsLength()])
    submit = SubmitField('Query')


# website pathing
@app_simple.route('/')
@app_simple.route('/index', methods=['GET', 'POST'])
def index_page():
    # open database object, use connection arguments to have different threads to calm sqlite
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})
    source_count = db.query(db.Sources).count()
    return render_template('index_simple.html', source_count=source_count)


@app_simple.route('/search', methods=['GET', 'POST'])
def search():
    form = SearchForm()
    query = form.search.data
    if request.method == 'POST' and form.validate_on_submit():
        return redirect((url_for('search_results', query=query)))
    return render_template('search.html', form=form)


@app_simple.route('/search_results/<query>')
def search_results(query: str):
    db = SimpleDB(db_file, connection_arguments={'check_same_thread': False})
    results: pd.DataFrame = db.search_object(query, fmt='pandas')
    query = query.upper()
    results: str = markdown(results.to_html())
    return render_template('search_results.html', query=query, results=results)


@app_simple.route('/feedback')
def feedback_page():
    return render_template('feedback.html')


@app_simple.route('/schema')
def schema_page():
    return render_template('schema.html')


if __name__ == '__main__':
    app_simple.run(host='127.0.0.1', port=8000, debug=True)
