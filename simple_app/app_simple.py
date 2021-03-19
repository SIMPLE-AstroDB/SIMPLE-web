from flask import Flask, render_template
import pandas as pd
from astrodbkit2.astrodb import Database

# initialise
app_simple = Flask(__name__)
app_simple.vars = dict()
app_simple.vars['query'] = ''
app_simple.vars['search'] = ''
app_simple.vars['specid'] = ''
app_simple.vars['source_id'] = ''

db_file = 'sqlite:///../SIMPLE.db'
db = Database(db_file)
pd.set_option('max_colwidth', None)  # deprecation warning


# website pathing
@app_simple.route('/')
@app_simple.route('/index')
@app_simple.route('/index', methods=['GET', 'POST'])
def simple_query():
    defquery = 'SELECT * FROM sources'
    if app_simple.vars['query'] == '':
        app_simple.vars['query'] = defquery

    source_count = db.query(db.Sources).count()
    test_query = db.search_object('twa 27', fmt='pandas').source.values

    return render_template('index_simple.html', source_count=source_count, test_query=test_query)


if __name__ == '__main__':
    app_simple.run(host='127.0.0.1', port=8000, debug=True)
