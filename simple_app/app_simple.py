from flask import Flask, render_template, request, redirect, make_response, url_for
#app_onc = Flask(__name__)
app_simple = Flask(__name__)

import astrodbkit2
from astrodbkit2 import astrodb
#import astrodbkit
#from astrodbkit import astrodb
import pandas as pd

#app_onc.vars = dict()
#app_onc.vars['query'] = ''
#app_onc.vars['search'] = ''
#app_onc.vars['specid'] = ''
#app_onc.vars['source_id'] = ''

app_simple.vars = dict()
app_simple.vars['query'] = ''
app_simple.vars['search'] = ''
app_simple.vars['specid'] = ''
app_simple.vars['source_id'] = ''


from astrodbkit2.astrodb import Database
#connection_string = 'sqlite:///bdnyc_database.db'   #Had been 'sqlite:///SIMPLE.db'
#db_dir = r'C:\Users\danie\SIMPLE-web'
#db = Database(connection_string)
#db.load_database(db_dir)

db_file = 'sqlite:///SIMPLE.db'
db = Database(db_file)
pd.set_option('max_colwidth', -1)

#db_file_OLD = 'bdnyc_database.db'
#db_OLD = astrodb.Database(db_file)
#pd.set_option('max_colwidth', -1)

# Redirect to the main page
#@app_onc.route('/')
#@app_onc.route('/index')

@app_simple.route('/')
@app_simple.route('/index')

# Page with a text box to take the SQL query
#@app_onc.route('/index', methods=['GET', 'POST'])

@app_simple.route('/index', methods=['GET', 'POST'])

def simple_query():
    defquery = 'SELECT * FROM sources'
    if app_simple.vars['query']=='':
        app_simple.vars['query'] = defquery

    # Get list of the catalogs
    #source_count, = db.list("SELECT Count(*) FROM sources").fetchone()
    source_count = db.query(db.Sources).count()
    #query_string = 'SELECT * FROM Photometry where Photometry.source like %J1%'
    #test_query = db.(query_string)
    test_query = db.search_object('twa 27', fmt='pandas').source.values
    #catalogs = db.query("SELECT * FROM publications", fmt='table')
    #catalogs = db.query("SELECT * FROM publications", "table")
    #cat_names = ''.join(['<li><a href="https://ui.adsabs.harvard.edu/?#abs/{}/abstract">{}</a></li>'.format(cat['bibcode'],cat['description'].replace('VizieR Online Data Catalog: ','')) for cat in catalogs])
    #cat_names = 'simple catalog names placeholder'

    #table_names = db.query("select * from sqlite_master where type='table' or type='view'")['name']

    #tables = '\n'.join(['<option value="{0}" {1}> {0}</option>'.format(t,'selected=selected' if t=='browse' else '') for t in table_names])

    #columns_html = []
    #columns_js = []
    #for tab in table_names:
     #   cols = list(db.query("pragma table_info('{}')".format(tab))['name'])

      #  col_html = ''.join(['<input type="checkbox" value="{0}" name="selections"> {0}<br>'.format(c) for c in cols])
       # columns_html.append('<div id="{}" class="columns" style="display:none">{}</div>'.format(tab,col_html))

        #col_js = ','.join(["{id:'"+c+"',label:'"+c+"',type:'string'}" for c in cols])
        #columns_js.append(col_js)

    #column_select = ''.join(columns_html)
    #column_script = ''.join(columns_js)

    return render_template('index_simple.html', source_count=source_count, test_query=test_query)
     #                      defsearch=app_simple.vars['search'], specid=app_simple.vars['specid'],
      #                     source_id=app_simple.vars['source_id'], version=astrodbkit2.__version__,
       #                    tables=tables, column_select=column_select, column_script=col_js)


## -- RUN
if __name__ == '__main__':
    # port = int(os.environ.get('PORT', 8000))
    #app_onc.run(host='127.0.0.1', port=8000, debug=True)
    app_simple.run(host='127.0.0.1', port=8000, debug=True)

