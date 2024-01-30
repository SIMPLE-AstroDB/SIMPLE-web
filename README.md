# SIMPLE Website 

This is the repo for the codes corresponding to generating the SIMPLE website, designed to be as
interactive as possible.  
### Installation
To run the application locally, clone the application repo and move into it with:

```bash
git clone https://github.com/SIMPLE-AstroDB/SIMPLE-web.git
cd SIMPLE-web
```

Then, if you are running conda (recommended):
```bash
conda env create -f environment.yml
```

### Refresh the database
Get a fresh copy of the database from the binary repo.
```bash
wget https://raw.githubusercontent.com/SIMPLE-AstroDB/SIMPLE-binary/main/SIMPLE.db
```

### Running
Then run the application with   
```bash 
python -m simple_app.app_simple
```
For more options (help) run
```bash 
python -m simple_app.app_simple -h
```
Launch a browser and enter the URL [http://127.0.0.1:8000](http://127.0.0.1:8000).  
If you have changed either the host or port with system arguments, use those instead.  

### Updating
We also recommend keeping up to date with the repo changes, and most importantly, 
the [astrodbkit2](https://github.com/dr-rodriguez/AstrodbKit2) package:
```bash
git pull
pip install -Ur requirements.txt
```
You can also get the latest copy of the SQLite database binary file again with:
```bash
wget https://raw.githubusercontent.com/SIMPLE-AstroDB/SIMPLE-binary/main/SIMPLE.db
```

## Apache Config
The major requirement for running this program on Apache
is [`mod_wsgi`](https://flask.palletsprojects.com/en/2.1.x/deploying/mod_wsgi/)
You will need to edit `/etc/httpd/conf/httpd.conf` by adding the line:
```
WSGIApplicationGroup %{GLOBAL}
```
Additionally, edit ` /etc/httpd/conf.d/wsgi.conf` with the lines:
```
WSGIScriptAlias / <your/path/to/simple>/simple_app.wsgi
Alias /static <your/path/to/simple>/simple_app/static
Alias /favicon.ico <your/path/to/simple>/simple_app/static/favicon.ico
```
Finally, in the `simple_root` symlink, swap
`
.
`
for
`
<your/path/to/simple>
`


## Further Details
For feedback, questions, or if you've found an error, 
please [create an Issue here](https://github.com/SIMPLE-AstroDB/SIMPLE-web/issues).

The database repo can be found [here](https://github.com/SIMPLE-AstroDB/SIMPLE-db).  

This application builds and expands on the original code used by:
 - [ONCdbWeb](https://github.com/ONCdb/ONCdbWeb) built by the ONCdb Team at STScI
 - [AstrodbWeb](https://github.com/dr-rodriguez/AstrodbWeb) built by BDNYC at AMNH
