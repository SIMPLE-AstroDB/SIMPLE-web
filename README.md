# SIMPLE Website 

This is the repo for the codes corresponding to generating the SIMPLE website, designed to be as
interactive as possible.  
### Installation
To run the application locally, clone the application repo and move into it with:

```bash
git clone --recurse-submodules https://github.com/SIMPLE-AstroDB/SIMPLE-web.git
cd SIMPLE-web
```

Then, if you are running conda (recommended):
```bash
conda env create -f environment.yml
```
or:
```bash
conda create --name simple --file requirements.txt
```

#### If you have already cloned the repo and need to add the submodule in
First time:
```bash
git submodule update --init
```

### Running
Then run the application with   
```python 
python simple_app/app_simple.py
```
For more options (help) run
```python 
python simple_app/app_simple.py -h
```
Launch a browser and enter the URL [http://127.0.0.1:8000](http://127.0.0.1:8000).  
If you have changed either the host or port with system arguments, use those instead.  

### Updating
We also recommend keeping up to date with the repo changes, and most importantly, 
the [astrodbkit2](https://github.com/dr-rodriguez/AstrodbKit2) package:
```bash
git pull --recurse-submodules upstream main
pip install git+https://github.com/dr-rodriguez/AstrodbKit2
```
Alternatively, one can update the submodule by:
```bash
cd SIMPLE-db
git pull upstream main
```

### Contributing
Alternatively, all contributors are very welcome to fork (button top right of page) the repo
to make edits to the website code, in which case:
```bash
git clone --recurse-submodules https://github.com/<your-github-username>/SIMPLE-web.git
cd SIMPLE-web
git remote add origin https://github.com/<your-github-username>/SIMPLE-web.git
git remote add upstream https://github.com/SIMPLE-AstroDB/SIMPLE-web.git
git remote -v
```
The last line is to verify the remote is set up correctly, you can then push to *your* (origin) repo 
(we advise using branches to easily isolate different changes you may be making) and pull from the upstream repo. 
You can then create online a pull request to merge your repo into the upsteam repo after review.
```bash
git pull --recurse-submodules upsteam main
git add <files you have changed>
git commit -m "A good, informative commit message"
git push origin main  # or instead of main, a different branch
```

For feedback, questions, or if you've found an error, 
please [create an Issue here](https://github.com/SIMPLE-AstroDB/SIMPLE-web/issues).

The database repo can be found [here](https://github.com/SIMPLE-AstroDB/SIMPLE-db).  

This application builds and expands on the original code used by:
 - [ONCdbWeb](https://github.com/ONCdb/ONCdbWeb) built by the ONCdb Team at STScI
 - [AstrodbWeb](https://github.com/dr-rodriguez/AstrodbWeb) built by BDNYC at AMNH
