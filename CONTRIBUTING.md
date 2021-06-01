### Contributing
All contributors are very welcome to fork (button top right of page) the repo
to make edits to the website code, in which case:
```bash
git clone https://github.com/<your-github-username>/SIMPLE-web.git
cd SIMPLE-web
git remote add origin https://github.com/<your-github-username>/SIMPLE-web.git
git remote add upstream https://github.com/SIMPLE-AstroDB/SIMPLE-web.git
git remote -v
```
The last line is to verify the remote is set up correctly, you can then push to *your* (origin) repo 
(we advise using branches to easily isolate different changes you may be making) and pull from the upstream repo. 
You can then create online a pull request to merge your repo into the upsteam repo after review.
```bash
git pull upstream main
git add <files you have changed>
git commit -m "A good, informative commit message"
git push origin main  # or instead of main, a different branch
```