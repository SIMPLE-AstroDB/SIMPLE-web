cd ~/ROOT
wget -O SIMPLE.sqlite https://raw.githubusercontent.com/SIMPLE-AstroDB/SIMPLE-binary/main/SIMPLE.sqlite
wget -O database.toml https://raw.githubusercontent.com/SIMPLE-AstroDB/SIMPLE-db/refs/heads/main/database.toml
git pull
# reload the app (mod_wsgi daemon mode restarts on a touched WSGI script): it keeps the database in memory
touch simple_app.wsgi
