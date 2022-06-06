#/usr/bin/python
import sys
sys.path.insert(0, "/var/www/webroot/ROOT/simple_app")
sys.path.insert(0, "/var/www/webroot/ROOT")
from simple_app import create_app
application = create_app()
