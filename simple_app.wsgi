#/usr/bin/python
import sys
sys.path.insert(0, "rootpath/simple_app")
sys.path.insert(0, "rootpath")
from simple_app import create_app
application = create_app()
