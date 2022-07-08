#/usr/bin/python
import sys
sys.path.insert(0, "simple_root/simple_app")
sys.path.insert(0, "simple_root")
from simple_app import create_app
application = create_app()
