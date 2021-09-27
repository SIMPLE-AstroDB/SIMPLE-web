"""
Testing the utils functions
"""
# external packages
import pytest
# internal packages
import os
from shutil import copy
# local packages
from simple_app.utils import *


db_name = 'temp.db'
db_cs = 'sqlite:///temp.db'


@pytest.fixture
def db():
    if os.path.exists(db_name):
        os.remove(db_name)
    copy('SIMPLE.db', db_name)
    assert os.path.exists(db_name)
    # Connect to the new database and confirm it has the Sources table
    db = SimpleDB(db_cs)
    assert db
    assert 'source' in [c.name for c in db.Sources.columns]
    return db


def test_all_sources(db):
    assert db
    assert all_sources(db_cs)
    return


def test_remove_database(db):
    # Clean up temporary database
    db.session.close()
    db.engine.dispose()
    if os.path.exists(db_name):
        os.remove(db_name)
    return
