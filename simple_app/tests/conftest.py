"""
Shared fixtures; tests need SIMPLE.sqlite and database.toml in the repository root
"""
import importlib
import sys
from shutil import copy

import pytest

from ..plots import main_plots
from ..utils import (SimpleDB, get_all_parallaxes, get_all_photometry, get_all_sources, get_all_spectral_types,
                     get_filters)


@pytest.fixture(scope='session')
def db_cs(tmp_path_factory) -> str:
    db_path = tmp_path_factory.mktemp('db') / 'SIMPLE.sqlite'
    copy('SIMPLE.sqlite', db_path)
    return f'sqlite:///{db_path}'


@pytest.fixture(scope='session')
def db(db_cs):
    return SimpleDB(db_cs)


@pytest.fixture(scope='session')
def filters(db_cs):
    return get_filters(db_cs)


@pytest.fixture(scope='session')
def sources(db_cs):
    return get_all_sources(db_cs)


@pytest.fixture(scope='session')
def photometry(db_cs, filters):
    return get_all_photometry(db_cs, filters)


@pytest.fixture(scope='session')
def parallaxes(db_cs):
    return get_all_parallaxes(db_cs)


@pytest.fixture(scope='session')
def spectral_types(db_cs):
    return get_all_spectral_types(db_cs)


@pytest.fixture(scope='session')
def plot_settings():
    return main_plots()


@pytest.fixture(scope='session')
def client(db_cs):
    # app_simple loads everything at import time from the command line arguments
    argv = sys.argv
    sys.argv = ['app_simple', '-f', db_cs.removeprefix('sqlite:///')]
    try:
        app_simple = importlib.import_module('simple_app.app_simple').app_simple
    finally:
        sys.argv = argv
    return app_simple.test_client()
