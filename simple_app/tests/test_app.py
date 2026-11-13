"""
Smoke tests of the website routes through the flask test client
"""
import pytest

TWA_SQL = 'select * from Sources where source like "%twa%"'


@pytest.mark.parametrize('route', ['/', '/about', '/search?search=twa', '/full_text_search?search=cruz',
                                   '/coordinate_query', '/raw_query', '/multi_plot',
                                   '/solo_result/2MASS J00192626+4614078'])
def test_pages(client, route):
    assert client.get(route).status_code == 200


def test_unknown_source(client):
    assert client.get('/solo_result/thisisabadquery').status_code == 404


def test_raw_query(client):
    # no csrf token posted: the form must still submit (#217)
    page = client.post('/raw_query', data={'sqlfield': TWA_SQL}).get_data(as_text=True)
    assert 'id="searchtable"' in page
    csv = client.get('/write_sql').get_data(as_text=True)
    assert csv.startswith('source,') and 'TWA' in csv


@pytest.mark.parametrize('sql, error', [('select * from NotaTable', 'Invalid SQL'),
                                        ('delete from Sources', 'Queries must start with'),
                                        ('select * from Sources; drop table Sources', 'Forbidden keyword')])
def test_raw_query_errors(client, sql, error):
    page = client.post('/raw_query', data={'sqlfield': sql}).get_data(as_text=True)
    assert error in page
    assert 'id="searchtable"' not in page


def test_coordinate_query(client):
    page = client.post('/coordinate_query', data={'query': '174.96308 -31.989305 10'}).get_data(as_text=True)
    assert 'id="searchtable"' in page
    page = client.post('/coordinate_query', data={'query': 'not coordinates'}).get_data(as_text=True)
    assert 'Cannot parse coordinates' in page


def test_propagate(client):
    res = client.get('/propagate/TWA 27?epoch=2030.5')
    assert res.status_code == 200 and res.json['used_parallax']
    assert client.get('/propagate/TWA 27').status_code == 400
    assert client.get('/propagate/thisisabadquery?epoch=2030').json['error'] == 'Source not found'


def test_query_builder_tables(client):
    page = client.get('/raw_query').get_data(as_text=True)
    assert 'id="querybuilder"' in page and '"Sources": ["source", "ra", "dec"' in page
