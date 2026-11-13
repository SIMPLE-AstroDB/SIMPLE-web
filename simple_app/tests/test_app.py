"""
Smoke tests of the website routes through the flask test client
"""
import re

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
    assert '/write_sql?query=' in page  # downloads carry the query, no server session
    csv = client.get('/write_sql', query_string={'query': TWA_SQL}).get_data(as_text=True)
    assert csv.startswith('source,') and 'TWA' in csv


@pytest.mark.parametrize('sql, error', [('select * from NotaTable', 'Invalid SQL'),
                                        ('delete from Sources', 'Queries must start with'),
                                        ('select * from Sources; drop table Sources', 'Forbidden keyword'),
                                        ('select * from Sources join Spectra', 'you must provide either')])
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
    assert client.get('/propagate/TWA 27?epoch=2030-07-01').json['date'] == '2030-07-01 00:00'
    assert client.get('/propagate/thisisabadquery?epoch=2030').json['error'] == 'Source not found'


def test_query_builder_tables(client):
    page = client.get('/raw_query').get_data(as_text=True)
    assert 'id="querybuilder"' in page and '"Sources": ["source", "ra", "dec"' in page


def test_raw_query_keywords_in_values(client):
    sql = "select * from Sources where comments like '%updated%' or comments like '%dropped%'"
    assert 'Forbidden keyword' not in client.post('/raw_query', data={'sqlfield': sql}).get_data(as_text=True)


def test_jdaviz_notebook(client):
    res = client.get('/jdaviz_notebook/TWA 27')
    assert res.status_code == 200 and 'TWA_27_jdaviz.ipynb' in res.headers['Content-Disposition']
    code = ''.join(res.json['cells'][1]['source'])
    compile(code, 'notebook', 'exec')  # valid python
    assert 'specviz.load(' in code and 'https://' in code
    assert client.get('/jdaviz_notebook/thisisabadquery').status_code == 404


def test_downloads_without_session(client):
    # a fresh client has no cookies: every download must work from its url alone
    page = client.get('/solo_result/TWA 27').get_data(as_text=True)
    assert '/write/Sources.csv?query=TWA+27' in page and '/write_spectra?query=TWA%2027' in page
    with client.application.test_client() as fresh:
        csv = fresh.get('/write/Sources.csv?query=TWA 27').get_data(as_text=True)
        assert csv.startswith('source,') and 'TWA 27' in csv
        res = fresh.get('/write_filt', query_string={'query': 'twa', 'ref_query': 'Kell15'})
        assert res.status_code == 200 and res.get_data(as_text=True).startswith('source,')


def test_solo_spectra(client):
    # TWA 26 has only spectra specutils alone cannot read; all three must now plot
    page = client.get('/solo_result/TWA 26').get_data(as_text=True)
    assert 'Could not plot' not in page and '["scale",{"type":"map","entries":[["J",' in page


def test_bokeh_javascript_matches_installed_version(client):
    # base.html loads static/js/bokeh-<installed version>.min.js: after upgrading bokeh, download the matching files
    import bokeh
    for name in ('bokeh', 'bokeh-widgets'):
        assert client.get(f'/static/js/{name}-{bokeh.__version__}.min.js').status_code == 200
    assert f'bokeh-{bokeh.__version__}.min.js' in client.get('/').get_data(as_text=True)


def _links(page: str, route: str) -> list:
    # href values of a route in a rendered page, html-unescaped as a browser would
    from html import unescape
    return [unescape(href) for href in re.findall(rf'href="({route}[^"]*)"', page)]


def test_download_links_url_sensitive_characters(client):
    # '+' in a source name decodes to a space, '&', '#' and '%' would cut or corrupt the query, if links were not
    # encoded: every download link is followed as the browser would
    source = '2MASS J00192626+4614078'
    page = client.get(f'/solo_result/{source}').get_data(as_text=True)
    hrefs = _links(page, '/write/Sources.csv') + _links(page, '/write_spectra')
    assert len(hrefs) == 2 and all('%2B' in href for href in hrefs), hrefs
    for href in hrefs:
        res = client.get(href)
        assert res.status_code == 200 and len(res.data), href
    sql = "select * from Sources where source like '%+46%' and (1 & 1) and comments is not '#x'"
    page = client.post('/raw_query', data={'sqlfield': sql}).get_data(as_text=True)
    href, = _links(page, '/write_sql')
    csv = client.get(href).get_data(as_text=True)
    assert source in csv
