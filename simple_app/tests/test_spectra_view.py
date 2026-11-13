"""
Testing the spectra reading and plotting
"""
import numpy as np
import pytest
from bokeh.themes import built_in_themes

from ..spectra_view import FILTERS, magnitudes, read_spectrum, spectra_panel, synflux

# formats specutils alone cannot read: the three spectra of TWA 26 and a SpeX SXD one of TWA 27
TWA26 = {'irs_text': 'https://s3.amazonaws.com/bdnyc/1139-3159+%285-35um%29.txt',
         'spex_text': 'https://s3.amazonaws.com/bdnyc/SpeX/Prism/2MASS_J11395113-3159214_spex_prz.txt',
         'iraf_multispec': 'https://s3.amazonaws.com/bdnyc/U20419.fits',
         'spex_rows': 'https://s3.amazonaws.com/bdnyc/SpeX/SXD/2MASS_J12073346-3932539_spex_sxd.fits'}


@pytest.mark.parametrize('kind, wave_range', [('irs_text', (5.217, 37.864)), ('spex_text', (0.65, 2.56)),
                                              ('iraf_multispec', (0.56, 0.99)), ('spex_rows', (0.80, 2.42))])
def test_read_spectrum(kind, wave_range):
    wave, flux, err = read_spectrum(TWA26[kind])
    assert wave.min() == pytest.approx(wave_range[0], abs=0.01) and wave.max() == pytest.approx(wave_range[1], abs=0.01)
    assert np.all(np.diff(wave) >= 0) and np.isfinite(flux).all()
    assert np.isfinite(err).any()


def test_read_spectrum_unreadable():
    with pytest.raises(ValueError, match='unreadable'):
        read_spectrum('https://s3.amazonaws.com/bdnyc/notaspectrum.txt')


def test_synflux():
    wave = np.linspace(0.5, 3, 5000)
    sf, cov = synflux(wave, np.full(wave.shape, 2e-15), FILTERS['J'])
    assert sf == pytest.approx(2e-15) and cov == pytest.approx(1)
    sf, cov = synflux(wave[wave < 1.24], np.full((wave < 1.24).sum(), 2e-15), FILTERS['J'])
    assert 0.2 < cov < 0.8  # half the filter


def test_magnitudes():
    rows = [{'band': '2MASS.J', 'magnitude': 12.}, {'band': 'WISE.W1', 'magnitude': 11.},
            {'band': 'WISE.W1', 'magnitude': 11.2}, {'band': 'WISE.W1', 'magnitude': 11.4}, {'band': 'X', 'magnitude': 1.}]
    assert magnitudes(rows) == {'J': 12., 'W1': pytest.approx(11.2)}
    assert magnitudes([]) == {}


def test_spectra_panel():
    spectra = [{'access_url': url, 'telescope': 'T', 'instrument': kind, 'mode': 'Missing', 'reference': 'Ref',
                'observation_date': '2007-03-16'} for kind, url in TWA26.items()]
    spectra.append({**spectra[0], 'access_url': 'https://s3.amazonaws.com/bdnyc/notaspectrum.txt', 'instrument': 'bad'})
    script, div, failures = spectra_panel('TWA 26', spectra, {'J': 12.686}, built_in_themes['night_sky'])
    assert script and div
    assert len(failures) == 1 and failures[0].startswith('T/bad (Ref): unreadable')
    assert 'Photoscale' in script  # the SpeX spectra cover J
    assert spectra_panel('TWA 26', spectra[-1:], {}, built_in_themes['night_sky'])[:2] == (None, None)
