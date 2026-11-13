"""
Spectra plot of a source page: every spectrum of the source on one plot, scaled in the browser (see JS_SCALE) by
photoscaling to the source's SIMPLE magnitudes through the filter curves of the SVO Filter Profile Service
(filters.json), with normalisation as fallback. Feature labelling, telluric bands, uncertainty bands, smoothing and the
synthetic photometry follow the GUCDS spectra browser.
"""
import json
import os
import re
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, urlparse

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.utils.data import download_file
from bokeh.document import Document
from bokeh.embed import components
from bokeh.layouts import column, row
from bokeh.models import (Band, BoxAnnotation, ColumnDataSource, CustomJS, HoverTool, Label, RangeSlider, Select,
                          Slider, Span, Toggle)
from bokeh.palettes import Colorblind8
from bokeh.plotting import figure
from bokeh.themes import Theme
from specutils import Spectrum

from .plots import bokeh_formatter

COVERAGE = 0.9  # fraction of a filter (photoscaling) or of the normalisation window a spectrum must cover to use it
COLOURS = Colorblind8[:7]  # its 8th colour is black, invisible on the dark theme
DASHES = ('solid', 'dashed', 'dotted', 'dotdash', 'dashdot')  # each further set of seven spectra
MAXPOINTS = 8000  # longer spectra are binned to about this many points for display
FLAMBDA = u.erg / u.s / u.cm ** 2 / u.AA
# filters.json band: SIMPLE photometry band, in the order a spectrum is photoscaled by the first one it covers
# (filter curves and zero points from the SVO Filter Profile Service; Pan-STARRS in AB, the others in Vega)
BANDS = {'J': '2MASS.J', 'H': '2MASS.H', 'Ks': '2MASS.Ks', 'RP': 'GAIA3.Grp', 'G': 'GAIA3.G', 'z': 'PAN-STARRS/PS1.z',
         'y': 'PAN-STARRS/PS1.y', 'i': 'PAN-STARRS/PS1.i', 'W1': 'WISE.W1', 'I1': 'IRAC.I1', 'W2': 'WISE.W2',
         'I2': 'IRAC.I2', 'I3': 'IRAC.I3', 'I4': 'IRAC.I4', 'W3': 'WISE.W3', 'W4': 'WISE.W4'}
# spectral features (micron), after SPLAT (Burgasser 2017), grouped by the spectral
# class in which they are typical: molecular bands are shaded, atomic lines dashed
_KI = ('line', 'K I', (0.7665, 0.7699, 1.1690, 1.1773, 1.2432, 1.2522))
_NAI = ('line', 'Na I', (0.8183, 0.8195, 1.1381, 1.1404, 2.2062, 2.2090))
_CSI = ('line', 'Cs I', (0.8521, 0.8943))
_RBI = ('line', 'Rb I', (0.7800, 0.7948))
_H2O = ('band', 'H₂O', ((0.925, 0.95), (1.325, 1.55), (1.72, 2.14)))
_CH4 = ('band', 'CH₄', ((1.1, 1.24), (1.6, 1.76), (2.2, 2.35), (3.2, 3.5)))
FEATURE_SETS = {
    'M dwarfs': (('band', 'CaH', ((0.6346, 0.639), (0.675, 0.705))), ('band', 'TiO', ((0.705, 0.727), (0.76, 0.80),
                                                                                       (0.845, 0.86))),
                 ('band', 'VO', ((0.735, 0.758),)), _KI, _NAI, ('band', 'FeH', ((0.98, 1.03),)), _H2O,
                 ('band', 'CO', ((2.29, 2.39),))),
    'L dwarfs': (('band', 'TiO', ((0.705, 0.727),)), ('band', 'CrH', ((0.8611, 0.8681),)),
                 ('band', 'FeH', ((0.8692, 0.875), (0.98, 1.03), (1.19, 1.25), (1.57, 1.64))),
                 ('band', 'VO', ((1.04, 1.08),)), _KI, _NAI, _RBI, _CSI, _H2O, ('band', 'CO', ((2.29, 2.39), (4.5, 4.9))),
                 ('band', 'CH₄', ((3.2, 3.5),))),
    'T dwarfs': (_KI, _NAI, _RBI, _CSI, ('band', 'FeH', ((0.98, 1.03),)), _H2O, _CH4, ('band', 'CO', ((4.5, 4.9),)),
                 ('band', 'CO₂', ((4.2, 4.35),))),
    'Y dwarfs': (('line', 'K I', (1.2432, 1.2522)), _H2O, _CH4, ('band', 'CO', ((4.5, 4.9),)),
                 ('band', 'CO₂', ((4.2, 4.35),))),
}
TELLURIC = ((0.686, 0.695), (0.759, 0.770), (0.813, 0.835), (0.895, 0.980), (1.11, 1.16), (1.35, 1.45),
            (1.80, 1.95), (2.50, 2.90), (4.20, 4.40))  # micron, the main telluric absorption bands

# ---------------------------------------------------------------------------------------------- browser-side code
JS_SCALE = """
// Puts every spectrum on one plot, best first: 1. photoscaled to the first band it covers; 2. if it covers the
// normalisation window, normalised to the photoscaled spectra there (to unity when none are); 3. matched in its
// wavelength overlap to an already scaled spectrum, repeatedly; 4. otherwise matched at the nearest edge of a scaled
// spectrum, which is only approximate. If 1 and 2 scale nothing, the widest spectrum is normalised to its median.
function boxcar(y, n) {
    if (n <= 1) return Float64Array.from(y);
    const out = new Float64Array(y.length), h = Math.floor(n / 2);
    for (let i = 0; i < y.length; i++) {
        let s = 0, k = 0;
        for (let j = Math.max(0, i - h); j <= Math.min(y.length - 1, i + h); j++)
            if (Number.isFinite(y[j])) { s += y[j]; k++; }
        out[i] = k ? s / k : NaN;
    }
    return out;
}
function median(a) {
    const v = Array.from(a).filter(Number.isFinite).sort((p, q) => p - q);
    if (!v.length) return NaN;
    const m = Math.floor(v.length / 2);
    return v.length % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}
function window_median(w, f, lo, hi) {
    // median flux in [lo, hi], or of the five valid points nearest its centre where fewer than three fall in it
    const sel = [];
    for (let i = 0; i < w.length; i++) if (w[i] > lo && w[i] < hi && Number.isFinite(f[i])) sel.push(f[i]);
    if (sel.length >= 3) return median(sel);
    const c = 0.5 * (lo + hi), idx = [];
    for (let i = 0; i < w.length; i++) if (Number.isFinite(f[i])) idx.push(i);
    idx.sort((a, b) => Math.abs(w[a] - c) - Math.abs(w[b] - c));
    return median(idx.slice(0, 5).map(i => f[i]));
}
function percentile(v, q) {
    if (!v.length) return NaN;
    const s = Float64Array.from(v).sort(), x = q / 100 * (s.length - 1), i = Math.floor(x);
    return i + 1 < s.length ? s[i] + (x - i) * (s[i + 1] - s[i]) : s[i];
}
const lo = norm.value[0], hi = norm.value[1], n = Math.round(smooth.value), N = sources.length;
const sm = sources.map(s => [boxcar(s.data.raw_flux, n), boxcar(s.data.raw_err, n).map(x => x / Math.sqrt(n))]);
const wave = sources.map(s => s.data.wave);
const scale = new Array(N).fill(NaN), how = new Array(N).fill('');
const done = i => Number.isFinite(scale[i]);
const good = x => Number.isFinite(x) && x > 0;
// 1. photoscaling
for (let i = 0; i < N; i++)
    for (const b of bands)
        if (Number.isFinite(meta[i].scale[b])) { scale[i] = meta[i].scale[b]; how[i] = ' (photoscaled to ' + b + ')'; break; }
// 2. normalisation in the window
const covers = i => Math.max(0, Math.min(hi, meta[i].wmax) - Math.max(lo, meta[i].wmin)) / (hi - lo) >= coverage;
const inwin = [];
for (let i = 0; i < N; i++) if (done(i) && covers(i)) inwin.push(window_median(wave[i], sm[i][0], lo, hi) * scale[i]);
const ref = median(inwin.filter(good));
for (let i = 0; i < N; i++) {
    if (done(i) || !covers(i)) continue;
    const m = window_median(wave[i], sm[i][0], lo, hi);
    if (good(m)) { scale[i] = (good(ref) ? ref : 1) / m; how[i] = ' (normalised)'; }
}
if (!scale.some(Number.isFinite)) {
    let k = 0;
    for (let i = 1; i < N; i++) if (meta[i].wmax - meta[i].wmin > meta[k].wmax - meta[k].wmin) k = i;
    const m = median(sm[k][0]);
    scale[k] = good(m) ? 1 / m : 1; how[k] = ' (normalised)';
}
// 3. overlaps with scaled spectra, until nothing changes
for (let changed = true; changed;) {
    changed = false;
    for (let j = 0; j < N; j++) {
        if (done(j)) continue;
        for (let k = 0; k < N; k++) {
            if (!done(k) || k == j) continue;
            const olo = Math.max(meta[j].wmin, meta[k].wmin), ohi = Math.min(meta[j].wmax, meta[k].wmax);
            if (ohi <= olo) continue;
            const mj = window_median(wave[j], sm[j][0], olo, ohi), mk = window_median(wave[k], sm[k][0], olo, ohi) * scale[k];
            if (!good(mj) || !good(mk)) continue;
            scale[j] = mk / mj; how[j] = ' (matched in overlap)'; changed = true;
            break;
        }
    }
}
// 4. the nearest edge of a scaled spectrum
for (let j = 0; j < N; j++) {
    if (done(j)) continue;
    let best = null;
    for (let k = 0; k < N; k++) {
        if (!done(k) || k == j) continue;
        const red = meta[j].wmin >= meta[k].wmax, gap = red ? meta[j].wmin - meta[k].wmax : meta[k].wmin - meta[j].wmax;
        if (best === null || gap < best[1]) best = [k, gap, red];
    }
    if (best === null) { scale[j] = 1; continue; }
    const [k, , red] = best, wj = 0.05 * (meta[j].wmax - meta[j].wmin), wk = 0.05 * (meta[k].wmax - meta[k].wmin);
    const mj = red ? window_median(wave[j], sm[j][0], meta[j].wmin, meta[j].wmin + wj)
                   : window_median(wave[j], sm[j][0], meta[j].wmax - wj, meta[j].wmax);
    const mk = scale[k] * (red ? window_median(wave[k], sm[k][0], meta[k].wmax - wk, meta[k].wmax)
                               : window_median(wave[k], sm[k][0], meta[k].wmin, meta[k].wmin + wk));
    scale[j] = good(mj) && good(mk) ? mk / mj : scale[k];
    how[j] = ' (approximate: no overlap)';
}
// scaled fluxes, legend, window marks, axis label and the y range fitted to every spectrum
const all = [];
for (let i = 0; i < N; i++) {
    const d = sources[i].data, f = sm[i][0], e = sm[i][1];
    d.flux = f.map(x => x * scale[i]);
    d.lower = f.map((x, k) => (x - e[k]) * scale[i]);
    d.upper = f.map((x, k) => (x + e[k]) * scale[i]);
    sources[i].change.emit();
    items[i].label = {value: meta[i].label + how[i]};
    if (lines[i].visible) for (const x of d.flux) if (Number.isFinite(x)) all.push(x);
}
marks[0].location = lo; marks[1].location = hi;
for (const mark of marks) mark.visible = how.includes(' (normalised)');  // only while the window is in use
yaxis.axis_label = how.some(h => h.includes('photoscaled')) ? 'Fλ [erg s⁻¹ cm⁻² Å⁻¹]' : 'Normalised flux';
if (all.length > 10) {
    const p1 = percentile(all, 1), p99 = percentile(all, 99), pad = 0.08 * (p99 - p1 || Math.abs(p99) || 1);
    const ylo = p1 < -0.02 * Math.abs(p99) ? p1 - pad : 0, yhi = p99 + pad;
    yr.start = ylo; yr.end = yhi; yr.reset_start = ylo; yr.reset_end = yhi;
}
"""

JS_LAYOUT_LABELS = """
// labels of the chosen feature set, placed in up to three rows along the top without overlapping at the current zoom;
// a label which does not fit is hidden until the plot is zoomed in (its band or line is always shown)
const cls = sel.value, xr = p.x_range, yr = p.y_range, W = p.inner_width > 50 ? p.inner_width : 700;
const last = [-1e9, -1e9, -1e9], cand = [];
for (const l of labs) { if (l.tags[1] == cls) cand.push(l); else l.visible = false; }
cand.sort((a, b) => Math.max(a.tags[2], xr.start) - Math.max(b.tags[2], xr.start));
for (const l of cand) {
    // a band's label goes to the middle of the part of the band in view
    const lo = Math.max(l.tags[2], xr.start), hi = Math.min(l.tags[3], xr.end);
    if (hi < lo) { l.visible = false; continue; }
    l.x = 0.5 * (lo + hi);
    const px = (l.x - xr.start) / (xr.end - xr.start) * W, w = 6.5 * l.text.length + 8;
    if (px < 0 || px > W) { l.visible = false; continue; }
    const r = [0, 1, 2].find(k => px - w / 2 > last[k]);
    if (r === undefined) { l.visible = false; continue; }
    last[r] = px + w / 2;
    l.tags = [r, cls, l.tags[2], l.tags[3]];
    l.y = yr.end - [0.05, 0.11, 0.17][r] * (yr.end - yr.start);
    l.visible = true;
}
"""

JS_KEEP_LABELS_ON_TOP = """
// keep the feature labels in their rows along the top (and the telluric marks along the bottom), whatever the zoom
for (const lab of labels) lab.y = cb_obj.end - [0.05, 0.11, 0.17, 0.95][lab.tags[0]] * (cb_obj.end - cb_obj.start);
"""


# ---------------------------------------------------------------------------------------------- data
def read_spectrum(url: str):
    """
    Wavelength (micron), flux (F_lambda in erg s-1 cm-2 A-1 where its units allow, otherwise as stored) and
    uncertainty (NaN where none) of a spectrum, sorted by wavelength and finite. specutils reads most files; plain
    text tables (wavelength, flux, uncertainty columns) and 1D/IRAF multispec image FITS it cannot read are read here,
    with wavelengths converted by to_micron(). Raises ValueError when nothing can read it.
    """
    url = quote(url, safe=':/?&=%+#')  # e.g. a unicode minus in a file name
    try:
        cached = download_file(url, cache=True)
    except Exception as error:
        raise ValueError(f'unreadable ({error})') from None
    # specutils guesses formats partly by file name, which the cached download lacks, and would otherwise download
    # the url again for every format it tries: it reads a link carrying the original name
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, os.path.basename(urlparse(url).path) or 'spectrum')
        os.symlink(cached, path)
        return _read_any(path)


def _read_any(path: str):
    """read_spectrum() of a local file."""
    try:
        with warnings.catch_warnings():  # header fixes and format hints from archival files, which only fill the log
            warnings.simplefilter('ignore')
            spectrum = Spectrum.read(path)
        wave = spectrum.spectral_axis.to(u.micron, equivalencies=u.spectral()).value
        stored = spectrum.flux.value
        # standard deviation in the units the flux is stored in
        err = spectrum.uncertainty.represent_as(StdDevUncertainty).array if spectrum.uncertainty is not None \
            else np.full(stored.shape, np.nan)
        equivalency = u.spectral_density(spectrum.spectral_axis)
        if spectrum.flux.unit.is_equivalent(FLAMBDA, equivalencies=equivalency):
            flux = spectrum.flux.to_value(FLAMBDA, equivalencies=equivalency)
            with np.errstate(all='ignore'):
                err = err * flux / stored  # the conversion is a factor per pixel
        else:
            flux = stored
    except Exception as specutils_error:
        # the extension only sets the order: some text tables are named .fits
        readers = (_read_image, _read_table) if path.lower().endswith(('.fits', '.fits.gz', '.fit')) \
            else (_read_table, _read_image)
        for reader in readers:
            try:
                wave, flux, err = reader(path)
                break
            except Exception:
                continue
        else:
            raise ValueError(f'unreadable ({str(specutils_error).splitlines()[0]})') from None
    wave, flux, err = (np.asarray(x, dtype=float).ravel() for x in (wave, flux, err))
    if not len(wave) == len(flux) == len(err):
        raise ValueError('wavelength, flux and uncertainty lengths differ')
    ok = np.isfinite(wave) & np.isfinite(flux) & ~np.isclose(flux, 0, atol=1e-30)
    if ok.sum() < 2:
        raise ValueError('no valid flux')
    err[~np.isfinite(err) | (err <= 0)] = np.nan
    order = np.argsort(wave[ok])
    return wave[ok][order], flux[ok][order], err[ok][order]


def to_micron(wave: np.ndarray) -> np.ndarray:
    """Wavelengths of unknown unit in micron: no ultracool dwarf spectrum has values under 100 (micron) in nm or
    Angstrom, nor over 4000 in nm (4 micron) or under 4000 in Angstrom (0.4 micron)."""
    median = np.nanmedian(wave)
    return wave / (1 if median < 100 else 1e3 if median < 4000 else 1e4)


def _read_table(path: str):
    """Columns of a whitespace or comma separated text spectrum: wavelength, flux and (when present) uncertainty.
    Fluxes declared in Jy or mJy (IPAC table units line, as in CASSIS Spitzer spectra, or YUNITS) become F_lambda;
    other units differ from F_lambda by a constant at most, which the scaling absorbs."""
    with open(path, errors='replace') as fh:
        lines = [line for line in fh if line.strip()]
    numeric = [line for line in lines if line.lstrip()[0] in '0123456789+-.']
    data = np.loadtxt(numeric, ndmin=2, delimiter=',' if ',' in numeric[0] else None)
    wave = to_micron(data[:, 0])
    flux, err = data[:, 1], data[:, 2] if data.shape[1] > 2 else np.full(len(data), np.nan)
    header = ''.join(line for line in lines if line.lstrip()[0] not in '0123456789+-.')
    fnu = re.search(r"YUNITS\s*=\s*'\s*(m?Jy)\b|\|\s*(?:microns?|um)\s*\|\s*(m?Jy)\s*\|", header)
    if fnu:
        to_flambda = u.Unit(fnu.group(1) or fnu.group(2)).to(FLAMBDA, equivalencies=u.spectral_density(wave * u.micron))
        flux, err = flux * to_flambda, err * to_flambda
    return wave, flux, err


def _read_image(path: str):
    """An image spectrum: rows (SpeX) or columns (X-shooter) of wavelength, flux and uncertainty when there is no
    wavelength solution, otherwise a 1D spectrum with a linear wavelength solution; for IRAF multispec cubes the first
    aperture, with the sigma band as uncertainty when there is one."""
    with fits.open(path) as hdus:
        head, data = hdus[0].header, np.asarray(hdus[0].data, dtype=float)
    if 'CRVAL1' not in head and data.ndim == 2:
        rows = data.T if data.shape[1] < data.shape[0] else data  # spectrum stored as columns
        if 2 <= rows.shape[0] <= 4:
            return to_micron(rows[0]), rows[1], rows[2] if rows.shape[0] > 2 else np.full(rows.shape[1], np.nan)
    n = head['NAXIS1']
    rows = data.reshape(-1, n)
    bands = [str(head.get(f'BANDID{i + 1}', '')).lower() for i in range(rows.shape[0])]
    sigma = next((i for i, b in enumerate(bands) if 'sigma' in b), None)
    step = head.get('CDELT1', head.get('CD1_1'))
    wave = head['CRVAL1'] + step * (np.arange(n) + 1 - head.get('CRPIX1', 1))
    return to_micron(wave), rows[0], rows[sigma] if sigma is not None else np.full(n, np.nan)


def load_filters() -> dict:
    """{band: (wavelength [micron], transmission, F_lambda zero point [erg s-1 cm-2 A-1])} from filters.json."""
    with open(os.path.join(os.path.dirname(__file__), 'filters.json')) as fh:
        raw = json.load(fh)
    return {b: (np.array(v['wave']), np.array(v['trans']), v['zp_flam']) for b, v in raw.items() if b in BANDS}


FILTERS = load_filters()


def synflux(wave: np.ndarray, flux: np.ndarray, filt: tuple):
    """Photon-weighted mean F_lambda of a spectrum through a filter, and the fraction of the filter it covers."""
    fw, ft, _ = filt
    ok = np.isfinite(flux)
    if ok.sum() < 2:
        return np.nan, 0.
    f = np.interp(fw, wave[ok], flux[ok], left=np.nan, right=np.nan)
    cov = np.isfinite(f)
    w = ft * fw
    tot = np.trapz(w, fw)
    if cov.sum() < 2 or tot <= 0:
        return np.nan, 0.
    return np.trapz(np.where(cov, f, 0.) * w, fw) / np.trapz(w * cov, fw), np.trapz(w * cov, fw) / tot


def magnitudes(photometry: list) -> dict:
    """{band: median magnitude} of a source's photometry (inventory rows), for the bands it can be photoscaled to."""
    if not photometry:
        return {}
    meds = pd.DataFrame(photometry).groupby('band').magnitude.median()
    return {b: float(meds[sb]) for b, sb in BANDS.items() if sb in meds and np.isfinite(meds[sb])}


def bin_for_display(wave, flux, err):
    """Medians in blocks of n consecutive pixels (uncertainties combined) for spectra longer than MAXPOINTS, and n."""
    n = int(np.ceil(len(wave) / MAXPOINTS))
    if n <= 1:
        return wave, flux, err, 1
    m = len(wave) // n * n
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        w = np.nanmean(wave[:m].reshape(-1, n), axis=1)
        f = np.nanmedian(flux[:m].reshape(-1, n), axis=1)
        e2 = err[:m].reshape(-1, n) ** 2
        e = np.sqrt(np.nanmean(e2, axis=1) / np.maximum(np.sum(np.isfinite(e2), axis=1), 1))
    return w, f, e, n


def default_window(wmin: float):
    """Normalisation window (centre, width), as in the GUCDS spectra browser."""
    if wmin < 0.8:
        return 0.815, 0.01
    if wmin < 1.4:
        return 1.50, 0.10
    return None


# ---------------------------------------------------------------------------------------------- plot
def _try_read(spec: dict):
    """read_spectrum() of a Spectra row, or the ValueError saying why it could not be read."""
    try:
        return read_spectrum(spec['access_url'])
    except ValueError as error:
        return error


def spectra_panel(source: str, spectra: list, mags: dict, theme: Theme):
    """
    The script and div of the spectra plot of a source, and the descriptions of the spectra that could not be read

    Parameters
    ----------
    source
        The source name
    spectra
        The Spectra table rows of the source (dicts with access_url, telescope, instrument, mode, observation_date,
        reference)
    mags
        {band: magnitude} for photoscaling, see magnitudes()
    theme
        The bokeh theme

    Returns
    -------
    script, div
        The plot, both None when no spectrum could be read
    failures
        Description of each spectrum which could not be read
    """
    with ThreadPoolExecutor(max_workers=8) as pool:  # mostly waiting on downloads
        read = list(pool.map(_try_read, spectra))
    sources, meta, failures = [], [], []
    for spec, result in zip(spectra, read):
        if isinstance(result, ValueError):
            name = '/'.join(str(spec[k]) for k in ('telescope', 'instrument', 'mode')
                            if spec[k] not in (None, 'Missing'))
            failures.append(f'{name} ({spec["reference"]}): {result}')
            continue
        wave, flux, err = result
        scales = {}
        for band, mag in mags.items():
            sf, cov = synflux(wave, flux, FILTERS[band])
            if cov >= COVERAGE and np.isfinite(sf) and sf != 0:
                scales[band] = float(FILTERS[band][2] * 10 ** (-0.4 * mag) / sf)
        date = pd.Timestamp(spec['observation_date']).date() if spec['observation_date'] is not None else ''
        label = f'{spec["telescope"]}-{spec["instrument"]}: {date}'
        wave, flux, err, nbin = bin_for_display(wave, flux, err)
        if nbin > 1:
            label += f' [binned ×{nbin}]'
        sources.append(ColumnDataSource(data=dict(
            wave=wave.astype(np.float32), raw_flux=flux.astype(np.float32), raw_err=err.astype(np.float32),
            flux=flux.astype(np.float32), lower=(flux - err).astype(np.float32), upper=(flux + err).astype(np.float32),
            snr=np.where(err > 0, flux / err, np.nan).astype(np.float32))))
        meta.append(dict(label=label, scale=scales, wmin=float(wave.min()), wmax=float(wave.max()),
                         has_err=bool(np.isfinite(err).any())))
    if not sources:
        return None, None, failures
    labels = [m['label'] for m in meta]
    for i, lab in enumerate(labels):  # the same instrument and date twice
        if labels.count(lab) > 1:
            meta[i]['label'] = f'{lab} ({i + 1})'

    wmin, wmax = min(m['wmin'] for m in meta), max(m['wmax'] for m in meta)
    pad = 0.02 * (wmax - wmin)
    xlo, xhi = wmin - pad, wmax + pad
    centre, width = default_window(wmin) or (float(np.median(np.concatenate([s.data['wave'] for s in sources]))), 0.2)
    p = figure(title='Spectra', outer_height=500, active_scroll='wheel_zoom', active_drag='box_zoom',
               tools='pan,wheel_zoom,box_zoom,save,reset', toolbar_location='left', sizing_mode='stretch_width',
               x_range=(xlo, xhi), x_axis_label='Wavelength [μm]', y_axis_label='Normalised flux')
    p = bokeh_formatter(p)
    lines, items, bands_err = [], [], []
    for i, (src, m) in enumerate(zip(sources, meta)):
        colour = COLOURS[i % len(COLOURS)]
        line = p.line('wave', 'flux', source=src, line_color=colour, line_width=2, name=m['label'],
                      line_dash=DASHES[i // len(COLOURS) % len(DASHES)], legend_label=m['label'])
        band = Band(base='wave', lower='lower', upper='upper', source=src, fill_color=colour, fill_alpha=0.2,
                    line_width=0, level='underlay', visible=m['has_err'])
        p.add_layout(band)
        lines.append(line)
        items.append(p.legend.items[-1])
        bands_err.append(band)
    p.legend.click_policy = 'hide'
    p.legend.label_text_font_size = '1.2em'
    p.add_tools(HoverTool(renderers=lines, tooltips=[('', '$name'), ('λ', '@wave{0.0000} μm'),
                                                     ('flux', '@flux{0.000e}'), ('S/N', '@snr{0.0}')]))
    marks = [Span(location=x, dimension='height', line_color='white', line_dash='dashed', visible=False)
             for x in (centre - 0.5 * width, centre + 0.5 * width)]
    for mark in marks:
        p.add_layout(mark)

    # features: shaded bands and dashed lines with labels along the top, one set per spectral class (shown with the
    # Spectral features menu); tellurics shaded with a ⊕ along the bottom
    char = 0.011 * (xhi - xlo)  # approximate width of one character of a label, as a fraction of the range
    feature_sets, feature_labels, edge_labels, telluric_objs = {}, [], [], []
    for cls, feats in FEATURE_SETS.items():
        objs = feature_sets.setdefault(cls, [])
        candidates = []  # (x, name, edges) of every label, placed below from blue to red
        for kind, name, waves in feats:
            for w in waves:
                lo, hi = (w if kind == 'band' else (w, w))
                if hi < xlo or lo > xhi:
                    continue
                obj = BoxAnnotation(left=lo, right=hi, fill_color='white', fill_alpha=0.08, visible=False,
                                    level='underlay') if kind == 'band' else \
                    Span(location=lo, dimension='height', line_color='white', line_dash='dotted', visible=False)
                p.add_layout(obj)
                objs.append(obj)
            if kind == 'band':  # one label per band, which knows its edges
                candidates += [(0.5 * (w[0] + w[1]), name, w[0], w[1]) for w in waves if w[1] > xlo and w[0] < xhi]
                continue
            groups = []  # lines of one multiplet closer than 3 per cent of the range share one label
            for x in sorted(w for w in waves if xlo < w < xhi):
                if groups and x - groups[-1][-1] < 0.03 * (xhi - xlo):
                    groups[-1].append(x)
                else:
                    groups.append([x])
            candidates += [(float(np.mean(g)), name, float(np.mean(g)), float(np.mean(g))) for g in groups]
        last = [(-1e9, ''), (-1e9, ''), (-1e9, '')]
        for x, name, edge0, edge1 in sorted(candidates):
            gap = [x - last[r][0] - (0.5 * (len(last[r][1]) + len(name)) + 1) * char for r in range(3)]
            row_ = next((r for r in range(3) if gap[r] > 0), int(np.argmax(gap)))  # every feature is labelled
            last[row_] = (x, name)
            lab = Label(x=x, y=0, text=name, text_color='white', text_align='center', visible=False,
                        tags=[row_, cls, edge0, edge1])
            p.add_layout(lab)
            objs.append(lab)
            feature_labels.append(lab)
            edge_labels.append(lab)
    for lo, hi in TELLURIC:
        if hi < xlo or lo > xhi:
            continue
        box = BoxAnnotation(left=lo, right=hi, fill_color='white', fill_alpha=0.10, visible=False, level='underlay')
        lab = Label(x=0.5 * (lo + hi), y=0, text='⊕', text_color='white', text_align='center', visible=False,
                    tags=[3])
        for obj in (box, lab):
            p.add_layout(obj)
            telluric_objs.append(obj)
        edge_labels.append(lab)
    keep = CustomJS(args=dict(labels=edge_labels), code=JS_KEEP_LABELS_ON_TOP)
    p.y_range.js_on_change('start', keep)
    p.y_range.js_on_change('end', keep)

    # controls: feature, telluric and uncertainty switches beside the plot, scaling below it
    featsel = Select(title='Spectral features', value='None', options=['None'] + list(FEATURE_SETS), width=180)
    telluric = Toggle(label='Telluric bands', width=180)
    errtog = Toggle(label='Uncertainties', active=True, width=180)
    norm = RangeSlider(start=round(xlo, 2), end=round(xhi, 2), value=(centre - 0.5 * width, centre + 0.5 * width),
                       step=0.005, title='Normalisation window [μm]', sizing_mode='stretch_width',
                       visible=not all(m['scale'] for m in meta))  # only spectra not photoscaled can use it
    smooth = Slider(start=1, end=25, value=1, step=2, title='Smoothing [pixels]', width=300)
    scale = CustomJS(args=dict(sources=sources, meta=meta, bands=list(BANDS), coverage=COVERAGE, norm=norm,
                               smooth=smooth, items=items, lines=lines, marks=marks, yaxis=p.yaxis[0], yr=p.y_range),
                     code=JS_SCALE)
    for widget in (norm, smooth):
        widget.js_on_change('value', scale)
    for line in lines:
        line.js_on_change('visible', scale)
    layout_labels = CustomJS(args=dict(labs=feature_labels, sel=featsel, p=p), code=JS_LAYOUT_LABELS)
    featsel.js_on_change('value', CustomJS(args=dict(sets=feature_sets, layout=layout_labels), code="""
        for (const [cls, objs] of Object.entries(sets))
            for (const o of objs) if (o.type != 'Label') o.visible = (cls == cb_obj.value);
        layout.execute(cb_obj);"""))
    for attr in ('start', 'end'):
        p.x_range.js_on_change(attr, layout_labels)
    p.js_on_change('inner_width', layout_labels)
    telluric.js_on_change('active', CustomJS(args=dict(objs=telluric_objs), code="""
        for (const o of objs) o.visible = cb_obj.active;"""))
    errtog.js_on_change('active', CustomJS(args=dict(bands=bands_err, lines=lines, meta=meta), code="""
        bands.forEach((b, i) => { b.visible = cb_obj.active && meta[i].has_err && lines[i].visible; });"""))
    for band, line, m in zip(bands_err, lines, meta):  # an uncertainty band follows its spectrum when hidden
        line.js_on_change('visible', CustomJS(args=dict(b=band, tog=errtog, has=m['has_err']),
                                              code='b.visible = cb_obj.visible && tog.active && has;'))

    layout = column(row(p, column(featsel, telluric, errtog, width=200), sizing_mode='stretch_width'),
                    row(norm, smooth, sizing_mode='stretch_width'), sizing_mode='stretch_width')
    Document().add_root(layout)
    layout.document.js_on_event('document_ready', scale)  # scales the spectra as the page opens
    script, div = components(layout, theme=theme)
    return script, div, failures
