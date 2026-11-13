"""
Interactive spectra plot of a source page, ported from the GUCDS spectra browser.

The source's spectra share one panel with Linear and Log tabs; everything below is done in the browser:
- the flux can be shown as stored, normalised by the median in a window (wavelength and width sliders), or photoscaled
  to a SIMPLE magnitude (2MASS J, H, Ks, Gaia G, RP, WISE W1, W2) through the filter curves of the SVO Filter Profile
  Service (filters.json); a spectrum that does not cover the chosen filter is normalised to the photoscaled spectra in
  the normalisation window instead, and its legend says which was done;
- boxcar smoothing, uncertainties as shaded bands, spectral features and telluric bands.
"""
import json
import os
import tempfile
import warnings
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
from bokeh.models import (Band, BoxAnnotation, CheckboxGroup, ColumnDataSource, CustomJS, Div, HoverTool,
                          InlineStyleSheet, Label, Legend, LegendItem, PrintfTickFormatter, RadioButtonGroup, Range1d,
                          RangeSlider, Select, Slider, Span, TabPanel, Tabs, Toggle)
from bokeh.plotting import figure
from bokeh.themes import Theme
from specutils import Spectrum

COVERAGE = 0.9  # fraction of a filter (photoscaling) or of the normalisation window a spectrum must cover to be rescaled
MAXPOINTS = 8000  # longer spectra are binned to about this many points for display
# colour-blind safe on the dark theme: the GUCDS order (black, blue, gold...) with white and Paul Tol's light blue
COLOURS = ('#FFFFFF', '#77AADD', '#DDAA33', '#EE8866', '#44BB99', '#99DDFF', '#FFAABB', '#BBCC33')
FEATURE_COLOUR, TELLURIC_COLOUR = '#BBBBBB', '#99DDFF'
FLAMBDA = u.erg / u.s / u.cm ** 2 / u.AA
# filters.json band: SIMPLE photometry band
BANDS = {'J': '2MASS.J', 'H': '2MASS.H', 'Ks': '2MASS.Ks', 'G': 'GAIA3.G', 'RP': 'GAIA3.Grp',
         'W1': 'WISE.W1', 'W2': 'WISE.W2'}
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
JS_FUNCTIONS = """
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
"""

JS_UPDATE = JS_FUNCTIONS + """
const mode = modes.active;                       // 0 as stored, 1 normalise, 2 photoscale
const band = bandsel.value;
const lo = centre.value - 0.5 * width.value, hi = centre.value + 0.5 * width.value;
const n = Math.round(smooth.value);
// 1. smoothed flux and uncertainty of every spectrum
const sm = [];
for (let i = 0; i < sources.length; i++) {
    const d = sources[i].data;
    const f = boxcar(d.raw_flux, n), e = boxcar(d.raw_err, n).map(x => x / Math.sqrt(Math.max(n, 1)));
    sm.push([f, e]);
}
// 2. scale factor and legend label of every spectrum
const scale = new Array(sources.length).fill(1), how = new Array(sources.length).fill('');
let ref = NaN;   // median of the photoscaled spectra in the window, for the spectra normalised to them
if (mode == 2) {
    const meds = [];
    for (let i = 0; i < sources.length; i++) {
        const s = meta[i].scale[band];
        if (s !== undefined && s !== null && Number.isFinite(s)) {
            scale[i] = s; how[i] = ' (photoscaled to ' + band + ')';
            const m = window_median(sources[i].data.wave, sm[i][0], lo, hi);
            if (Number.isFinite(m) && meta[i].wmin < lo && meta[i].wmax > hi) meds.push(m * s);
        } else how[i] = ' (normalised)';
    }
    ref = meds.length ? median(meds) : 1;
}
for (let i = 0; i < sources.length; i++) {
    if (mode == 1 || (mode == 2 && how[i] == ' (normalised)')) {
        // only a spectrum covering most of the window is normalised; any other is left as stored
        const covered = Math.max(0, Math.min(hi, meta[i].wmax) - Math.max(lo, meta[i].wmin)) / (hi - lo);
        if (covered >= coverage) {
            const m = window_median(sources[i].data.wave, sm[i][0], lo, hi);
            scale[i] = (Number.isFinite(m) && m != 0 ? 1 / Math.abs(m) : 1) * (mode == 2 ? ref : 1);
            if (mode == 1) how[i] = ' (normalised)';
        } else {
            scale[i] = 1;
            how[i] = ' (as stored: outside window)';
        }
    }
    const d = sources[i].data, f = sm[i][0], e = sm[i][1];
    const nf = new Float64Array(f.length), nl = new Float64Array(f.length), nu = new Float64Array(f.length);
    for (let k = 0; k < f.length; k++) {
        nf[k] = f[k] * scale[i];
        nl[k] = (f[k] - e[k]) * scale[i];
        nu[k] = (f[k] + e[k]) * scale[i];
    }
    d.flux = nf; d.lower = nl; d.upper = nu;
    sources[i].change.emit();
    for (const item of items[i]) item.label = {value: meta[i].label + how[i]};
    const showmark = how[i] == ' (normalised)';
    for (const sp of marks[i]) { sp.visible = showmark && lines[i][0].visible; }
    marks[i][0].location = lo; marks[i][1].location = hi; marks[i][2].location = lo; marks[i][3].location = hi;
}
// 3. axis labels, and the plotting window: the y-range is fitted to the visible spectra in the shown wavelength range,
//    and the reset button returns to a window showing every visible spectrum in full, for this mode
const ylab = ['Flux as stored', 'Normalised flux', 'Fλ [erg s⁻¹ cm⁻² Å⁻¹]'][mode];
for (const ax of yaxes) ax.axis_label = ylab;
const xr = figs[0].x_range, shown = [], all = [];
// spectra left as stored do not set the flux range while others are rescaled, which they would dwarf or flatten
const unscaled = how.map(h => h.includes('outside window')), anyscaled = unscaled.some(x => !x);
let wlo = Infinity, whi = -Infinity;
for (let i = 0; i < sources.length; i++) {
    if (!lines[i][0].visible) continue;
    const d = sources[i].data, sets_flux = !(unscaled[i] && anyscaled);
    for (let k = 0; k < d.wave.length; k++) {
        if (!Number.isFinite(d.flux[k])) continue;
        if (d.wave[k] < wlo) wlo = d.wave[k];
        if (d.wave[k] > whi) whi = d.wave[k];
        if (!sets_flux) continue;
        all.push(d.flux[k]);
        if (d.wave[k] >= xr.start && d.wave[k] <= xr.end) shown.push(d.flux[k]);
    }
}
function linrange(v) {   // 1st-99th percentiles, padded; from zero unless the fluxes go clearly negative
    const lo = percentile(v, 1), hi = percentile(v, 99), pad = 0.08 * (hi - lo || Math.abs(hi) || 1);
    return [lo < -0.02 * Math.abs(hi) ? lo - pad : 0, hi + pad];
}
function logrange(v) {
    const pos = v.filter(x => x > 0);
    return pos.length > 10 ? [percentile(pos, 1) / 2, percentile(pos, 99) * 2] : null;
}
if (all.length > 10) {
    const [ylo, yhi] = linrange(shown.length > 10 ? shown : all);
    const [rlo, rhi] = linrange(all);
    const wpad = 0.02 * (whi - wlo);
    for (const f of figs) { f.x_range.reset_start = wlo - wpad; f.x_range.reset_end = whi + wpad; }
    figs[0].y_range.reset_start = rlo; figs[0].y_range.reset_end = rhi;
    // slider bounds around the full range of this mode in the tab shown, before the ranges (and so the slider) change
    const lrr0 = logrange(all);
    if (tabs.active == 1 && lrr0 !== null) {
        yslider.start = lrr0[0] / 10; yslider.end = lrr0[1] * 10; yslider.step = (yslider.end - yslider.start) / 400;
    } else {
        yslider.start = rlo - 0.5 * (rhi - rlo); yslider.end = rhi + 0.5 * (rhi - rlo); yslider.step = (rhi - rlo) / 400;
    }
    figs[0].y_range.start = ylo; figs[0].y_range.end = yhi;
    const lr = logrange(shown.length > 10 ? shown : all), lrr = logrange(all);
    if (lr !== null) { figs[1].y_range.start = lr[0]; figs[1].y_range.end = lr[1]; }
    if (lrr !== null) { figs[1].y_range.reset_start = lrr[0]; figs[1].y_range.reset_end = lrr[1]; }
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
    const f = [0.05, 0.11, 0.17][r];
    l.tags = [r, cls, l.tags[2], l.tags[3]];
    l.y = (is_log && yr.start > 0) ? Math.pow(10, Math.log10(yr.end) - f * (Math.log10(yr.end) - Math.log10(yr.start)))
                                   : yr.end - f * (yr.end - yr.start);
    l.visible = true;
}
"""

JS_KEEP_LABELS_ON_TOP = """
// keep the feature labels in their rows along the top of the plot, whatever the zoom
const yr = cb_obj, logy = is_log;
for (const lab of labels) {
    const f = [0.05, 0.11, 0.17, 0.95][lab.tags[0]];
    if (logy && yr.start > 0) lab.y = Math.pow(10, Math.log10(yr.end) - f * (Math.log10(yr.end) - Math.log10(yr.start)));
    else lab.y = yr.end - f * (yr.end - yr.start);
}
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
    """Columns of a whitespace or comma separated text spectrum: wavelength, flux and (when present) uncertainty."""
    with open(path, errors='replace') as fh:  # only the lines starting with a number
        lines = [line for line in fh if line.strip() and line.lstrip()[0] in '0123456789+-.']
    data = np.loadtxt(lines, ndmin=2, delimiter=',' if ',' in lines[0] else None)
    return to_micron(data[:, 0]), data[:, 1], data[:, 2] if data.shape[1] > 2 else np.full(len(data), np.nan)


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
def spectra_panel(source: str, spectra: list, mags: dict, theme: Theme):
    """
    The script and div of the spectra plot of a source, and the descriptions of the spectra that could not be read

    Parameters
    ----------
    source
        The source name, as plot title
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
    sources, meta, failures = [], [], []
    for spec in spectra:
        name = '/'.join(str(spec[k]) for k in ('telescope', 'instrument', 'mode') if spec[k] not in (None, 'Missing'))
        try:
            wave, flux, err = read_spectrum(spec['access_url'])
        except ValueError as error:
            failures.append(f'{name} ({spec["reference"]}): {error}')
            continue
        date = pd.Timestamp(spec['observation_date']).date() if spec['observation_date'] is not None else ''
        label = f'{spec["telescope"]}-{spec["instrument"]}: {date}'
        scales = {}
        for band, mag in mags.items():
            sf, cov = synflux(wave, flux, FILTERS[band])
            if cov >= COVERAGE and np.isfinite(sf) and sf != 0:
                scales[band] = float(FILTERS[band][2] * 10 ** (-0.4 * mag) / sf)
        wave, flux, err, nbin = bin_for_display(wave, flux, err)
        if nbin > 1:
            label += f' [binned ×{nbin}]'
        snr = np.where(err > 0, flux / err, np.nan)
        sources.append(ColumnDataSource(data=dict(
            wave=wave.astype(np.float32), raw_flux=flux.astype(np.float32), raw_err=err.astype(np.float32),
            flux=flux.astype(np.float32), lower=(flux - err).astype(np.float32), upper=(flux + err).astype(np.float32),
            snr=snr.astype(np.float32))))
        meta.append(dict(label=label, scale=scales, wmin=float(np.nanmin(wave)), wmax=float(np.nanmax(wave)),
                         has_err=bool(np.isfinite(err).any())))
    if not sources:
        return None, None, failures
    labels = [m['label'] for m in meta]
    for i, lab in enumerate(labels):  # the same instrument and date twice
        if labels.count(lab) > 1:
            meta[i]['label'] = f'{lab} ({i + 1})'

    wmin = min(m['wmin'] for m in meta)
    win = default_window(wmin) or (float(np.median(np.concatenate([s.data['wave'] for s in sources]))), 0.2)
    available = [b for b in BANDS if b in mags and any(b in m['scale'] for m in meta)]
    # the plot opens photoscaled to the first band, or normalised; spectra it leaves as stored (see JS_UPDATE) start
    # hidden while others are rescaled, as their flux would be off the scale (shown again from the legend)
    lo_, hi_ = win[0] - 0.5 * win[1], win[0] + 0.5 * win[1]
    rescaled = [(available and available[0] in m['scale'])
                or max(0., min(hi_, m['wmax']) - max(lo_, m['wmin'])) / (hi_ - lo_) >= COVERAGE for m in meta]
    hidden = [not r and any(rescaled) for r in rescaled]

    # opening plot window around the spectra shown
    shown = [i for i, h in enumerate(hidden) if not h]
    wmin, wmax = min(meta[i]['wmin'] for i in shown), max(meta[i]['wmax'] for i in shown)
    pad = 0.02 * (wmax - wmin)
    xlo, xhi = wmin - pad, wmax + pad
    fpad = 0.02 * (max(m['wmax'] for m in meta) - min(m['wmin'] for m in meta))
    flo, fhi = min(m['wmin'] for m in meta) - fpad, max(m['wmax'] for m in meta) + fpad  # every spectrum
    allf = np.concatenate([sources[i].data['raw_flux'] for i in shown])
    allf = allf[np.isfinite(allf)]
    p1, p99 = (np.percentile(allf, (1, 99)) if len(allf) > 10 else (0., 1.))
    pad = 0.08 * ((p99 - p1) or abs(p99) or 1.)  # the same window as linrange() in the browser code
    ylo, yhi = (p1 - pad if p1 < -0.02 * abs(p99) else 0.), p99 + pad
    pos = allf[allf > 0]
    lylo, lyhi = (np.percentile(pos, 1) / 2, np.percentile(pos, 99) * 2) if len(pos) > 10 else (1e-3, 10.)

    n_spec = len(sources)
    figs, lines, bands_err, marks, items = [], [[] for _ in range(n_spec)], [[] for _ in range(n_spec)], \
        [[] for _ in range(n_spec)], [[] for _ in range(n_spec)]
    feature_sets, telluric_objs, top_labels, feature_labels = {}, [], [[], []], [[], []]
    for k, ytype in enumerate(('linear', 'log')):
        p = figure(tools='pan,box_zoom,wheel_zoom,reset,save', toolbar_location='left', title=source, height=640,
                   sizing_mode='stretch_width', active_drag='box_zoom', active_scroll='wheel_zoom', y_axis_type=ytype,
                   x_axis_label='Wavelength [µm]', y_axis_label='Flux as stored', x_range=Range1d(xlo, xhi),
                   y_range=Range1d(ylo, yhi) if ytype == 'linear' else Range1d(lylo, lyhi))
        p.title.text_font_size = '16pt'
        p.title.align = 'center'
        for ax in (p.xaxis, p.yaxis):
            ax.major_label_text_font_size = '12pt'
            ax.axis_label_text_font_size = '13pt'
        legend_items = []
        for i, (src, m) in enumerate(zip(sources, meta)):
            c = COLOURS[i % len(COLOURS)]
            ln = p.line('wave', 'flux', source=src, color=c, line_width=1.2, name=m['label'],
                        line_dash='solid' if i < len(COLOURS) else 'dashed', visible=not hidden[i])
            b = Band(base='wave', lower='lower', upper='upper', source=src, fill_color=c, fill_alpha=0.2,
                     line_width=0, level='underlay', visible=m['has_err'] and not hidden[i])
            p.add_layout(b)
            for loc in win[0] - 0.5 * win[1], win[0] + 0.5 * win[1]:
                sp = Span(location=loc, dimension='height', line_color=c, line_dash='dashed', line_width=1,
                          visible=False)
                p.add_layout(sp)
                marks[i].append(sp)
            item = LegendItem(label=m['label'], renderers=[ln])
            legend_items.append(item)
            items[i].append(item)
            lines[i].append(ln)
            bands_err[i].append(b)
        p.add_layout(Legend(items=legend_items, click_policy='hide', label_text_font_size='11pt', title='Spectra'),
                     'right')
        p.add_tools(HoverTool(renderers=[r for ls in lines for r in ls[-1:]], mode='mouse',
                              tooltips=[('', '$name'), ('λ', '@wave{0.0000} µm'), ('flux', '@flux{0.000e}'),
                                        ('S/N', '@snr{0.0}')]))

        # features: shaded bands and dashed lines with grey labels along the top, one set per spectral class (shown
        # with the Spectral features menu); tellurics shaded with a ⊕ along the bottom
        def label_y(r):  # height of a label in row r: 0-2 along the top, 3 along the bottom
            fr = (0.05, 0.11, 0.17, 0.95)[r]
            if ytype == 'log':
                return 10 ** (np.log10(lyhi) - fr * (np.log10(lyhi) - np.log10(lylo)))
            return yhi - fr * (yhi - ylo)

        char = 0.011 * (fhi - flo)  # approximate width of one character of a 9 pt label, as a fraction of the range
        for cls, feats in FEATURE_SETS.items():
            objs = feature_sets.setdefault(cls, [])
            candidates = []  # (x, name, edges) of every label, placed below from blue to red
            for kind, name, waves in feats:
                for w in waves:
                    lo_, hi_ = (w if kind == 'band' else (w, w))
                    if hi_ < flo or lo_ > fhi:
                        continue
                    if kind == 'band':
                        obj = BoxAnnotation(left=lo_, right=hi_, fill_color=FEATURE_COLOUR, fill_alpha=0.08,
                                            visible=False, level='underlay')
                    else:
                        obj = Span(location=lo_, dimension='height', line_color=FEATURE_COLOUR, line_dash='dashed',
                                   line_width=1, visible=False)
                    p.add_layout(obj)
                    objs.append(obj)
                if kind == 'band':  # one label per band, which knows its edges
                    candidates += [(0.5 * (w[0] + w[1]), name, w[0], w[1]) for w in waves
                                   if w[1] > flo and w[0] < fhi]
                    continue
                ws = sorted(w for w in waves if flo < w < fhi)
                groups = []  # lines of one multiplet closer than 3 per cent of the range share one label
                for x in ws:
                    if groups and x - groups[-1][-1] < 0.03 * (fhi - flo):
                        groups[-1].append(x)
                    else:
                        groups.append([x])
                candidates += [(float(np.mean(g)), name, float(np.mean(g)), float(np.mean(g))) for g in groups]
            last = [(-1e9, ''), (-1e9, ''), (-1e9, '')]
            for x, name, edge0, edge1 in sorted(candidates):
                gap = [x - last[r][0] - (0.5 * (len(last[r][1]) + len(name)) + 1) * char for r in range(3)]
                row_ = next((r for r in range(3) if gap[r] > 0), int(np.argmax(gap)))  # every feature is labelled
                last[row_] = (x, name)
                lab = Label(x=x, y=label_y(row_), text=name, text_font_size='9pt', text_color=FEATURE_COLOUR,
                            text_align='center', visible=False, tags=[row_, cls, edge0, edge1])
                feature_labels[k].append(lab)
                p.add_layout(lab)
                objs.append(lab)
                top_labels[k].append(lab)
        for lo_, hi_ in TELLURIC:
            if hi_ < flo or lo_ > fhi:
                continue
            box = BoxAnnotation(left=lo_, right=hi_, fill_color=TELLURIC_COLOUR, fill_alpha=0.10, visible=False,
                                level='underlay')
            lab = Label(x=0.5 * (lo_ + hi_), y=label_y(3), text='⊕', text_font_size='11pt', text_color=TELLURIC_COLOUR,
                        text_align='center', visible=False, tags=[3])
            for obj in (box, lab):
                p.add_layout(obj)
                telluric_objs.append(obj)
            top_labels[k].append(lab)
        keep = CustomJS(args=dict(labels=top_labels[k], is_log=(k == 1)), code=JS_KEEP_LABELS_ON_TOP)
        p.y_range.js_on_change('start', keep)
        p.y_range.js_on_change('end', keep)
        figs.append(p)
    figs[1].x_range = figs[0].x_range  # the two tabs share the wavelength range

    # controls
    active_style = InlineStyleSheet(css='.bk-btn.bk-active {background-color: #77AADD; color: #000; '
                                        'border-color: #77AADD; font-weight: bold;}')
    # SIMPLE spectra come in mixed units, so they open photoscaled when possible, otherwise normalised
    modes = RadioButtonGroup(labels=['As stored', 'Normalise'] + (['Photoscale'] if available else []),
                             active=2 if available else 1,
                             width=240, stylesheets=[active_style])
    bandsel = Select(title='Photoscale to', value=available[0] if available else '',
                     options=[(b, f'{b} = {mags[b]:.2f} mag') for b in available], visible=bool(available), width=240)
    centre = Slider(title='Normalisation wavelength [µm]', start=round(flo, 3), end=round(fhi, 3),
                    value=round(win[0], 3), step=0.005, width=240)
    width = Slider(title='Normalisation width [µm]', start=0.005, end=0.5, value=win[1], step=0.005, width=240)
    smooth = Slider(title='Smoothing [pixels]', start=1, end=25, value=1, step=2, width=240)
    xslider = RangeSlider(title='Wavelength range [µm]', start=round(flo, 2), end=round(fhi, 2), step=0.01,
                          value=(round(xlo, 2), round(xhi, 2)), width=240)
    yslider = RangeSlider(title='Flux range', start=ylo - (yhi - ylo), end=yhi + (yhi - ylo),
                          step=abs(yhi - ylo) / 200 or 0.01, value=(ylo, yhi), width=240,
                          format=PrintfTickFormatter(format='%.2e'))
    errtog = Toggle(label='Show uncertainties', active=True, button_type='primary', width=240)
    featsel = Select(title='Spectral features', value='None', options=['None'] + list(FEATURE_SETS), width=240)
    overlays = CheckboxGroup(labels=['Telluric bands'], active=[], width=240)
    tabs = Tabs(tabs=[TabPanel(child=figs[0], title='Linear'), TabPanel(child=figs[1], title='Log')],
                sizing_mode='stretch_width')
    note = Div(text='<small>Photoscaled spectra are scaled so that their synthetic magnitude in the chosen filter '
                    'equals the SIMPLE value; spectra marked (normalised) do not cover it, and are matched to the '
                    'photoscaled spectra in the normalisation window instead. Spectra covering less than '
                    f'{COVERAGE:.0%} of that window are never rescaled.</small>', width=240,
               visible=bool(available))
    update = CustomJS(args=dict(sources=sources, meta=meta, modes=modes, bandsel=bandsel, centre=centre, width=width,
                                smooth=smooth, figs=figs, yaxes=[f.yaxis[0] for f in figs], lines=lines, items=items,
                                marks=marks, yslider=yslider, tabs=tabs, coverage=COVERAGE), code=JS_UPDATE)
    for w, attr in ((modes, 'active'), (bandsel, 'value'), (centre, 'value'), (width, 'value'), (smooth, 'value')):
        w.js_on_change(attr, update)
    for ls in lines:
        for ln in ls:
            ln.js_on_change('visible', update)
    xslider.js_on_change('value', CustomJS(args=dict(r=figs[0].x_range),
                                           code='r.start = cb_obj.value[0]; r.end = cb_obj.value[1];'))
    yslider.js_on_change('value', CustomJS(args=dict(lin=figs[0].y_range, log=figs[1].y_range, tabs=tabs), code='''
        // the flux slider sets the range of the tab shown; the lower edge stays positive on the log scale
        let lo = cb_obj.value[0], hi = cb_obj.value[1];
        if (tabs.active == 1) {
            if (!(hi > 0)) return;
            if (!(lo > 0)) lo = hi * 1e-4;
            log.start = lo; log.end = hi;
        } else { lin.start = lo; lin.end = hi; }'''))
    # the sliders follow the plot whenever it is zoomed, panned, reset or refitted
    follow = '''
        const lo = Math.min(r.start, r.end), hi = Math.max(r.start, r.end), span = hi - lo;
        if (!(span > 0)) return;
        if (lo < s.start || hi > s.end || span < 0.02 * (s.end - s.start)) {
            s.start = lo - span; s.end = hi + span; s.step = span / 200;
        }
        if (Math.abs(s.value[0] - lo) > 1e-9 * span || Math.abs(s.value[1] - hi) > 1e-9 * span) s.value = [lo, hi];'''
    for r, sl, tab in ((figs[0].x_range, xslider, -1), (figs[0].y_range, yslider, 0), (figs[1].y_range, yslider, 1)):
        cb = CustomJS(args=dict(r=r, s=sl, tabs=tabs, tab=tab),
                      code='if (tab >= 0 && tabs.active != tab) return;' + follow)
        r.js_on_change('start', cb)
        r.js_on_change('end', cb)
    tabs.js_on_change('active', CustomJS(args=dict(rs=[figs[0].y_range, figs[1].y_range], s=yslider), code='''
        // on changing tab, the flux slider takes the range of the tab now shown
        const r = rs[cb_obj.active], lo = Math.min(r.start, r.end), hi = Math.max(r.start, r.end), span = hi - lo;
        s.start = cb_obj.active == 1 ? lo / 10 : lo - span; s.end = cb_obj.active == 1 ? hi * 10 : hi + span;
        s.step = (s.end - s.start) / 400; s.value = [lo, hi];'''))
    errtog.js_on_click(CustomJS(args=dict(bands=bands_err, lines=lines, meta=meta), code="""
        for (let i = 0; i < bands.length; i++)
            for (let k = 0; k < bands[i].length; k++)
                bands[i][k].visible = cb_obj.active && meta[i].has_err && lines[i][k].visible;"""))
    for i in range(n_spec):  # the uncertainty band follows its spectrum when hidden from the legend
        for k, ln in enumerate(lines[i]):
            ln.js_on_change('visible', CustomJS(args=dict(b=bands_err[i][k], tog=errtog, has=meta[i]['has_err']),
                                                code='b.visible = cb_obj.visible && tog.active && has;'))
    layouts = [CustomJS(args=dict(labs=feature_labels[k], sel=featsel, p=figs[k], is_log=(k == 1)),
                        code=JS_LAYOUT_LABELS) for k in range(2)]
    featsel.js_on_change('value', CustomJS(args=dict(sets=feature_sets, layouts=layouts), code="""
        for (const [cls, objs] of Object.entries(sets))
            for (const o of objs) if (o.type != 'Label') o.visible = (cls == cb_obj.value);
        for (const L of layouts) L.execute(cb_obj);"""))
    for k in range(2):
        figs[0].x_range.js_on_change('start', layouts[k])
        figs[0].x_range.js_on_change('end', layouts[k])
        figs[k].js_on_change('inner_width', layouts[k])
    overlays.js_on_change('active', CustomJS(args=dict(t=telluric_objs), code="""
        for (const o of t) o.visible = cb_obj.active.includes(0);"""))
    controls = column(modes, bandsel, centre, width, smooth, xslider, yslider, errtog, featsel, overlays, note,
                      width=260, sizing_mode='fixed')
    layout = row(controls, tabs, sizing_mode='stretch_width')
    Document().add_root(layout)
    layout.document.js_on_event('document_ready', update)  # applies the opening mode
    script, div = components(layout, theme=theme)
    return script, div, failures
