import os, math, json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import interpolate
import folium
from streamlit.components.v1 import html as st_html
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# SET PAGE CONFIG — HARUS PALING PERTAMA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FOL Leak Detection v5.1",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PSI_TO_M = 0.703070

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE REGISTRY — v5.1 (HGL-enabled)
# ─────────────────────────────────────────────────────────────────────────────

PIPELINES = {

    "RTU": {
        "label":        "RTU (Crude Oil) — 63.4 km",
        "xlsx":         "xlsx.xlsx",          # lat/lon/elev → KP haversine + peta + HGL
        "length_km":    63.4,
        "diameter_in":  7.981,
        "wall_thk_in":  0.322,
        "roughness_in": 0.001,
        "flow_rate":    5000,
        "fluid_type":   "Crude Oil",
        "sg_fluid":     0.85,                    # Condensate ~0.70-0.78 | Crude ~0.85
        "sensor_kp":      [0.0, 8.3, 15.9, 24.9, 30.8, 38.9, 44.5, 56.5, 63.0],
        "default_normal": [490.876, 446.756, 409.605, 309.338, 205.738, 191.680, 132.656, 71.525, 2.215],
        "default_drop":   [465.490, 423.330, 378.147, 293.360, 195.424, 182.974, 127.041, 69.688, 2.259],
        "historical_data": [
            {"sensor_normal": [490.876, 446.756, 409.605, 309.338, 205.738, 191.680, 132.656, 71.525, 2.215],
             "sensor_drop":   [465.490, 423.330, 378.147, 293.360, 195.424, 182.974, 127.041, 69.688, 2.259],
             "actual_leak_km": 2.1},
        ],
    },

}

# v5.1 tuning
WEIGHT_POWER = 2   # 1 = lembut (v4), 2 = metode akurat dominan

METHOD_KEYS = ['suspicion_index', 'gradient', 'region', 'interpolation',
               'weighted', 'transition', 'hgl_slope_break', 'hgl_grad_ratio']
METHOD_LABELS = {
    'suspicion_index': 'Suspicion Index ★',
    'gradient':        'Gradient',
    'region':          'Region',
    'interpolation':   'Interpolation',
    'weighted':        'Weighted Avg',
    'transition':      'Transition',
    'hgl_slope_break': 'HGL Slope-Break ⛰️',
    'hgl_grad_ratio':  'HGL Grad Ratio ⛰️',
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILS: HAVERSINE + LOAD COORDS (lat/lon/elev → KP kumulatif)
# ─────────────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


@st.cache_data
def load_coords(xlsx_filename: str) -> list:
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), xlsx_filename),
        os.path.join(os.getcwd(), xlsx_filename),
        f"/mnt/user-data/uploads/{xlsx_filename}",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return []

    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]
    lat_col  = next((c for c in df.columns if 'lat' in c), None)
    lon_col  = next((c for c in df.columns if 'lon' in c or 'lng' in c), None)
    elev_col = next((c for c in df.columns if 'alt' in c or 'elev' in c), None)
    kp_col   = next((c for c in df.columns if c in ('kp', 'km') or 'kp' in c), None)
    if elev_col is None:
        return []

    elevs = df[elev_col].values.astype(float)

    if lat_col and lon_col:
        lats = df[lat_col].values.astype(float)
        lons = df[lon_col].values.astype(float)
        coords, cum_km = [], 0.0
        for i in range(len(lats)):
            if i > 0:
                cum_km += haversine(lats[i-1], lons[i-1], lats[i], lons[i])
            coords.append({"km": round(cum_km, 3), "lat": lats[i], "lon": lons[i], "elev": elevs[i]})
        return coords
    elif kp_col:
        # Format A: kp + elev saja (tanpa lat/lon → peta nonaktif, HGL tetap jalan)
        kms = df[kp_col].values.astype(float)
        order = np.argsort(kms)
        return [{"km": float(kms[i]), "lat": None, "lon": None, "elev": float(elevs[i])} for i in order]
    return []


def get_latlon_at_km(coords: list, target_km: float):
    if not coords or coords[0]['lat'] is None:
        return 0.0, 0.0, 0.0
    if target_km <= coords[0]['km']:
        return coords[0]['lat'], coords[0]['lon'], coords[0]['elev']
    if target_km >= coords[-1]['km']:
        return coords[-1]['lat'], coords[-1]['lon'], coords[-1]['elev']
    for i in range(len(coords) - 1):
        k0, k1 = coords[i]['km'], coords[i+1]['km']
        if k0 <= target_km <= k1:
            t = (target_km - k0) / (k1 - k0) if k1 != k0 else 0.0
            return (
                coords[i]['lat']  + t * (coords[i+1]['lat']  - coords[i]['lat']),
                coords[i]['lon']  + t * (coords[i+1]['lon']  - coords[i]['lon']),
                coords[i]['elev'] + t * (coords[i+1]['elev'] - coords[i]['elev']),
            )
    return coords[-1]['lat'], coords[-1]['lon'], coords[-1]['elev']


def elev_arrays(coords):
    if not coords:
        return None, None
    return (np.array([p['km'] for p in coords], dtype=float),
            np.array([p['elev'] for p in coords], dtype=float))


def elev_at_km(elev_km, elev_m, km):
    if elev_km is None:
        return None
    return np.interp(km, elev_km, elev_m)


# ─────────────────────────────────────────────────────────────────────────────
# v5.1: DETEKSI SENSOR OUTLIER (ΔP tidak konsisten dengan tetangga)
# Rule: |residual| > 3 psi DAN |residual| > 40% dari ekspektasi.
# ─────────────────────────────────────────────────────────────────────────────

def detect_outlier_sensors(locs, norm, drop):
    dP = norm - drop
    out = []
    for i in range(1, len(locs) - 1):
        exp = np.interp(locs[i], [locs[i-1], locs[i+1]], [dP[i-1], dP[i+1]])
        res = dP[i] - exp
        if abs(res) > 3.0 and abs(res) > 0.4 * abs(exp):
            out.append(i)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZER CLASS v5.1 — 6 pressure-based + 2 HGL methods
# ─────────────────────────────────────────────────────────────────────────────

class PipelineLeakAnalyzer:
    """
    Metode 1-6 pressure-based (elevasi cancel out pada ΔP).
    Metode 7-8 HGL-based (butuh elevasi + SG):
      7. HGL Slope-Break  : fit 2 garis piecewise ke HGL drop; break = leak
      8. HGL Gradient Ratio: transisi rasio gradien drop/normal per segmen
    HGL = P(psi) × 0.70307 / SG + z(m)
    """
    PSI_TO_M = 0.703070

    def __init__(self, locations, normal_p, drop_p, elev=None, sg=0.85, calibration=None):
        self.locations   = np.array(locations, dtype=float)
        self.normal_p    = np.array(normal_p,  dtype=float)
        self.drop_p      = np.array(drop_p,    dtype=float)
        self.n_sensors   = len(self.locations)
        self.calibration = calibration
        self.sg          = float(sg)
        self.delta_p     = self.normal_p - self.drop_p
        with np.errstate(divide='ignore', invalid='ignore'):
            self.pressure_ratio = np.abs(self.delta_p) / np.abs(self.normal_p) * 100
        self.pressure_ratio = np.nan_to_num(self.pressure_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        self.abs_delta_p    = np.abs(self.delta_p)

        if elev is not None:
            self.elev     = np.array(elev, dtype=float)
            c             = self.PSI_TO_M / self.sg
            self.hgl_norm = self.normal_p * c + self.elev
            self.hgl_drop = self.drop_p   * c + self.elev
        else:
            self.elev = self.hgl_norm = self.hgl_drop = None

        self.results = {}

    def _apply_bias(self, key, raw):
        if self.calibration and key in self.calibration.get('bias', {}):
            return raw - self.calibration['bias'][key]
        return raw

    # ── Metode 1-6 ──
    def calculate_suspicion_index(self):
        si = np.zeros(self.n_sensors)
        for i in range(self.n_sensors):
            df = self.abs_delta_p[i]
            rf = self.pressure_ratio[i]
            if 0 < i < self.n_sensors - 1:
                nf = max(0.0, df - (self.abs_delta_p[i-1] + self.abs_delta_p[i+1]) / 2)
            elif i == 0:
                nf = max(0.0, df - self.abs_delta_p[i+1]) if self.n_sensors > 1 else 0.0
            else:
                nf = max(0.0, df - self.abs_delta_p[i-1])
            si[i] = df * 0.4 + rf * 0.3 + nf * 0.3
        return si

    def calculate_gradients(self):
        ng, dg, chg, locs = [], [], [], []
        for i in range(self.n_sensors - 1):
            dist = self.locations[i+1] - self.locations[i]
            if dist == 0:
                continue
            n = (self.normal_p[i+1] - self.normal_p[i]) / dist
            d = (self.drop_p[i+1]   - self.drop_p[i])   / dist
            ng.append(n); dg.append(d); chg.append(abs(n - d))
            locs.append((self.locations[i] + self.locations[i+1]) / 2)
        return {'locations': locs, 'normal': ng, 'drop': dg, 'change': chg}

    def region_analysis(self, n_regions=5):
        mn, mx = self.locations.min(), self.locations.max()
        if mn == mx:
            return [{'name': 'Region 1', 'start': mn, 'end': mx, 'center': mn,
                     'score': 0, 'avg_delta': 0, 'max_delta': 0, 'avg_ratio': 0,
                     'n_sensors': self.n_sensors}]
        rs = (mx - mn) / n_regions
        regions = []
        for i in range(n_regions):
            s, e = mn + i * rs, mn + (i+1) * rs
            mask = (self.locations >= s) & (self.locations <= e)
            if np.any(mask):
                ad = float(np.mean(self.abs_delta_p[mask]))
                ar = float(np.mean(self.pressure_ratio[mask]))
                regions.append({
                    'name': f'Region {i+1}', 'start': s, 'end': e,
                    'center': (s + e) / 2, 'score': ad * ar, 'avg_delta': ad,
                    'max_delta': float(np.max(self.abs_delta_p[mask])),
                    'avg_ratio': ar, 'n_sensors': int(np.sum(mask))
                })
        return sorted(regions, key=lambda x: x['score'], reverse=True) or \
               [{'name': 'R1', 'start': mn, 'end': mx, 'center': (mn+mx)/2,
                 'score': 0, 'avg_delta': 0, 'max_delta': 0, 'avg_ratio': 0,
                 'n_sensors': self.n_sensors}]

    def interpolate_location(self):
        if self.n_sensors < 4:
            return float(self.locations[np.argmax(self.abs_delta_p)])
        try:
            f = interpolate.interp1d(self.locations, self.abs_delta_p,
                                     kind='cubic', fill_value='extrapolate')
            x = np.linspace(self.locations.min(), self.locations.max(), 1000)
            return float(x[np.argmax(f(x))])
        except Exception:
            return float(self.locations[np.argmax(self.abs_delta_p)])

    def weighted_average_location(self, si):
        tw = float(np.sum(si))
        return float(np.mean(self.locations)) if tw == 0 else float(np.sum(si * self.locations) / tw)

    def transition_point_analysis(self):
        if self.n_sensors < 2:
            return float(self.locations[0])
        mc, tp = 0.0, float(self.locations[0])
        for i in range(self.n_sensors - 1):
            c = abs(self.abs_delta_p[i+1] - self.abs_delta_p[i])
            if c > mc:
                mc = c
                tp = (self.locations[i] + self.locations[i+1]) / 2
        return float(tp)

    # ── Metode 7 (v5): HGL Slope-Break ──
    def hgl_slope_break(self):
        if self.hgl_drop is None or self.n_sensors < 4:
            return None, None
        x, y = self.locations, self.hgl_drop
        best_sse, best = np.inf, None
        for k in range(2, self.n_sensors - 1):
            cu = np.polyfit(x[:k], y[:k], 1)
            cd = np.polyfit(x[k:], y[k:], 1)
            sse = (np.sum((np.polyval(cu, x[:k]) - y[:k])**2) +
                   np.sum((np.polyval(cd, x[k:]) - y[k:])**2))
            if sse < best_sse:
                if abs(cu[0] - cd[0]) < 1e-12:
                    xb = (x[k-1] + x[k]) / 2
                else:
                    xb = (cd[1] - cu[1]) / (cu[0] - cd[0])
                xb = float(np.clip(xb, x[0], x[-1]))
                best_sse = sse
                best = {'split_idx': k, 'coef_up': cu.tolist(),
                        'coef_dn': cd.tolist(), 'break_km': xb, 'sse': float(sse)}
        return (best['break_km'], best) if best else (None, None)

    # ── Metode 8 (v5): HGL Gradient Ratio ──
    def hgl_gradient_ratio(self):
        if self.hgl_norm is None or self.n_sensors < 3:
            return None
        x  = self.locations
        gn = np.diff(self.hgl_norm) / np.diff(x)
        gd = np.diff(self.hgl_drop) / np.diff(x)
        r  = np.where(np.abs(gn) > 1e-9, gd / gn, 1.0)
        n_seg = len(r)
        best_gain, best_b = -np.inf, None
        for b in range(1, n_seg):
            gain = float(np.mean(r[:b]) - np.mean(r[b:]))
            if gain > best_gain:
                best_gain, best_b = gain, b
        if best_b is None:
            return None
        return float(x[best_b])

    def run_full_analysis(self):
        si      = self.calculate_suspicion_index()
        top_idx = int(np.argmax(si))
        grads   = self.calculate_gradients()
        regions = self.region_analysis()

        raw = {
            'suspicion_index': float(self.locations[top_idx]),
            'gradient':        float(grads['locations'][int(np.argmax(grads['change']))]) if grads['change'] else float(np.mean(self.locations)),
            'region':          float(regions[0]['center']),
            'interpolation':   self.interpolate_location(),
            'weighted':        self.weighted_average_location(si),
            'transition':      self.transition_point_analysis(),
        }

        hgl_break_km, hgl_fit = self.hgl_slope_break()
        hgl_ratio_km          = self.hgl_gradient_ratio()
        if hgl_break_km is not None:
            raw['hgl_slope_break'] = hgl_break_km
        if hgl_ratio_km is not None:
            raw['hgl_grad_ratio'] = hgl_ratio_km

        corrected = {k: self._apply_bias(k, v) for k, v in raw.items()}
        lo, hi = self.locations.min(), self.locations.max() + 5
        corrected = {k: float(np.clip(v, lo, hi)) for k, v in corrected.items()}

        method_order = list(corrected.keys())
        if self.calibration and 'weights' in self.calibration:
            w = np.array([self.calibration['weights'].get(k, 1.0) for k in method_order])
        else:
            # default tanpa kalibrasi: HGL fisika-based diberi bobot lebih
            defw = {'hgl_slope_break': 3.0, 'hgl_grad_ratio': 2.0}
            w = np.array([defw.get(k, 1.0) for k in method_order])

        estimates = np.array([corrected[k] for k in method_order])
        self.results.update({
            'suspicion_index':        si,
            'top_sensor_idx':         top_idx,
            'top_sensor_si':          float(si[top_idx]),
            'top_sensor_location':    corrected['suspicion_index'],
            'gradient_location':      corrected['gradient'],
            'region_location':        corrected['region'],
            'interpolation_location': corrected['interpolation'],
            'weighted_location':      corrected['weighted'],
            'transition_location':    corrected['transition'],
            'hgl_break_location':     corrected.get('hgl_slope_break'),
            'hgl_ratio_location':     corrected.get('hgl_grad_ratio'),
            'hgl_fit':                hgl_fit,
            'gradients': grads, 'regions': regions, 'top_region': regions[0],
            'final_estimate':  float(np.average(estimates, weights=w)),
            'estimate_std':    float(np.std(estimates)),
            'method_weights':  dict(zip(method_order, w.tolist())),
            'method_estimates': corrected,
        })

        std = self.results['estimate_std']
        if std < 5:    conf = "HIGH (90-95%)"
        elif std < 10: conf = "HIGH (85-90%)"
        elif std < 15: conf = "MEDIUM (75-85%)"
        else:          conf = "MEDIUM (70-75%)"
        self.results['confidence'] = conf
        return self.results


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION v5.1 — bias shrinkage n/(n+2) + weight (1/MAE)^power
# historical_data di-pass sebagai tuple of JSON strings (hashable utk cache)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def build_calibration(historical_json_tuple: tuple, sensor_locs_tuple: tuple,
                      elev_tuple, sg: float, auto_exclude: bool, weight_power: int):
    if not historical_json_tuple:
        return None
    historical_data = [json.loads(s) for s in historical_json_tuple]

    sensor_locs = np.array(sensor_locs_tuple)
    if elev_tuple is not None:
        e_km = np.array(elev_tuple[0]); e_m = np.array(elev_tuple[1])
        sensor_elev_all = np.interp(sensor_locs, e_km, e_m)
    else:
        sensor_elev_all = None

    errors = {}
    def _add(k, v, actual):
        errors.setdefault(k, []).append(v - actual)

    n_valid = 0
    for rec in historical_data:
        norm_arr = np.array(rec['sensor_normal'], dtype=float)
        drop_arr = np.array(rec['sensor_drop'],   dtype=float)
        actual   = float(rec['actual_leak_km'])

        n = min(len(norm_arr), len(drop_arr), len(sensor_locs))
        mask  = ~((norm_arr[:n] == 0) & (drop_arr[:n] == 0))
        locs_ = sensor_locs[:n][mask]
        norm_ = norm_arr[:n][mask]
        drop_ = drop_arr[:n][mask]
        elev_ = sensor_elev_all[:n][mask] if sensor_elev_all is not None else None

        # outlier exclusion konsisten dengan input utama
        if auto_exclude and len(locs_) >= 4:
            out = detect_outlier_sensors(locs_, norm_, drop_)
            if out:
                keep = np.array([i not in out for i in range(len(locs_))])
                locs_, norm_, drop_ = locs_[keep], norm_[keep], drop_[keep]
                if elev_ is not None:
                    elev_ = elev_[keep]

        if len(locs_) < 2:
            continue
        n_valid += 1

        az = PipelineLeakAnalyzer(locs_, norm_, drop_, elev=elev_, sg=sg, calibration=None)
        si      = az.calculate_suspicion_index()
        grads   = az.calculate_gradients()
        regions = az.region_analysis()

        _add('suspicion_index', float(locs_[np.argmax(si)]), actual)
        _add('gradient',
             float(grads['locations'][int(np.argmax(grads['change']))]) if grads['change'] else float(np.mean(locs_)),
             actual)
        _add('region',        float(regions[0]['center']), actual)
        _add('interpolation', az.interpolate_location(),   actual)
        _add('weighted',      az.weighted_average_location(si), actual)
        _add('transition',    az.transition_point_analysis(),   actual)

        bk, _ = az.hgl_slope_break()
        if bk is not None:
            _add('hgl_slope_break', bk, actual)
        rt = az.hgl_gradient_ratio()
        if rt is not None:
            _add('hgl_grad_ratio', rt, actual)

    if not errors.get('suspicion_index'):
        return None

    method_keys = list(errors.keys())
    shrink  = n_valid / (n_valid + 2.0)   # n=1→0.33 | n=3→0.60 | n=10→0.83
    bias    = {k: float(np.mean(errors[k])) * shrink for k in method_keys}
    mae     = {k: float(np.mean(np.abs(np.array(errors[k]) - bias[k]))) for k in method_keys}
    eps     = 0.5
    wr      = {k: (1.0 / (mae[k] + eps)) ** weight_power for k in method_keys}
    tw      = sum(wr.values())
    weights = {k: wr[k] / tw * len(method_keys) for k in method_keys}
    return {'n_samples': n_valid, 'shrink': shrink, 'bias': bias, 'mae': mae, 'weights': weights}


def make_historical_json_tuple(historical_data: list) -> tuple:
    return tuple(json.dumps(d, sort_keys=True) for d in historical_data)


# ─────────────────────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────────────────────

def make_map(analyzer, results, coords, sensor_kp_all, active_mask, pipeline_name, calibration):
    fe  = results['final_estimate']
    std = results['estimate_std']
    si  = results['suspicion_index']

    if not coords or coords[0]['lat'] is None:
        return None, 0.0, 0.0, "#"

    mid = coords[len(coords)//2]
    m   = folium.Map(location=[mid['lat'], mid['lon']], zoom_start=12,
                     tiles='CartoDB dark_matter')

    folium.PolyLine([(p['lat'], p['lon']) for p in coords],
                    color='#58a6ff', weight=3, opacity=0.8,
                    tooltip=f'Pipeline {pipeline_name}').add_to(m)

    step = max(1, int(coords[-1]['km'] / 5))
    prev_mark = -99
    for p in coords:
        if p['km'] - prev_mark >= step and p['km'] > 0.5:
            folium.CircleMarker([p['lat'], p['lon']], radius=3,
                color='#8b949e', fill=True, fill_color='#8b949e', fill_opacity=0.6,
                tooltip=f"KP {p['km']:.1f} km | Elev {p['elev']:.0f} m").add_to(m)
            prev_mark = p['km']

    for i, kp in enumerate(analyzer.locations):
        lat, lon, elev = get_latlon_at_km(coords, kp)
        ratio  = float(analyzer.pressure_ratio[i])
        si_val = float(si[i])
        color  = '#f85149' if si_val == float(si.max()) else '#d29922' if ratio > 25 else '#3fb950'
        folium.CircleMarker([lat, lon], radius=10, color=color, weight=2,
            fill=True, fill_color=color, fill_opacity=0.85,
            tooltip=(f"<b>Sensor @ KP {kp:.1f} km</b><br>"
                     f"Normal: {analyzer.normal_p[i]:.2f} psi | Drop: {analyzer.drop_p[i]:.2f} psi<br>"
                     f"ΔP: {analyzer.delta_p[i]:+.2f} psi | Ratio: {ratio:.1f}%<br>"
                     f"SI: {si_val:.2f} | Elev: {elev:.0f} m")
        ).add_to(m)

    for i in range(len(sensor_kp_all)):
        if not active_mask[i]:
            lat, lon, _ = get_latlon_at_km(coords, sensor_kp_all[i])
            folium.CircleMarker([lat, lon], radius=8, color='#6e7681', weight=2,
                fill=True, fill_color='#21262d', fill_opacity=0.9,
                tooltip=f"⚠️ SENSOR OFFLINE @ KP {sensor_kp_all[i]:.1f} km").add_to(m)

    primary_pts  = [(p['lat'], p['lon']) for p in coords if (fe-10) <= p['km'] <= (fe+10)]
    critical_pts = [(p['lat'], p['lon']) for p in coords if (fe-5)  <= p['km'] <= (fe+5)]
    if len(primary_pts)  > 1:
        folium.PolyLine(primary_pts,  color='#d29922', weight=6, opacity=0.4,
                        tooltip=f'Primary Zone KP {fe-10:.1f}–{fe+10:.1f}').add_to(m)
    if len(critical_pts) > 1:
        folium.PolyLine(critical_pts, color='#f85149', weight=6, opacity=0.55,
                        tooltip=f'Critical Zone KP {fe-5:.1f}–{fe+5:.1f}').add_to(m)

    leak_lat, leak_lon, leak_elev = get_latlon_at_km(coords, fe)
    gmaps      = f"https://www.google.com/maps?q={leak_lat:.6f},{leak_lon:.6f}"
    calib_note = f"✓ {calibration['n_samples']} sampel historis" if calibration else "ℹ tanpa kalibrasi"
    hgl_note   = "⛰️ HGL aktif" if analyzer.hgl_drop is not None else "HGL nonaktif"

    folium.Marker([leak_lat, leak_lon],
        icon=folium.Icon(color='red', icon='fire', prefix='fa'),
        tooltip=f"🔴 ESTIMATED LEAK @ KP {fe:.1f} km",
        popup=folium.Popup(
            f"""<div style="font-family:monospace;min-width:240px;">
              <b style="color:#c0392b;font-size:14px;">🔴 LEAK ESTIMATE</b><br><br>
              <b>Jalur:</b> {pipeline_name}<br>
              <b>KP:</b> {fe:.2f} ± {std:.1f} km<br>
              <b>Lat:</b> {leak_lat:.6f}<br>
              <b>Lon:</b> {leak_lon:.6f}<br>
              <b>Elevasi:</b> {leak_elev:.0f} m<br>
              <b>Confidence:</b> {results['confidence']}<br>
              <span style="color:#8b949e;font-size:11px;">{calib_note} | {hgl_note}</span><br><br>
              <a href="{gmaps}" target="_blank"
                 style="background:#c0392b;color:#fff;padding:5px 10px;
                        border-radius:4px;text-decoration:none;">
                 📍 Google Maps
              </a></div>""", max_width=300)
    ).add_to(m)

    folium.Circle([leak_lat, leak_lon], radius=std * 1000,
        color='#f85149', fill=True, fill_color='#f85149', fill_opacity=0.07,
        weight=1.5, dash_array='6', tooltip=f'Uncertainty ±{std:.1f} km').add_to(m)

    legend = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:#161b22;border:1px solid #30363d;border-radius:8px;
                padding:12px 16px;font-family:monospace;font-size:12px;color:#c9d1d9;">
      <b>🛢️ {pipeline_name}</b><br><br>
      <span style="color:#58a6ff;">━━</span> Rute Pipeline<br>
      <span style="color:#f85149;">━━</span> Critical Zone (±5 km)<br>
      <span style="color:#d29922;">━━</span> Primary Zone (±10 km)<br>
      <span style="color:#f85149;">●</span> Sensor (High SI)<br>
      <span style="color:#3fb950;">●</span> Sensor (Normal)<br>
      <span style="color:#6e7681;">●</span> Sensor Offline<br>
      🔴 Leak @ KP {fe:.1f} km
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m, leak_lat, leak_lon, gmaps


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS — 6 panel dark + HGL panel terpisah
# ─────────────────────────────────────────────────────────────────────────────

BG='#0d1117'; CARD='#161b22'; RED='#f85149'; GRN='#3fb950'
BLU='#58a6ff'; PRP='#bc8cff'; YLW='#d29922'; GRID='#21262d'; BRN='#9e6a03'

def _style_ax(a):
    a.set_facecolor(CARD)
    for sp in a.spines.values(): sp.set_edgecolor(GRID)
    a.tick_params(colors='#8b949e', labelsize=8)
    a.xaxis.label.set_color('#8b949e'); a.yaxis.label.set_color('#8b949e')
    a.title.set_color('#c9d1d9'); a.grid(True, color=GRID, linewidth=0.5, alpha=0.7)


def make_plots(analyzer, results):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.subplots_adjust(hspace=0.5, wspace=0.38)
    ax = [fig.add_subplot(2, 3, i+1) for i in range(6)]
    for a in ax:
        _style_ax(a)

    fe  = results['final_estimate']
    si  = results['suspicion_index']
    thr = float(np.percentile(si, 60))

    # 1. Pressure profile
    ax[0].plot(analyzer.locations, analyzer.normal_p, 'o-', color=GRN, lw=2.5, ms=9, label='Normal')
    ax[0].plot(analyzer.locations, analyzer.drop_p,   's-', color=RED, lw=2.5, ms=9, label='Anomaly')
    ax[0].axvline(fe, color=RED, ls='--', lw=2, alpha=0.7, label=f'Est. KP {fe:.1f}')
    ax[0].set_title('Pressure Profiles', fontweight='bold')
    ax[0].set_xlabel('KP (km)'); ax[0].set_ylabel('Pressure (psi)')
    ax[0].legend(fontsize=8, framealpha=0.2)

    # 2. Delta P
    colors2 = [RED if s > thr else BLU for s in si]
    bw = max(0.5, float(np.diff(analyzer.locations).min()) * 0.5) if analyzer.n_sensors > 1 else 1.0
    ax[1].bar(analyzer.locations, analyzer.delta_p, width=bw, color=colors2, alpha=0.85, edgecolor=GRID)
    ax[1].axvline(fe, color=RED, ls='--', lw=2, alpha=0.7)
    ax[1].axhline(0, color='#8b949e', lw=1)
    ax[1].set_title('ΔP = Normal − Anomaly', fontweight='bold')
    ax[1].set_xlabel('KP (km)'); ax[1].set_ylabel('ΔP (psi)')

    # 3. Suspicion Index
    ax[2].plot(analyzer.locations, si, 'o-', color=PRP, lw=2.5, ms=9, zorder=3)
    ax[2].fill_between(analyzer.locations, 0, si, alpha=0.2, color=PRP)
    ax[2].axhline(thr, color=RED, ls='--', lw=1.5, label='Threshold')
    ax[2].axvline(fe,  color=RED, ls='--', lw=2,   alpha=0.7)
    ti = results['top_sensor_idx']
    ax[2].plot(analyzer.locations[ti], si[ti], '*', color=YLW, ms=20, zorder=5,
               label=f'Max SI={si[ti]:.2f}')
    ax[2].set_title('Suspicion Index ★', fontweight='bold')
    ax[2].set_xlabel('KP (km)'); ax[2].set_ylabel('SI')
    ax[2].legend(fontsize=8, framealpha=0.2)

    # 4. Pressure Ratio
    c_ratio = ['#b91c1c' if r > 75 else RED if r > 50 else YLW if r > 25 else GRN
               for r in analyzer.pressure_ratio]
    ax[3].bar(analyzer.locations, analyzer.pressure_ratio, width=bw, color=c_ratio, alpha=0.85, edgecolor=GRID)
    ax[3].axvline(fe, color=RED, ls='--', lw=2, alpha=0.7)
    ax[3].set_title('|ΔP| / P_normal × 100%', fontweight='bold')
    ax[3].set_xlabel('KP (km)'); ax[3].set_ylabel('Ratio (%)')
    ax[3].legend(handles=[
        mpatches.Patch(color='#b91c1c', label='>75%'),
        mpatches.Patch(color=RED,       label='50-75%'),
        mpatches.Patch(color=YLW,       label='25-50%'),
        mpatches.Patch(color=GRN,       label='<25%'),
    ], fontsize=7, framealpha=0.2)

    # 5. Gradient change
    gl = results['gradients']['locations']
    gc = results['gradients']['change']
    if gl:
        ax[4].plot(gl, gc, 'o-', color=BLU, lw=2.5, ms=9)
        ax[4].fill_between(gl, 0, gc, alpha=0.2, color=BLU)
    ax[4].axvline(fe, color=RED, ls='--', lw=2, alpha=0.7)
    ax[4].set_title('Gradient Change (psi/km)', fontweight='bold')
    ax[4].set_xlabel('KP (km)'); ax[4].set_ylabel('|ΔGradient|')

    # 6. Method comparison — 8 metode v5.1
    mes = results['method_estimates']
    mw  = results.get('method_weights', {})
    keys = list(mes.keys())
    mnames = [METHOD_LABELS.get(k, k) for k in keys]
    mvals  = [mes[k] for k in keys]
    mc = ['#b91c1c', RED, RED, BLU, YLW, YLW, BRN, BRN][:len(keys)]
    bars = ax[5].barh(mnames, mvals, color=mc, alpha=0.85, edgecolor=GRID)
    for b, k in zip(bars, keys):
        ax[5].text(b.get_width() + 0.1, b.get_y() + b.get_height() / 2,
                   f"w={mw.get(k, 1):.2f}", va='center', fontsize=7, color='#8b949e')
    ax[5].axvline(results['final_estimate'], color=GRN, ls='--', lw=2.5,
                  label=f"Final KP {results['final_estimate']:.1f}")
    ax[5].set_title('Method Comparison + Weight (8 metode)', fontweight='bold')
    ax[5].set_xlabel('KP (km)')
    ax[5].tick_params(axis='y', labelsize=7)
    ax[5].legend(fontsize=8, framealpha=0.2)

    return fig


def make_hgl_plot(analyzer, results, elev_km, elev_m, fluid_type):
    plt.style.use('dark_background')
    fig, (axE, axH) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, facecolor=BG,
                                   gridspec_kw={'height_ratios': [1, 2.2]})
    for a in (axE, axH):
        _style_ax(a)
    fe = results['final_estimate']

    # Panel 1: Elevasi
    axE.fill_between(elev_km, elev_m, elev_m.min() - 5, alpha=0.2, color=BRN)
    axE.plot(elev_km, elev_m, color=BRN, lw=1.5)
    axE.scatter(analyzer.locations, analyzer.elev, s=90, zorder=5,
                color=BLU, edgecolor='white', label='Sensor')
    axE.axvline(fe, color=RED, ls='--', lw=2, alpha=0.7)
    axE.set_ylabel('Elevasi (m)')
    axE.set_title('Profil Elevasi + Posisi Sensor', fontweight='bold')
    axE.legend(fontsize=8, framealpha=0.2)

    # Panel 2: HGL
    axH.plot(analyzer.locations, analyzer.hgl_norm, 'o-', color=GRN, lw=2.5, ms=9, label='HGL Normal')
    axH.plot(analyzer.locations, analyzer.hgl_drop, 's-', color=RED, lw=2.5, ms=9, label='HGL Drop')

    hf = results.get('hgl_fit')
    if hf:
        xb = hf['break_km']
        xu = np.linspace(analyzer.locations[0],  xb, 50)
        xd = np.linspace(xb, analyzer.locations[-1], 50)
        axH.plot(xu, np.polyval(hf['coef_up'], xu), '--', color=YLW, lw=2,
                 label=f"Fit upstream ({hf['coef_up'][0]:.2f} m/km)")
        axH.plot(xd, np.polyval(hf['coef_dn'], xd), '--', color=PRP, lw=2,
                 label=f"Fit downstream ({hf['coef_dn'][0]:.2f} m/km)")
        axH.plot(xb, np.polyval(hf['coef_up'], xb), '*', color='gold', ms=24, zorder=6,
                 markeredgecolor='black', label=f"Slope-break @ KP {xb:.1f}")

    axH.axvline(fe, color=RED, ls='--', lw=2.5, alpha=0.7, label=f'Final KP {fe:.1f}')
    if results.get('hgl_ratio_location') is not None:
        axH.axvline(results['hgl_ratio_location'], color='#2dd4bf', ls=':', lw=2,
                    alpha=0.9, label=f"HGL Ratio @ KP {results['hgl_ratio_location']:.1f}")
    axH.set_xlabel('KP (km)'); axH.set_ylabel('HGL (m) = P×0.703/SG + z')
    axH.set_title(f'Hydraulic Grade Line — SG={analyzer.sg} ({fluid_type}) | Leak = titik patah slope',
                  fontweight='bold')
    axH.legend(fontsize=8, ncol=2, framealpha=0.2)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Barlow',sans-serif;}
.stApp{background:#0d1117;color:#e6edf3;}
section[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #21262d;}
section[data-testid="stSidebar"] *{color:#c9d1d9 !important;}
.header-banner{background:linear-gradient(135deg,#0d1117 0%,#1c2128 50%,#0d1117 100%);
  border:1px solid #30363d;border-left:4px solid #f85149;padding:1.2rem 1.8rem;
  border-radius:6px;margin-bottom:1.5rem;}
.header-banner h1{font-family:'Barlow',sans-serif;font-weight:800;font-size:1.5rem;
  color:#f0f6fc;margin:0;}
.header-banner p{color:#8b949e;margin:0.2rem 0 0 0;font-size:0.82rem;
  font-family:'Share Tech Mono',monospace;}
.metric-card{background:#161b22;border:1px solid #30363d;border-radius:6px;
  padding:0.9rem 1.1rem;text-align:center;}
.metric-card .label{font-family:'Share Tech Mono',monospace;font-size:0.68rem;
  color:#8b949e;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem;}
.metric-card .value{font-family:'Barlow',sans-serif;font-size:1.7rem;
  font-weight:800;color:#f0f6fc;line-height:1;}
.metric-card .sub{font-family:'Share Tech Mono',monospace;font-size:0.68rem;
  color:#8b949e;margin-top:0.2rem;}
.result-box{background:#161b22;border:2px solid #f85149;border-radius:8px;
  padding:1.4rem 2rem;text-align:center;margin:1rem 0;}
.result-box .kp-label{font-family:'Share Tech Mono',monospace;font-size:0.78rem;
  color:#f85149;text-transform:uppercase;letter-spacing:0.15em;}
.result-box .kp-value{font-family:'Barlow',sans-serif;font-size:3.2rem;
  font-weight:800;color:#f0f6fc;line-height:1;margin:0.2rem 0;}
.result-box .kp-std{font-family:'Share Tech Mono',monospace;font-size:0.95rem;color:#8b949e;}
.calib-box{background:rgba(63,185,80,0.08);border:1px solid #3fb950;border-radius:6px;
  padding:0.7rem 1rem;margin:0.4rem 0;font-family:'Share Tech Mono',monospace;
  font-size:0.78rem;color:#3fb950;}
.warn-box{background:rgba(210,153,34,0.1);border:1px solid #d29922;border-radius:6px;
  padding:0.7rem 1rem;margin:0.4rem 0;font-family:'Share Tech Mono',monospace;
  font-size:0.78rem;color:#d29922;}
.dead-box{background:rgba(248,81,73,0.08);border:1px solid #f85149;border-radius:6px;
  padding:0.7rem 1rem;margin:0.4rem 0;font-family:'Share Tech Mono',monospace;
  font-size:0.78rem;color:#f85149;}
.info-box{background:rgba(88,166,255,0.08);border:1px solid #58a6ff;border-radius:6px;
  padding:0.7rem 1rem;margin:0.4rem 0;font-family:'Share Tech Mono',monospace;
  font-size:0.78rem;color:#58a6ff;}
.hgl-box{background:rgba(158,106,3,0.12);border:1px solid #9e6a03;border-radius:6px;
  padding:0.7rem 1rem;margin:0.4rem 0;font-family:'Share Tech Mono',monospace;
  font-size:0.78rem;color:#d29922;}
.sec-header{font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#8b949e;
  text-transform:uppercase;letter-spacing:0.15em;border-bottom:1px solid #21262d;
  padding-bottom:0.3rem;margin:1.1rem 0 0.7rem 0;}
.stButton>button{background:#f85149 !important;color:#fff !important;border:none !important;
  border-radius:6px !important;font-family:'Barlow',sans-serif !important;font-weight:700 !important;
  font-size:1rem !important;padding:0.6rem 2rem !important;width:100% !important;}
.stButton>button:hover{background:#da3633 !important;}
.stNumberInput input{background:#0d1117 !important;border:1px solid #30363d !important;
  color:#e6edf3 !important;border-radius:4px !important;
  font-family:'Share Tech Mono',monospace !important;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-banner">
  <h1>🛢️ FOL Pipeline Leak Detection System v5.1</h1>
  <p>PRESSURE + HGL ELEVATION-CORRECTED · 8 METHODS · MAE-WEIGHTED CALIBRATION · PT PERTAMINA EP JAMBI FIELD</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🗺️ Pilih Jalur Pipeline")
    selected_name = st.selectbox(
        "Jalur",
        options=list(PIPELINES.keys()),
        format_func=lambda k: PIPELINES[k]['label'],
        label_visibility="collapsed"
    )

    cfg = PIPELINES[selected_name]
    st.markdown("---")

    st.markdown("### ⚙️ Parameter Jalur")
    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                color:#8b949e;line-height:2.0;">
    Panjang &nbsp;&nbsp;&nbsp;: <b style="color:#c9d1d9;">{cfg['length_km']} km</b><br>
    Diameter&nbsp;&nbsp;: <b style="color:#c9d1d9;">{cfg['diameter_in']}" ID</b><br>
    Wall Thk &nbsp;: <b style="color:#c9d1d9;">{cfg['wall_thk_in']}"</b><br>
    Roughness : <b style="color:#c9d1d9;">{cfg['roughness_in']}"</b><br>
    Flow Rate : <b style="color:#c9d1d9;">{cfg['flow_rate']} bbl/day</b><br>
    Fluid&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b style="color:#c9d1d9;">{cfg['fluid_type']}</b><br>
    SG Fluida : <b style="color:#c9d1d9;">{cfg['sg_fluid']}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧪 Opsi v5.1")
    auto_exclude = st.toggle("Auto-exclude sensor outlier (ΔP)", value=True,
                             help="Sensor interior dengan ΔP menyimpang >3 psi & >40% dari interpolasi tetangga dikeluarkan otomatis")

    # ── Load coords + elevasi ──
    coords = load_coords(cfg['xlsx'])
    elev_km, elev_m = elev_arrays(coords)
    hgl_available = elev_km is not None

    if hgl_available:
        st.markdown(f"""
        <div class="hgl-box">
        ⛰️ HGL aktif (SG={cfg['sg_fluid']})<br>
        {len(coords)} titik elevasi<br>
        Metode 7-8 ikut voting
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warn-box">
        ⚠ {cfg['xlsx']} tidak ditemukan<br>
        Metode HGL (7-8) nonaktif<br>
        → fallback 6 metode
        </div>""", unsafe_allow_html=True)

    # ── Build calibration v5.1 ──
    hist_json_tuple = make_historical_json_tuple(cfg['historical_data'])
    elev_tuple = (tuple(elev_km.tolist()), tuple(elev_m.tolist())) if hgl_available else None
    calib = build_calibration(hist_json_tuple, tuple(cfg['sensor_kp']),
                              elev_tuple, cfg['sg_fluid'], auto_exclude, WEIGHT_POWER)

    st.markdown("---")
    if calib:
        warn_low = "<br>⚠ sampel &lt; 3 — kalibrasi lemah" if calib['n_samples'] < 3 else ""
        st.markdown(f"""
        <div class="calib-box">
        ✓ Kalibrasi v5.1 aktif<br>
        {calib['n_samples']} sampel | shrink ×{calib['shrink']:.2f}<br>
        Weight power = {WEIGHT_POWER}<br>
        SI MAE = {calib['mae']['suspicion_index']:.1f} km{warn_low}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warn-box">
        ⚠ Belum ada data historis<br>
        → default weight:<br>
        HGL break 3× | HGL ratio 2×
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
                color:#8b949e;line-height:1.8;">
    💡 Sensor MATI?<br>Isi Normal P = 0<br>dan Drop P = 0
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — SENSOR INPUT
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f'<div class="sec-header">📡 Input Tekanan Sensor — {cfg["label"]}</div>',
            unsafe_allow_html=True)

n_sensors = st.number_input(
    "Jumlah sensor aktif di jalur ini",
    min_value=2, max_value=15,
    value=len(cfg['sensor_kp']),
    step=1
)

if not coords:
    st.markdown(f'<div class="warn-box">⚠️ File {cfg["xlsx"]} tidak ditemukan — '
                f'letakkan di folder yang sama dengan app ini. Peta & HGL nonaktif.</div>',
                unsafe_allow_html=True)
else:
    total_km = coords[-1]['km']
    st.markdown(f'<div class="info-box">📏 Panjang jalur terdeteksi dari xlsx: '
                f'<b>{total_km:.2f} km</b> ({len(coords)} titik koordinat) '
                f'| Elevasi {elev_m.min():.0f}–{elev_m.max():.0f} m '
                f'(≈{(elev_m.max()-elev_m.min())*cfg["sg_fluid"]/PSI_TO_M:.1f} psi distorsi)</div>',
                unsafe_allow_html=True)

ch = st.columns([1, 2, 2, 2, 1])
ch[0].markdown("**Sensor**"); ch[1].markdown("**KP (km)**")
ch[2].markdown("**Normal P (psi)**"); ch[3].markdown("**Drop P (psi)**")
ch[4].markdown("**Status**")

sensor_kp, sensor_normal, sensor_drop = [], [], []

for i in range(n_sensors):
    kp_def = cfg['sensor_kp'][i]       if i < len(cfg['sensor_kp'])      else float(i * 5)
    np_def = cfg['default_normal'][i]  if i < len(cfg['default_normal']) else 100.0
    dp_def = cfg['default_drop'][i]    if i < len(cfg['default_drop'])   else 98.0

    cols = st.columns([1, 2, 2, 2, 1])
    with cols[0]:
        st.markdown(f"<div style='padding-top:0.5rem;font-family:Share Tech Mono,monospace;"
                    f"font-size:0.8rem;color:#8b949e;'>S{i+1}</div>", unsafe_allow_html=True)
    with cols[1]:
        kp = st.number_input(f"kp_{selected_name}_{i}", value=kp_def, step=0.1,
                             format="%.1f", label_visibility="collapsed",
                             key=f"kp_{selected_name}_{i}")
    with cols[2]:
        np_val = st.number_input(f"np_{selected_name}_{i}", value=np_def, step=0.01,
                                 format="%.3f", label_visibility="collapsed",
                                 key=f"np_{selected_name}_{i}")
    with cols[3]:
        dp_val = st.number_input(f"dp_{selected_name}_{i}", value=dp_def, step=0.01,
                                 format="%.3f", label_visibility="collapsed",
                                 key=f"dp_{selected_name}_{i}")
    with cols[4]:
        icon = '🔴' if (np_val == 0.0 and dp_val == 0.0) else '🟢'
        st.markdown(f"<div style='padding-top:0.5rem;font-size:1.1rem;'>{icon}</div>",
                    unsafe_allow_html=True)

    sensor_kp.append(kp); sensor_normal.append(np_val); sensor_drop.append(dp_val)

st.markdown("")
run_btn = st.button("🔍 RUN ANALYSIS")

if run_btn:
    kp_arr   = np.array(sensor_kp)
    norm_arr = np.array(sensor_normal)
    drop_arr = np.array(sensor_drop)

    # ── Dead sensor filter ──
    active_mask = ~((norm_arr == 0) & (drop_arr == 0))
    dead_idx    = np.where(~active_mask)[0]
    active_locs = kp_arr[active_mask]
    active_norm = norm_arr[active_mask]
    active_drop = drop_arr[active_mask]
    n_active = int(np.sum(active_mask))
    n_dead   = n_sensors - n_active

    if n_dead > 0:
        dead_txt = " | ".join([f"S{i+1} @ KP {kp_arr[i]:.1f} km" for i in dead_idx])
        st.markdown(f'<div class="dead-box">🔴 SENSOR OFFLINE: {dead_txt}</div>',
                    unsafe_allow_html=True)

    if n_active < 2:
        st.error(f"❌ Minimal 2 sensor aktif! Saat ini hanya {n_active}.")
        st.stop()

    if len(active_locs) > 1 and not np.all(np.diff(active_locs) > 0):
        st.error("❌ KP sensor harus ascending (urut naik)!")
        st.stop()

    # ── v5.1: Outlier detection ──
    out_idx = detect_outlier_sensors(active_locs, active_norm, active_drop)
    if out_idx:
        dP_a = active_norm - active_drop
        rows = []
        for i in out_idx:
            exp = np.interp(active_locs[i],
                            [active_locs[i-1], active_locs[i+1]],
                            [dP_a[i-1], dP_a[i+1]])
            rows.append(f"KP {active_locs[i]:.1f}: ΔP={dP_a[i]:.2f} vs ekspektasi ≈{exp:.2f} psi "
                        f"(residual {dP_a[i]-exp:+.2f})")
        action = "→ AUTO-EXCLUDE aktif, sensor dikeluarkan" if auto_exclude \
                 else "→ AUTO-EXCLUDE nonaktif, tetap dipakai (hasil bisa bias)"
        st.markdown(f'<div class="warn-box">⚠ SENSOR OUTLIER (ΔP tidak konsisten dgn tetangga):<br>'
                    + "<br>".join(rows) +
                    f'<br>Kemungkinan: drift instrumen / snapshot beda waktu / leak tepat di sensor — cek fisik!<br>'
                    f'<b>{action}</b></div>', unsafe_allow_html=True)

    if out_idx and auto_exclude:
        keep = np.array([i not in out_idx for i in range(len(active_locs))])
        active_locs = active_locs[keep]
        active_norm = active_norm[keep]
        active_drop = active_drop[keep]
        n_active = len(active_locs)
        if n_active < 2:
            st.error("❌ Terlalu banyak sensor dikeluarkan!")
            st.stop()

    max_gap = float(np.max(np.diff(active_locs))) if len(active_locs) > 1 else 0.0
    if max_gap > 12:
        st.markdown(f'<div class="warn-box">⚠️ Gap sensor max = {max_gap:.1f} km '
                    f'→ akurasi di zona tersebut lebih rendah</div>', unsafe_allow_html=True)

    if calib:
        st.markdown(
            f'<div class="calib-box">✓ Kalibrasi historis v5.1 aktif — '
            f'{calib["n_samples"]} sampel | shrink ×{calib["shrink"]:.2f} | '
            f'SI MAE = {calib["mae"]["suspicion_index"]:.1f} km</div>',
            unsafe_allow_html=True)

    # ── Elevasi sensor aktif → HGL ──
    active_elev = elev_at_km(elev_km, elev_m, active_locs) if hgl_available else None

    # ── RUN v5.1 ──
    analyzer = PipelineLeakAnalyzer(active_locs, active_norm, active_drop,
                                    elev=active_elev, sg=cfg['sg_fluid'],
                                    calibration=calib)
    results  = analyzer.run_full_analysis()
    fe   = results['final_estimate']
    std  = results['estimate_std']
    conf = results['confidence']
    si   = results['suspicion_index']
    mes  = results['method_estimates']
    mw   = results['method_weights']
    n_methods = len(mes)

    # ── Metric cards ──
    st.markdown('<div class="sec-header">📊 Hasil Analisis</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
          <div class="label">Sensor Aktif</div>
          <div class="value">{n_active}<span style="font-size:1rem;color:#8b949e;">/{n_sensors}</span></div>
          <div class="sub">{n_dead} offline{f' | {len(out_idx)} outlier' if out_idx and auto_exclude else ''}</div></div>""",
          unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
          <div class="label">Metode Voting</div>
          <div class="value">{n_methods}</div>
          <div class="sub">{'6 ΔP + 2 HGL ⛰️' if n_methods == 8 else 'pressure-based only'}</div></div>""",
          unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
          <div class="label">Peak Suspicion Index</div>
          <div class="value" style="color:#f85149;">{results['top_sensor_si']:.2f}</div>
          <div class="sub">KP {results['top_sensor_location']:.1f} km</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
          <div class="label">Confidence</div>
          <div class="value" style="font-size:1rem;">{conf}</div>
          <div class="sub">std = {std:.2f} km</div></div>""", unsafe_allow_html=True)

    # ── Final estimate ──
    flags = []
    if calib:
        flags.append("<span style='font-size:0.8rem;color:#3fb950;'>✓ calibrated</span>")
    if analyzer.hgl_drop is not None:
        flags.append("<span style='font-size:0.8rem;color:#d29922;'>⛰️ HGL</span>")
    flag_html = " " + " ".join(flags) if flags else ""
    st.markdown(f"""
    <div class="result-box">
      <div class="kp-label">🎯 Estimasi Lokasi Kebocoran / Pengambilan Ilegal{flag_html}</div>
      <div class="kp-value">KP {fe:.1f}</div>
      <div class="kp-std">± {std:.1f} km &nbsp;|&nbsp; Focus: KP {max(0,fe-3):.1f} – {fe+3:.1f}</div>
    </div>""", unsafe_allow_html=True)

    # ── Method table (8 metode) ──
    st.markdown('<div class="sec-header">🔢 Perbandingan Metode (v5.1 — 8 Metode)</div>',
                unsafe_allow_html=True)
    keys = list(mes.keys())
    mdf = pd.DataFrame({
        'Method':        [METHOD_LABELS.get(k, k) for k in keys],
        'Est. KP (km)':  [round(mes[k], 2) for k in keys],
        'Weight':        [f"{mw.get(k, 1):.3f}" for k in keys],
        'Bias(km)':      [f"{calib['bias'][k]:+.2f}" if calib and k in calib['bias'] else '-' for k in keys],
        'MAE(km)':       [f"{calib['mae'][k]:.2f}"  if calib and k in calib['mae']  else '-' for k in keys],
    })
    st.dataframe(mdf, use_container_width=True, hide_index=True)

    # ── Sensor detail ──
    st.markdown('<div class="sec-header">📋 Detail Sensor Aktif</div>', unsafe_allow_html=True)
    ddf = pd.DataFrame({
        'KP (km)':        [f"{l:.1f}" for l in analyzer.locations],
        'Normal P (psi)': [f"{p:.3f}" for p in analyzer.normal_p],
        'Drop P (psi)':   [f"{p:.3f}" for p in analyzer.drop_p],
        'ΔP (psi)':       [f"{d:+.3f}" for d in analyzer.delta_p],
        '|ΔP| (psi)':     [f"{d:.3f}" for d in analyzer.abs_delta_p],
        'Ratio (%)':      [f"{r:.2f}" for r in analyzer.pressure_ratio],
        'SI':             [f"{s:.2f}" for s in si],
    })
    if analyzer.elev is not None:
        ddf.insert(1, 'Elev (m)', [f"{e:.1f}" for e in analyzer.elev])
        c = PSI_TO_M / analyzer.sg
        ddf['HGL Norm (m)'] = [f"{v:.1f}" for v in analyzer.hgl_norm]
        ddf['HGL Drop (m)'] = [f"{v:.1f}" for v in analyzer.hgl_drop]
    st.dataframe(ddf.sort_values('SI', ascending=False), use_container_width=True, hide_index=True)

    # ── Map ──
    st.markdown('<div class="sec-header">🗺️ Peta Pipeline & Estimasi Lokasi Kebocoran</div>',
                unsafe_allow_html=True)

    fol_map, leak_lat, leak_lon, gmaps = make_map(
        analyzer, results, coords, kp_arr, active_mask, selected_name, calib)

    if fol_map:
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-left:4px solid #f85149;
                    border-radius:6px;padding:1rem 1.2rem;margin-bottom:0.8rem;
                    font-family:'Share Tech Mono',monospace;">
          <span style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">
            📍 Koordinat Estimasi Kebocoran — {selected_name}
          </span><br>
          <span style="color:#f0f6fc;font-size:1.05rem;font-weight:700;">
            KP {fe:.2f} km &nbsp;|&nbsp; {leak_lat:.6f}, {leak_lon:.6f}
          </span><br>
          <a href="{gmaps}" target="_blank"
             style="display:inline-block;margin-top:0.6rem;background:#f85149;color:#fff;
                    padding:0.4rem 1rem;border-radius:5px;text-decoration:none;
                    font-size:0.82rem;font-weight:700;">
            🗺️ BUKA DI GOOGLE MAPS
          </a>
          <span style="color:#8b949e;font-size:0.72rem;"> atau klik marker 🔴 di peta</span>
        </div>""", unsafe_allow_html=True)
        st_html(fol_map._repr_html_(), height=520)
    else:
        st.warning("Peta tidak tersedia — file koordinat xlsx tidak ditemukan / tanpa lat-lon.")

    # ── Inspection zones ──
    st.markdown('<div class="sec-header">🚨 Zona Inspeksi</div>', unsafe_allow_html=True)
    z1, z2, z3 = st.columns(3)
    with z1:
        st.markdown(f"""<div class="metric-card" style="border-color:#3fb950;">
          <div class="label" style="color:#3fb950;">Primary Zone</div>
          <div class="value" style="font-size:1.1rem;">KP {max(0,fe-10):.1f} – {fe+10:.1f}</div>
          <div class="sub">20 km coverage</div></div>""", unsafe_allow_html=True)
    with z2:
        st.markdown(f"""<div class="metric-card" style="border-color:#d29922;">
          <div class="label" style="color:#d29922;">Critical Zone</div>
          <div class="value" style="font-size:1.1rem;">KP {max(0,fe-5):.1f} – {fe+5:.1f}</div>
          <div class="sub">10 km coverage</div></div>""", unsafe_allow_html=True)
    with z3:
        st.markdown(f"""<div class="metric-card" style="border-color:#f85149;">
          <div class="label" style="color:#f85149;">Highest Priority</div>
          <div class="value" style="font-size:1.1rem;">KP {max(0,fe-3):.1f} – {fe+3:.1f}</div>
          <div class="sub">6 km focus area</div></div>""", unsafe_allow_html=True)

    # ── Charts ──
    st.markdown('<div class="sec-header">📈 Visualisasi Analisis</div>', unsafe_allow_html=True)
    fig = make_plots(analyzer, results)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── HGL viz + tabel gradien per segmen ──
    if analyzer.hgl_drop is not None:
        st.markdown('<div class="sec-header">⛰️ HGL — Hydraulic Grade Line (v5)</div>',
                    unsafe_allow_html=True)
        figH = make_hgl_plot(analyzer, results, elev_km, elev_m, cfg['fluid_type'])
        st.pyplot(figH, use_container_width=True)
        plt.close(figH)

        gn = np.diff(analyzer.hgl_norm) / np.diff(analyzer.locations)
        gd = np.diff(analyzer.hgl_drop) / np.diff(analyzer.locations)
        r  = np.where(np.abs(gn) > 1e-9, gd / gn, 1.0)
        seg_df = pd.DataFrame({
            'Segment': [f"KP {analyzer.locations[i]:.1f}–{analyzer.locations[i+1]:.1f}"
                        for i in range(analyzer.n_sensors - 1)],
            'Grad Normal (m/km)': [f"{v:.3f}" for v in gn],
            'Grad Drop (m/km)':   [f"{v:.3f}" for v in gd],
            'Ratio d/n':          [f"{v:.3f}" for v in r],
            'Indikasi': ['⬇ flow turun (downstream leak?)' if v < 0.9
                         else '⬆ flow naik (upstream leak?)' if v > 1.1
                         else '— normal' for v in r],
        })
        st.dataframe(seg_df, use_container_width=True, hide_index=True)
        st.markdown('<div class="info-box">Ratio &lt; 1 → flow berkurang → segmen downstream dari leak. '
                    'HGL = P×0.703/SG + z; leak = titik patah slope.</div>', unsafe_allow_html=True)

    # ── Export CSV ──
    st.markdown('<div class="sec-header">💾 Export</div>', unsafe_allow_html=True)
    exp_df = pd.DataFrame({
        'KP (km)':         analyzer.locations,
        'Normal P (psi)':  analyzer.normal_p,
        'Drop P (psi)':    analyzer.drop_p,
        'Delta P (psi)':   analyzer.delta_p,
        '|Delta P| (psi)': analyzer.abs_delta_p,
        'Ratio (%)':       analyzer.pressure_ratio,
        'Suspicion Index': si,
    })
    if analyzer.elev is not None:
        exp_df['Elev (m)']     = analyzer.elev
        exp_df['HGL Norm (m)'] = analyzer.hgl_norm
        exp_df['HGL Drop (m)'] = analyzer.hgl_drop
    summary = pd.DataFrame([{
        'KP (km)': f'FINAL ESTIMATE: KP {fe:.2f} ± {std:.2f} km | Jalur: {selected_name} | {n_methods} metode',
        'Normal P (psi)': '', 'Drop P (psi)': '', 'Delta P (psi)': '',
        '|Delta P| (psi)': '',
        'Ratio (%)': f'Lat:{leak_lat:.6f} Lon:{leak_lon:.6f}',
        'Suspicion Index': conf,
    }])
    csv_out = pd.concat([exp_df, summary]).to_csv(index=False)
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_out,
        file_name=f"leak_v51_{selected_name.replace(' ','_').replace('→','to')}.csv",
        mime="text/csv"
    )

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#8b949e;
                font-family:Share Tech Mono,monospace;">
      <div style="font-size:3rem;margin-bottom:1rem;">🛢️</div>
      <div>Isi nilai tekanan sensor → klik <b>RUN ANALYSIS</b></div>
      <div style="font-size:0.78rem;margin-top:0.5rem;">
        v5.1: 6 metode ΔP + 2 metode HGL (elevasi) · outlier auto-exclude · bias shrinkage
      </div>
      <div style="font-size:0.78rem;margin-top:0.3rem;">
        Sensor mati? Isi Normal P = 0 dan Drop P = 0
      </div>
    </div>""", unsafe_allow_html=True)