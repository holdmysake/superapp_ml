import os
import math
import json
import numpy as np
import pandas as pd
from scipy import interpolate
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────────────
# UTILS: HAVERSINE + LOAD COORDS (lat/lon/elev → KP kumulatif)
# ─────────────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


@lru_cache(maxsize=32)
def load_coords(xlsx_filename: str) -> list:
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), xlsx_filename),
        os.path.join(os.getcwd(), xlsx_filename),
        f"/mnt/user-data/uploads/{xlsx_filename}",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        if os.path.isabs(xlsx_filename) and os.path.exists(xlsx_filename):
            path = xlsx_filename
        else:
            return []

    try:
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
            kms = df[kp_col].values.astype(float)
            order = np.argsort(kms)
            return [{"km": float(kms[i]), "lat": None, "lon": None, "elev": float(elevs[i])} for i in order]
    except Exception as e:
        print(f"Error loading coordinates from {path}: {e}")
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


def detect_outlier_sensors(locs, norm, drop):
    locs = np.array(locs)
    norm = np.array(norm)
    drop = np.array(drop)
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
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=32)
def build_calibration(historical_json_tuple: tuple, sensor_locs_tuple: tuple,
                      elev_tuple=None, sg: float=0.85, auto_exclude: bool=True, weight_power: int=2):
    if not historical_json_tuple:
        return None
    historical_data = [json.loads(s) for s in historical_json_tuple]

    sensor_locs = np.array(sensor_locs_tuple)
    if elev_tuple is not None:
        e_km = np.array(elev_tuple[0])
        e_m = np.array(elev_tuple[1])
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
    shrink  = n_valid / (n_valid + 2.0)
    bias    = {k: float(np.mean(errors[k])) * shrink for k in method_keys}
    mae     = {k: float(np.mean(np.abs(np.array(errors[k]) - bias[k]))) for k in method_keys}
    eps     = 0.5
    wr      = {k: (1.0 / (mae[k] + eps)) ** weight_power for k in method_keys}
    tw      = sum(wr.values())
    weights = {k: wr[k] / tw * len(method_keys) for k in method_keys}
    return {'n_samples': n_valid, 'shrink': shrink, 'bias': bias, 'mae': mae, 'weights': weights}
