import os
import math
import json
import numpy as np
import pandas as pd
from scipy import interpolate
from functools import lru_cache


# ─────────────────────────────────────────────────────────────────────────────
# UTILS: HAVERSINE + LOAD COORDS
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
        return []

    try:
        df = pd.read_excel(path)
        df.columns = [c.strip().lower() for c in df.columns]
        lat_col  = next(c for c in df.columns if 'lat' in c)
        lon_col  = next(c for c in df.columns if 'lon' in c)
        elev_col = next(c for c in df.columns if 'alt' in c or 'elev' in c)

        lats  = df[lat_col].values.astype(float)
        lons  = df[lon_col].values.astype(float)
        elevs = df[elev_col].values.astype(float)

        coords, cum_km = [], 0.0
        for i in range(len(lats)):
            if i > 0:
                cum_km += haversine(lats[i-1], lons[i-1], lats[i], lons[i])
            coords.append({"km": round(cum_km, 3), "lat": lats[i], "lon": lons[i], "elev": elevs[i]})
        return coords
    except Exception as e:
        print(f"Error loading coordinates from {path}: {e}")
        return []


def get_latlon_at_km(coords: list, target_km: float):
    if not coords:
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


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_historical_data(file_or_list) -> list:
    if isinstance(file_or_list, str):
        if not os.path.isabs(file_or_list):
            candidates = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), file_or_list),
                os.path.join(os.getcwd(), file_or_list),
            ]
            path = next((p for p in candidates if os.path.exists(p)), None)
        else:
            path = file_or_list if os.path.exists(file_or_list) else None

        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading historical data from {path}: {e}")
                return []
        return []
    elif isinstance(file_or_list, list):
        return file_or_list
    return []


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=32)
def build_calibration(historical_json_tuple: tuple, sensor_locs_tuple: tuple):
    """
    historical_json_tuple : tuple of JSON strings — tiap string = 1 record dict.
    sensor_locs_tuple     : tuple of float KP locations.
    """
    if not historical_json_tuple:
        return None
    historical_data = [json.loads(s) for s in historical_json_tuple]

    sensor_locs = np.array(sensor_locs_tuple)
    method_keys = ['suspicion_index', 'gradient', 'region', 'interpolation', 'weighted', 'transition']
    errors = {k: [] for k in method_keys}

    for rec in historical_data:
        norm_arr = np.array(rec['sensor_normal'], dtype=float)
        drop_arr = np.array(rec['sensor_drop'],   dtype=float)
        actual   = float(rec['actual_leak_km'])
        n = min(len(norm_arr), len(drop_arr), len(sensor_locs))
        locs_ = sensor_locs[:n]
        norm_ = norm_arr[:n]
        drop_ = drop_arr[:n]
        mask  = ~((norm_ == 0) & (drop_ == 0))
        locs_ = locs_[mask]
        norm_ = norm_[mask]
        drop_ = drop_[mask]
        if len(locs_) < 2:
            continue

        az = PipelineLeakAnalyzer(locs_, norm_, drop_, calibration=None)
        si = az.calculate_suspicion_index()
        grads   = az.calculate_gradients()
        regions = az.region_analysis()

        pred = {
            'suspicion_index': float(locs_[np.argmax(si)]),
            'gradient':        float(grads['locations'][int(np.argmax(grads['change']))]) if grads['change'] else float(np.mean(locs_)),
            'region':          float(regions[0]['center']) if regions else float(np.mean(locs_)),
            'interpolation':   az.interpolate_location(),
            'weighted':        az.weighted_average_location(si),
            'transition':      az.transition_point_analysis(),
        }
        for k in method_keys:
            errors[k].append(pred[k] - actual)

    if not errors['suspicion_index']:
        return None

    bias    = {k: float(np.mean(errors[k])) for k in method_keys}
    mae     = {k: float(np.mean(np.abs(errors[k]))) for k in method_keys}
    eps     = 0.5
    wr      = {k: 1.0 / (mae[k] + eps) for k in method_keys}
    tw      = sum(wr.values())
    weights = {k: wr[k] / tw * len(method_keys) for k in method_keys}
    return {'n_samples': len(historical_data), 'bias': bias, 'mae': mae, 'weights': weights}


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class PipelineLeakAnalyzer:
    def __init__(self, locations, normal_p, drop_p, calibration=None):
        self.locations   = np.array(locations, dtype=float)
        self.normal_p    = np.array(normal_p,  dtype=float)
        self.drop_p      = np.array(drop_p,    dtype=float)
        self.n_sensors   = len(self.locations)
        self.calibration = calibration
        self.delta_p     = self.normal_p - self.drop_p
        with np.errstate(divide='ignore', invalid='ignore'):
            self.pressure_ratio = np.abs(self.delta_p) / np.abs(self.normal_p) * 100
        self.pressure_ratio = np.nan_to_num(self.pressure_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        self.abs_delta_p    = np.abs(self.delta_p)
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
        corrected = {k: self._apply_bias(k, v) for k, v in raw.items()}
        pipe_max  = self.locations.max() + 5
        corrected = {k: float(np.clip(v, 0, pipe_max)) for k, v in corrected.items()}

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
            'gradients': grads, 'regions': regions, 'top_region': regions[0],
        })

        method_order = ['suspicion_index', 'gradient', 'region', 'interpolation', 'weighted', 'transition']
        estimates    = np.array([corrected[k] for k in method_order])
        if self.calibration and 'weights' in self.calibration:
            w = np.array([self.calibration['weights'].get(k, 1.0) for k in method_order])
        else:
            w = np.ones(len(method_order))

        self.results['final_estimate']  = float(np.average(estimates, weights=w))
        self.results['estimate_std']    = float(np.std(estimates))
        self.results['method_weights']  = dict(zip(method_order, w.tolist()))

        std = self.results['estimate_std']
        if std < 3:    conf = "HIGH (90-95%)"
        elif std < 6:  conf = "HIGH (85-90%)"
        elif std < 10: conf = "MEDIUM (75-85%)"
        else:          conf = "MEDIUM (70-75%)"
        self.results['confidence'] = conf
        return self.results