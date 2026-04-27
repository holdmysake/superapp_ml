import pickle
import numpy as np
import pandas as pd
from scipy import interpolate
from flask import Blueprint, request, jsonify
import os
from extensions import db
from models import Trunkline, Spot

# ── Pipeline config (hanya untuk referensi default, data sensor dari DB) ──────
PIPELINE_CONFIG = {
    "tline_id": "bjg_tpn",
    "model_file": "data/bjg_tpn/sav.sav",
    "elevation_file": "data/bjg_tpn/xlsx.xlsx",
    "total_length": 26.6,
    "example_normal": [136.0, 112.14, 95.4, 37.1],
    "example_drop": [133.0, 110.1, 82.5, 34.5],
    "description": "Pipeline dari Betara Jambi (BJG) menuju Tempino (TPN)",
    "fluid_type": "Crude Oil",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_elevation_data(file_path):
    try:
        df = pd.read_excel(file_path)
        df.columns = ['latitude', 'longitude', 'elevation']
        distances = [0.0]
        for i in range(1, len(df)):
            lat1, lon1 = df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude']
            lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
            R = 6371
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) *
                 np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))
            distances.append(distances[-1] + R * c)
        df['distance_km'] = distances
        return df, None
    except FileNotFoundError:
        return None, f"File '{file_path}' tidak ditemukan"
    except Exception as e:
        return None, f"Error loading elevation data: {str(e)}"


def load_model(file_path):
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, f"Model file '{file_path}' tidak ditemukan"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


# ── EnhancedLeakAnalyzer (harus ada di sini agar pickle.load bisa resolve) ───

class EnhancedLeakAnalyzer:
    def __init__(self, base_config, elevation_df=None):
        self.base_config = base_config
        self.elevation_df = elevation_df
        self.has_elevation_data = elevation_df is not None

    def predict(self, sensor_locations, normal_pressure, drop_pressure, sensor_names=None):
        locations = np.array(sensor_locations)
        normal_p  = np.array(normal_pressure)
        drop_p    = np.array(drop_pressure)
        n_sensors = len(locations)

        if len(normal_p) != n_sensors:
            raise ValueError(f"normal_pressure harus {n_sensors} elemen")
        if len(drop_p) != n_sensors:
            raise ValueError(f"drop_pressure harus {n_sensors} elemen")
        if sensor_names is None:
            sensor_names = [f'Sensor {i+1} (KP {loc:.1f})' for i, loc in enumerate(locations)]

        if self.has_elevation_data:
            elev_interp = interpolate.interp1d(
                self.elevation_df['distance_km'], self.elevation_df['elevation'],
                kind='cubic', fill_value='extrapolate')
            elevations = elev_interp(locations)
        else:
            elevations = np.zeros(n_sensors)

        delta_p     = normal_p - drop_p
        abs_delta_p = np.abs(delta_p)
        with np.errstate(divide='ignore', invalid='ignore'):
            pressure_ratio = abs_delta_p / np.abs(normal_p) * 100
        pressure_ratio = np.nan_to_num(pressure_ratio, 0.0)

        suspicion_index = self._calculate_suspicion_index(abs_delta_p, pressure_ratio, n_sensors)
        susp_loc     = self._suspicion_method(locations, suspicion_index)
        grad_loc     = self._gradient_method(locations, normal_p, drop_p)
        interp_loc   = self._interpolation_method(locations, abs_delta_p)
        weighted_loc = self._weighted_method(locations, suspicion_index)
        elev_loc     = self._elevation_method(locations, normal_p, drop_p, elevations, n_sensors)

        cfg = self.base_config
        final_estimate = (
            susp_loc     * cfg['FINAL_ESTIMATE_WEIGHTS']['suspicion'] +
            interp_loc   * cfg['FINAL_ESTIMATE_WEIGHTS']['interpolation'] +
            grad_loc     * cfg['FINAL_ESTIMATE_WEIGHTS']['gradient'] +
            elev_loc     * cfg['FINAL_ESTIMATE_WEIGHTS']['elevation'] +
            weighted_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['weighted']
        )
        estimate_std = np.std([susp_loc, interp_loc, grad_loc, elev_loc, weighted_loc])

        if   estimate_std < 2: confidence = "VERY HIGH (95%+)"
        elif estimate_std < 4: confidence = "HIGH (90-95%)"
        elif estimate_std < 6: confidence = "HIGH (85-90%)"
        else:                  confidence = "MEDIUM (75-85%)"

        return {
            'final_estimate': final_estimate,
            'estimate_std':   estimate_std,
            'confidence':     confidence,
            'zones': {
                'focus':    (final_estimate - 3,  final_estimate + 3),
                'critical': (final_estimate - 5,  final_estimate + 5),
                'primary':  (final_estimate - 10, final_estimate + 10),
            },
            'top_sensor_idx': np.argmax(suspicion_index),
            'methods': {
                'suspicion':     susp_loc,
                'interpolation': interp_loc,
                'gradient':      grad_loc,
                'elevation':     elev_loc,
                'weighted':      weighted_loc,
            },
            'sensor_data': {
                'locations':         locations,
                'names':             sensor_names,
                'elevations':        elevations,
                'normal_pressure':   normal_p,
                'drop_pressure':     drop_p,
                'delta_pressure':    delta_p,
                'abs_delta_pressure': abs_delta_p,
                'pressure_ratio':    pressure_ratio,
                'suspicion_index':   suspicion_index,
            },
        }

    def _calculate_suspicion_index(self, abs_delta_p, pressure_ratio, n_sensors):
        cfg = self.base_config
        suspicion_index = np.zeros(n_sensors)
        for i in range(n_sensors):
            delta_factor = abs_delta_p[i]
            ratio_factor = pressure_ratio[i]
            if i > 0 and i < n_sensors - 1:
                neighbor_avg  = (abs_delta_p[i-1] + abs_delta_p[i+1]) / 2
                neighbor_diff = abs_delta_p[i] - neighbor_avg
            elif i == 0 and n_sensors > 1:
                neighbor_diff = abs_delta_p[i] - abs_delta_p[i+1]
            elif i == n_sensors - 1 and n_sensors > 1:
                neighbor_diff = abs_delta_p[i] - abs_delta_p[i-1]
            else:
                neighbor_diff = 0
            neighbor_factor = max(0, neighbor_diff)
            suspicion_index[i] = (
                delta_factor    * cfg['SUSPICION_WEIGHTS'][0] +
                ratio_factor    * cfg['SUSPICION_WEIGHTS'][1] +
                neighbor_factor * cfg['SUSPICION_WEIGHTS'][2]
            )
        return suspicion_index

    def _suspicion_method(self, locations, suspicion_index):
        cfg     = self.base_config
        top_idx = np.argmax(suspicion_index)
        location = locations[top_idx] + cfg['UPSTREAM_BIAS_PRIMARY']
        return max(location, locations[0])

    def _gradient_method(self, locations, normal_p, drop_p):
        cfg = self.base_config
        if len(locations) < 2:
            return locations[0]
        changes, locs = [], []
        for i in range(len(locations) - 1):
            dist = locations[i+1] - locations[i]
            if dist > 0:
                norm_grad = (normal_p[i+1] - normal_p[i]) / dist
                drop_grad = (drop_p[i+1]   - drop_p[i])   / dist
                changes.append(np.abs(norm_grad - drop_grad))
                locs.append((locations[i] + locations[i+1]) / 2)
        if not changes:
            return locations[0]
        return locs[np.argmax(changes)] + cfg['UPSTREAM_BIAS_GRADIENT']

    def _interpolation_method(self, locations, abs_delta_p):
        cfg = self.base_config
        if len(locations) < 4:
            return locations[np.argmax(abs_delta_p)] + cfg['UPSTREAM_BIAS_INTERP']
        try:
            f      = interpolate.interp1d(locations, abs_delta_p, kind='cubic', fill_value='extrapolate')
            x_fine = np.linspace(locations.min(), locations.max(), 2000)
            y_fine = f(x_fine)
            return x_fine[np.argmax(y_fine)] + cfg['UPSTREAM_BIAS_INTERP']
        except Exception:
            return locations[np.argmax(abs_delta_p)] + cfg['UPSTREAM_BIAS_INTERP']

    def _weighted_method(self, locations, suspicion_index):
        cfg   = self.base_config
        total = np.sum(suspicion_index)
        if total == 0:
            return np.mean(locations)
        return np.sum(suspicion_index * locations) / total + cfg['UPSTREAM_BIAS_WEIGHTED']

    def _elevation_method(self, locations, normal_p, drop_p, elevations, n_sensors):
        cfg = self.base_config
        if not self.has_elevation_data:
            return locations[np.argmax(np.abs(normal_p - drop_p))] + cfg['UPSTREAM_BIAS_PRIMARY']
        psi_per_meter = cfg['PSI_PER_METER']
        ref_elev  = elevations[0]
        elev_corr = (elevations - ref_elev) * psi_per_meter
        normal_corr = normal_p - elev_corr
        drop_corr   = drop_p   - elev_corr
        anomaly_scores = np.zeros(n_sensors)
        for i in range(1, n_sensors):
            dist = locations[i] - locations[i-1]
            if dist > 0:
                exp_grad = (normal_corr[i-1] - normal_corr[i]) / dist
                act_grad = (drop_corr[i-1]   - drop_corr[i])   / dist
                anom = abs(act_grad - exp_grad)
                anomaly_scores[i-1] += anom * 0.5
                anomaly_scores[i]   += anom * 0.5
        return locations[np.argmax(anomaly_scores)] + cfg['UPSTREAM_BIAS_PRIMARY']


# ── GPSLocationMapper ─────────────────────────────────────────────────────────

class GPSLocationMapper:
    def __init__(self, elevation_df):
        self.elevation_df    = elevation_df
        self.lat_interpolator = interpolate.interp1d(
            elevation_df['distance_km'], elevation_df['latitude'],
            kind='cubic', fill_value='extrapolate')
        self.lon_interpolator = interpolate.interp1d(
            elevation_df['distance_km'], elevation_df['longitude'],
            kind='cubic', fill_value='extrapolate')

    def get_coordinates(self, kp_km):
        return float(self.lat_interpolator(kp_km)), float(self.lon_interpolator(kp_km))

    def get_google_maps_link(self, kp_km, zoom=18):
        lat, lon = self.get_coordinates(kp_km)
        return f"https://maps.google.com/?q={lat},{lon}&ll={lat},{lon}&z={zoom}"


# ── Blueprint ─────────────────────────────────────────────────────────────────

fol2_bp = Blueprint('fol2', __name__)


def _load_and_prepare_model(model_file, elevation_file):
    """
    Load .sav (pickle), patch PSI_PER_METER, inject elevation data.
    Return (model, gps_mapper, elev_source, error_str)
    """
    model, error = load_model(model_file)
    if error:
        return None, None, None, error

    # ── Patch PSI_PER_METER (identik dengan fol2 standalone) ──
    model.base_config['PSI_PER_METER'] = 0.1209

    # ── Load fresh elevation ──
    elev_df, elev_error = load_elevation_data(elevation_file)
    gps_mapper  = None
    elev_source = "NONE"

    if elev_df is None:
        # Fallback ke elevation bawaan .sav
        if model.has_elevation_data and model.elevation_df is not None:
            gps_mapper  = GPSLocationMapper(model.elevation_df)
            elev_source = f"sav_builtin ({len(model.elevation_df)} titik)"
        else:
            model.has_elevation_data = False
            elev_source = "NONE - pure fallback"
    else:
        model.elevation_df      = elev_df
        model.has_elevation_data = True
        gps_mapper  = GPSLocationMapper(elev_df)
        elev_source = f"xlsx_fresh ({len(elev_df)} titik)"

    return model, gps_mapper, elev_source, None


@fol2_bp.route('/predict_bjg_tpn', methods=['POST'])
def predict_bjg_tpn():
    data = request.get_json()
    if not data or 'normal' not in data or 'drop' not in data:
        return jsonify({'error': 'Invalid input, normal and drop are required'}), 400

    normal = data['normal']
    drop   = data['drop']

    # ── Ambil data dari DB (sama seperti predict.py) ──────────────────────────
    tline = Trunkline.query.filter_by(tline_id=PIPELINE_CONFIG['tline_id']).first()
    if not tline:
        return jsonify({'error': 'Trunkline not found'}), 404

    spots = Spot.query.filter_by(tline_id=tline.tline_id).order_by(Spot.kp_pos).all()
    if not spots:
        return jsonify({'error': 'No spots found for the given trunkline'}), 404

    sensor_locations = [spot.kp_pos   for spot in spots]
    sensor_names     = [spot.spot_name for spot in spots]
    n_sensors        = len(sensor_locations)
    total_length     = tline.tline_length  # dari DB

    # ── Validasi panjang input ────────────────────────────────────────────────
    drop_arr = drop[0] if isinstance(drop[0], list) else drop
    if len(normal) != n_sensors:
        return jsonify({'error': f'normal harus {n_sensors} elemen'}), 400
    if len(drop_arr) != n_sensors:
        return jsonify({'error': f'drop harus {n_sensors} elemen'}), 400

    # ── Load model & elevation (cara fol2) ────────────────────────────────────
    model_file     = PIPELINE_CONFIG['model_file']
    elevation_file = PIPELINE_CONFIG['elevation_file']
    model, gps_mapper, elev_source, load_error = _load_and_prepare_model(model_file, elevation_file)
    if load_error:
        return jsonify({'error': load_error}), 500

    # ── Log base_config (untuk debug, tidak wajib di response) ───────────────
    cfg = model.base_config
    fw  = cfg.get('FINAL_ESTIMATE_WEIGHTS', {})

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        prediction = model.predict(sensor_locations, normal, drop_arr, sensor_names)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    final_kp = prediction['final_estimate']
    focus    = prediction['zones']['focus']
    critical = prediction['zones']['critical']
    primary  = prediction['zones']['primary']

    # Clamp zones ke panjang pipeline
    if total_length:
        focus    = (max(0, focus[0]),    min(total_length, focus[1]))
        critical = (max(0, critical[0]), min(total_length, critical[1]))
        primary  = (max(0, primary[0]),  min(total_length, primary[1]))

    # ── GPS ───────────────────────────────────────────────────────────────────
    gps_data   = None
    maps_link  = None
    zone_gps   = {}

    if gps_mapper is not None:
        try:
            lat, lon  = gps_mapper.get_coordinates(final_kp)
            maps_link = gps_mapper.get_google_maps_link(final_kp, zoom=18)
            gps_data  = {'latitude': lat, 'longitude': lon}

            for zone_name, zone in [('focus', focus), ('critical', critical), ('primary', primary)]:
                lat_s, lon_s = gps_mapper.get_coordinates(zone[0])
                lat_e, lon_e = gps_mapper.get_coordinates(zone[1])
                zone_gps[zone_name] = {
                    'start': {'latitude': lat_s, 'longitude': lon_s,
                              'maps_link': gps_mapper.get_google_maps_link(zone[0])},
                    'end':   {'latitude': lat_e, 'longitude': lon_e,
                              'maps_link': gps_mapper.get_google_maps_link(zone[1])},
                }
        except Exception as e:
            gps_data = None
            maps_link = f"GPS error: {str(e)}"

    # ── Unused sensors (nilai 0) ──────────────────────────────────────────────
    unused_sensors = [
        sensor_names[i]
        for i in range(n_sensors)
        if normal[i] == 0 or drop_arr[i] == 0
    ]

    # ── Response ──────────────────────────────────────────────────────────────
    sd = prediction['sensor_data']
    response = {
        'tline_id':     tline.tline_id,
        'tline_name':   tline.tline_name,
        'tline_length': tline.tline_length,
        'elev_source':  elev_source,
        'model_config': {
            'UPSTREAM_BIAS': [
                cfg.get('UPSTREAM_BIAS_PRIMARY'),
                cfg.get('UPSTREAM_BIAS_GRADIENT'),
                cfg.get('UPSTREAM_BIAS_INTERP'),
                cfg.get('UPSTREAM_BIAS_WEIGHTED'),
            ],
            'SUSPICION_WEIGHTS':  cfg.get('SUSPICION_WEIGHTS'),
            'FINAL_WEIGHTS': {
                'suspicion':     fw.get('suspicion'),
                'interpolation': fw.get('interpolation'),
                'gradient':      fw.get('gradient'),
                'elevation':     fw.get('elevation'),
                'weighted':      fw.get('weighted'),
            },
            'PSI_PER_METER': cfg.get('PSI_PER_METER'),
        },
        'prediction': {
            'final_estimate': float(prediction['final_estimate']),
            'estimate_std':   float(prediction['estimate_std']),
            'confidence':     prediction['confidence'],
            'zones': {
                'focus':    list(focus),
                'critical': list(critical),
                'primary':  list(primary),
            },
            'top_sensor_idx': int(prediction['top_sensor_idx']),
            'methods': {k: float(v) for k, v in prediction['methods'].items()},
            'sensor_data': {
                'locations':          sd['locations'].tolist(),
                'names':              sd['names'],
                'elevations':         sd['elevations'].tolist(),
                'normal_pressure':    sd['normal_pressure'].tolist(),
                'drop_pressure':      sd['drop_pressure'].tolist(),
                'delta_pressure':     sd['delta_pressure'].tolist(),
                'abs_delta_pressure': sd['abs_delta_pressure'].tolist(),
                'pressure_ratio':     sd['pressure_ratio'].tolist(),
                'suspicion_index':    sd['suspicion_index'].tolist(),
            },
        },
        'gps_coordinates': gps_data,
        'google_maps_link': maps_link,
        'zone_gps':        zone_gps,
        'unused_sensors':  unused_sensors,
    }

    return jsonify(response), 200