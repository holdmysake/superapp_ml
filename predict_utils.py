import pickle
import numpy as np
import pandas as pd
from scipy import interpolate


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_elevation_data(file_path):
    try:
        df = pd.read_excel(file_path)
        df.columns = ['latitude', 'longitude', 'elevation']
        distances = [0.0]
        for i in range(1, len(df)):
            lat1, lon1 = df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude']
            lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
            R    = 6371
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
        import __main__
        if not hasattr(__main__, 'EnhancedLeakAnalyzer'):
            __main__.EnhancedLeakAnalyzer = EnhancedLeakAnalyzer
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, f"Model file '{file_path}' tidak ditemukan"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def load_and_prepare_model(model_file, elevation_file):
    model, error = load_model(model_file)
    if error:
        return None, None, error

    model.base_config['PSI_PER_METER'] = 0.1209

    elev_df, _ = load_elevation_data(elevation_file)
    gps_mapper = None

    if elev_df is None:
        if model.has_elevation_data and model.elevation_df is not None:
            gps_mapper = GPSLocationMapper(model.elevation_df)
        else:
            model.has_elevation_data = False
    else:
        model.elevation_df       = elev_df
        model.has_elevation_data = True
        gps_mapper               = GPSLocationMapper(elev_df)

    return model, gps_mapper, None


# ── EnhancedLeakAnalyzer ──────────────────────────────────────────────────────

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

        if   estimate_std < 2: confidence = "95%+"
        elif estimate_std < 4: confidence = "90-95%"
        elif estimate_std < 6: confidence = "85-90%"
        else:                  confidence = "75-85%"

        return {
            'final_estimate': final_estimate,
            'estimate_std':   estimate_std,
            'confidence':     confidence,
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
        cfg      = self.base_config
        top_idx  = np.argmax(suspicion_index)
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
        self.elevation_df     = elevation_df
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