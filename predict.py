from flask import Blueprint, request, jsonify
import json
import numpy as np
from extensions import db
from models import Trunkline, Spot
from predict_utils import (
    load_coords,
    get_latlon_at_km,
    load_historical_data,
    build_calibration,
    PipelineLeakAnalyzer
)

predict_bp = Blueprint('predict_bp', __name__)

def predict(tline_id):
    data = request.get_json()
    if not data or 'normal' not in data or 'drop' not in data:
        return jsonify({'error': 'Invalid input, normal and drop are required'}), 400

    normal = data['normal']
    drop   = data['drop']

    tline = Trunkline.query.filter_by(tline_id=tline_id).first()
    if not tline:
        return jsonify({'error': 'Trunkline not found'}), 404

    spots = Spot.query.filter_by(tline_id=tline.tline_id).order_by(Spot.kp_pos).all()
    if not spots:
        return jsonify({'error': 'No spots found for the given trunkline'}), 404

    sensor_locations = [spot.kp_pos    for spot in spots]
    sensor_names     = [spot.spot_name for spot in spots]
    n_sensors        = len(sensor_locations)

    if len(normal) != n_sensors:
        return jsonify({'error': f'normal harus {n_sensors} elemen'}), 400

    drop_list = drop if isinstance(drop[0], list) else [drop]

    for i, d in enumerate(drop_list):
        if len(d) != n_sensors:
            return jsonify({'error': f'drop[{i}] harus {n_sensors} elemen'}), 400

    # Dynamic paths based on folder structure matching tline_id
    xlsx_path = f"data/{tline_id}/xlsx.xlsx"
    json_path = f"data/{tline_id}/json.json"

    coords = load_coords(xlsx_path)

    # Load historical calibration data if the JSON file exists
    hist_data = load_historical_data(json_path)

    results = []
    for idx, drop_arr in enumerate(drop_list):
        # Filter active sensors (Offline sensor if Normal P == 0 and Drop P == 0)
        kp_arr   = np.array(sensor_locations)
        norm_arr = np.array(normal)
        drop_arr = np.array(drop_arr)

        active_mask = ~((norm_arr == 0) & (drop_arr == 0))
        active_locs = kp_arr[active_mask]
        active_norm = norm_arr[active_mask]
        active_drop = drop_arr[active_mask]
        n_active = int(np.sum(active_mask))

        if n_active < 2:
            return jsonify({'error': f'Minimal 2 sensor aktif untuk drop ke-{idx}! Saat ini hanya {n_active}.'}), 400

        try:
            # Build calibration
            hist_json_tuple = tuple(json.dumps(rec, sort_keys=True) for rec in hist_data)
            calib = build_calibration(hist_json_tuple, tuple(sensor_locations))

            # Instantiate analyzer & run analysis
            analyzer = PipelineLeakAnalyzer(active_locs, active_norm, active_drop, calibration=calib)
            prediction = analyzer.run_full_analysis()
        except Exception as e:
            return jsonify({'error': f'Prediction drop[{idx}] failed: {str(e)}'}), 500

        final_kp = float(prediction['final_estimate'])
        std      = float(prediction['estimate_std'])
        conf     = prediction['confidence']

        maps_link = None
        if coords:
            try:
                leak_lat, leak_lon, _ = get_latlon_at_km(coords, final_kp)
                maps_link = f"https://www.google.com/maps?q={leak_lat:.6f},{leak_lon:.6f}"
            except Exception as e:
                maps_link = f"GPS error: {str(e)}"

        message = (
            f"Terjadi kebocoran pada titik {final_kp:.2f} KM "
            f"dengan kemungkinan pergeseran sejauh {std:.2f} KM. "
            f"Tingkat keakuratan prediksi sebesar {conf}."
        )

        results.append({
            'drop_index':       idx,
            'message':          message,
            'google_maps_link': maps_link,
        })
    
    return jsonify(results), 200

@predict_bp.route("/predict_bjg_tpn", methods=['POST'])
def predict_bjg_tpn():
    return predict("bjg_tpn")

@predict_bp.route("/predict_btj_bjg", methods=['POST'])
def predict_btj_bjg():
    return predict("btj_bjg")

@predict_bp.route("/predict_kas_tpn", methods=['POST'])
def predict_kas_tpn():
    return predict("kas_tpn")

@predict_bp.route("/predict_ktt_kas", methods=['POST'])
def predict_ktt_kas():
    return predict("ktt_kas")

@predict_bp.route("/predict_sgl_kas", methods=['POST'])
def predict_sgl_kas():
    return predict("sgl_kas")