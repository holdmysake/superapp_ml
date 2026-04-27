from flask import Blueprint, request, jsonify
from extensions import db
from models import Trunkline, Spot
from predict_utils import load_and_prepare_model

predict_bp = Blueprint('predict_bp', __name__)

def predict(tline_id):
    model_file     = f"data/{tline_id}/sav.sav"
    elevation_file = f"data/{tline_id}/xlsx.xlsx"

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

    model, gps_mapper, load_error = load_and_prepare_model(model_file, elevation_file)
    if load_error:
        return jsonify({'error': load_error}), 500

    results = []
    for idx, drop_arr in enumerate(drop_list):
        try:
            prediction = model.predict(sensor_locations, normal, drop_arr, sensor_names)
        except Exception as e:
            return jsonify({'error': f'Prediction drop[{idx}] failed: {str(e)}'}), 500

        final_kp = float(prediction['final_estimate'])
        std      = float(prediction['estimate_std'])
        conf     = prediction['confidence']

        maps_link = None
        if gps_mapper is not None:
            try:
                maps_link = gps_mapper.get_google_maps_link(final_kp, zoom=18)
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