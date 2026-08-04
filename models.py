# filepath: /d:/fol_super_app/superapp_ml/models.py
from extensions import db  # Impor db dari extensions.py

class Trunkline(db.Model):
    __tablename__ = 'trunkline'

    id = db.Column(db.Integer, primary_key=True)
    tline_id = db.Column(db.String(255), nullable=False, unique=True)
    tline_name = db.Column(db.String(255), nullable=False)
    tline_length = db.Column(db.Integer)

    spots = db.relationship('Spot', backref='trunkline', lazy=True)

    def __repr__(self):
        return f"<Trunkline {self.tline_name}>"

class Spot(db.Model):
    __tablename__ = 'spot'

    id = db.Column(db.Integer, primary_key=True)
    spot_id = db.Column(db.String(255), nullable=False)
    spot_name = db.Column(db.String(255), nullable=False)
    tline_id = db.Column(db.String(255), db.ForeignKey('trunkline.tline_id'), nullable=False)
    kp_pos = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f"<Spot {self.spot_name}>"

class PredRes(db.Model):
    __tablename__ = 'pred_res'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pred_res_id = db.Column(db.String(15), nullable=False, unique=True)
    tline_id = db.Column(db.String(255), nullable=False)
    drop_index = db.Column(db.Integer, nullable=False)
    message = db.Column(db.Text, nullable=False)
    google_maps_link = db.Column(db.String(255), nullable=True)
    final_estimate = db.Column(db.Double, nullable=False)
    estimate_std = db.Column(db.Double, nullable=False)
    confidence = db.Column(db.String(50), nullable=False)
    method_estimates = db.Column(db.JSON, nullable=False)
    method_weights = db.Column(db.JSON, nullable=False)
    gradients = db.Column(db.JSON, nullable=False)
    regions = db.Column(db.JSON, nullable=False)
    hgl_fit = db.Column(db.JSON, nullable=True)
    sensors = db.Column(db.JSON, nullable=False)
    is_saved = db.Column(db.Boolean, nullable=False, default=False)
    timestamp = db.Column(db.DateTime, nullable=False)

    def __repr__(self):
        return f"<PredRes {self.pred_res_id} - {self.tline_id}>"