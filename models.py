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