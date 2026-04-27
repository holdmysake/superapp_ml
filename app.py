# filepath: /d:/fol_super_app/superapp_ml/app.py
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv # type: ignore
import os
from urllib.parse import quote
from sqlalchemy import text # type: ignore
from predict import predict_bp
from extensions import db

load_dotenv()

def create_app():
    app = Flask(__name__)

    db_password = quote(os.getenv('DB_PASSWORD'))

    app.config['SQLALCHEMY_DATABASE_URI'] = (
        f"mysql+pymysql://{os.getenv('DB_USERNAME')}:{db_password}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    try:
        db.init_app(app)
        with app.app_context():
            with db.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        print("Database connection successful!")
    except Exception as e:
        print(f"Database connection failed: {e}")

    @app.route("/")
    def index():
        return jsonify({"message": "Welcome to the Flask app with MySQL!"})

    app.register_blueprint(predict_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)