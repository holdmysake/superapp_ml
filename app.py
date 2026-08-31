from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv # type: ignore
from flask_cors import CORS # type: ignore
import os
from urllib.parse import quote
from sqlalchemy import text # type: ignore
from predict import predict_bp
from extensions import db

load_dotenv()

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for the entire app

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

    app.register_blueprint(predict_bp)

    import threading
    import time
    from datetime import datetime, timedelta

    def start_cleanup_scheduler(app):
        # Hindari running dua kali saat reload di mode Debug
        if app.debug and os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
            return

        def cleanup_loop():
            # Beri jeda 10 detik agar server up sepenuhnya terlebih dahulu
            time.sleep(10)
            while True:
                with app.app_context():
                    try:
                        from models import PredRes
                        threshold_time = datetime.now() - timedelta(hours=24)
                        deleted_count = db.session.query(PredRes).filter(
                            PredRes.is_saved == False,
                            PredRes.timestamp < threshold_time
                        ).delete()
                        db.session.commit()
                        if deleted_count > 0:
                            print(f"[{datetime.now()}] Background cleanup: Sukses menghapus {deleted_count} draf prediksi.")
                    except Exception as e:
                        print(f"[{datetime.now()}] Background cleanup error: {e}")
                
                # Hitung jumlah detik tersisa hingga pukul 00:00:00 (jam 12 malam berikutnya)
                now = datetime.now()
                tomorrow = now + timedelta(days=1)
                next_midnight = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0)
                sleep_seconds = int((next_midnight - now).total_seconds())
                
                # Tidurkan thread sampai tepat pukul 12 malam
                time.sleep(sleep_seconds)

        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

    start_cleanup_scheduler(app)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999, debug=True)