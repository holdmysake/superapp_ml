import os
import sys
from datetime import datetime, timedelta

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from models import PredRes
from extensions import db

def run_cleanup():
    app = create_app()
    with app.app_context():
        try:
            # Hapus draft prediksi (is_saved = False/0) yang umurnya sudah lebih dari 24 jam
            # untuk menghindari terhapusnya draf yang saat ini sedang aktif dihitung/diedit
            threshold_time = datetime.now() - timedelta(hours=24)
            deleted_count = db.session.query(PredRes).filter(
                PredRes.is_saved == False,
                PredRes.timestamp < threshold_time
            ).delete()
            
            db.session.commit()
            print(f"[{datetime.now()}] Cleanup sukses: Menghapus {deleted_count} draft prediksi yang tidak disimpan.")
        except Exception as e:
            db.session.rollback()
            print(f"[{datetime.now()}] Cleanup gagal: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    run_cleanup()
