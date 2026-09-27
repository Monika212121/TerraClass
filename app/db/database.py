import sqlite3
from pathlib import Path
from datetime import datetime, timezone



CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    tile_name TEXT NOT NULL,

    tile_sha256 TEXT NOT NULL,

    predicted_label TEXT NOT NULL,

    confidence REAL NOT NULL,

    model_version TEXT NOT NULL,

    inference_time_ms REAL NOT NULL,

    created_at TEXT NOT NULL
)
"""


class PredictionRepository:

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents= True, exist_ok= True)

        self._initialize()



    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection



    def _initialize(self):
        with self._connect() as connection:
            connection.execute(CREATE_TABLE)
            connection.commit()


    def save(self, tile_name: str, tile_sha256: str, predicted_label: str, confidence: float, model_version: str, inference_time_ms: float):

        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:

            cursor = connection.execute(
                """
                INSERT INTO predictions (
                    tile_name,
                    tile_sha256,
                    predicted_label,
                    confidence,
                    model_version,
                    inference_time_ms,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tile_name,
                    tile_sha256,
                    predicted_label,
                    confidence,
                    model_version,
                    inference_time_ms,
                    created_at,
                )
            )

            connection.commit()

            prediction_id = cursor.lastrowid

        return {
            "prediction_id":
                prediction_id,

            "tile_name":
                tile_name,

            "tile_sha256":
                tile_sha256,

            "predicted_label":
                predicted_label,

            "confidence":
                confidence,

            "model_version":
                model_version,

            "inference_time_ms":
                inference_time_ms,

            "created_at":
                created_at,
        }


    def get_all(self, limit: int = 100):
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()

        return [
            {
                "prediction_id": row["id"],
                "tile_name": row["tile_name"],
                "tile_sha256": row["tile_sha256"],
                "predicted_label": row["predicted_label"],
                "confidence": row["confidence"],
                "model_version": row["model_version"],
                "inference_time_ms": row["inference_time_ms"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]