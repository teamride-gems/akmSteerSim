import csv
from pathlib import Path

class LapMetricsLogger:
    FIELDNAMES = [
        "lap_id",
        "policy_id",
        "action_space_id",
        "track_id",
        "lap_status",       # SUCCESS | CRASH | TIMEOUT
        "lap_time_sec",
        "lap_progress",     # float in [0, 1]
    ]

    def __init__(self, csv_path: str = "metrics/lap_metrics.csv"):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.FIELDNAMES)
        self.csv_writer.writeheader()

    def log_lap(
        self,
        lap_id: int,
        policy_id: str,
        action_space_id: str,
        track_id: str,
        lap_status: str,        # "SUCCESS" | "CRASH" | "TIMEOUT"
        lap_time_sec: float,
        lap_progress: float,    # [0.0, 1.0]
    ) -> None:
        assert lap_status in {"SUCCESS", "CRASH", "TIMEOUT"}, (
            f"Invalid lap_status: {lap_status!r}"
        )
        self.csv_writer.writerow({
            "lap_id": lap_id,
            "policy_id": policy_id,
            "action_space_id": action_space_id,
            "track_id": track_id,
            "lap_status": lap_status,
            "lap_time_sec": round(lap_time_sec, 4),
            "lap_progress": round(float(lap_progress), 6),
        })
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()