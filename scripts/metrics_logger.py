import csv
from pathlib import Path

class LapMetricsLogger:
    def __init__(self, csv_path: str = "metrics/lap_metrics.csv"):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["lap_id", "lap_time_sec"])

    def log_lap(self, lap_id: int, lap_time_sec: float):
        self.csv_writer.writerow([lap_id, lap_time_sec])
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()