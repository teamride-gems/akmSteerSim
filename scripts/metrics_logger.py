import csv
from pathlib import Path

class LapMetricsLogger:
    FIELDNAMES = [
        "lap_id",
        "policy_id",
        "action_space_id",
        "track_id",
        "lap_status",
        "lap_time_sec",
        "lap_progress",
    ]

    def __init__(self, csv_path: str = "metrics/lap_metrics.csv", timestep_csv_path: str | None = None):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.FIELDNAMES)
        self.csv_writer.writeheader()

        # time step edits - nik
        self.timestep_csv_path = None
        self.timestep_csv_file = None
        self.timestep_csv_writer = None
        self._prev_t_sec = None
        self._prev_speed_mps = None

        # time step edits - nik
        if timestep_csv_path is not None:
            self.enable_step_log(timestep_csv_path)

    def log_lap(
        self,
        lap_id: int,
        lap_time_sec: float,
        policy_id: str = "heuristic run",
        action_space_id: str = "N/A",
        track_id: str = "Sakhir",
        lap_status: str = "SUCCESS",
        lap_progress: float = 0.0,
    ) -> None:
        if lap_status not in {"SUCCESS", "CRASH", "TIMEOUT"}:
            raise ValueError(f"Invalid lap_status: {lap_status!r}")

        self.csv_writer.writerow(
            {
                "lap_id": int(lap_id),
                "policy_id": policy_id,
                "action_space_id": action_space_id,
                "track_id": track_id,
                "lap_status": lap_status,
                "lap_time_sec": round(float(lap_time_sec), 4),
                "lap_progress": round(float(lap_progress), 6),
            }
        )
        self.csv_file.flush()

    # time step edits - nik
    def enable_step_log(self, csv_path: str = "metrics/timestep_metrics.csv"):
        self.timestep_csv_path = Path(csv_path)
        self.timestep_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.timestep_csv_file = open(self.timestep_csv_path, "w", newline="")
        self.timestep_csv_writer = csv.writer(self.timestep_csv_file)
        self.timestep_csv_writer.writerow(["lap_id", "t_sec", "speed_mps", "acceleration_mps2"])
        self._prev_t_sec = None
        self._prev_speed_mps = None

    # time step edits - nik
    def log_step(self, t_sec: float, speed_mps: float, acceleration_mps2: float, lap_id: int = -1):
        if self.timestep_csv_writer is None:
            raise RuntimeError("Timestep log not enabled.")
        self.timestep_csv_writer.writerow([int(lap_id), float(t_sec), float(speed_mps), float(acceleration_mps2)])
        self.timestep_csv_file.flush()

    # time step edits - nik
    def log_step_speed(self, t_sec: float, speed_mps: float, lap_id: int = -1):
        t_sec = float(t_sec)
        speed_mps = float(speed_mps)

        if self._prev_t_sec is None or self._prev_speed_mps is None:
            acceleration_mps2 = 0.0
        else:
            dt = max(t_sec - self._prev_t_sec, 1e-9)
            acceleration_mps2 = (speed_mps - self._prev_speed_mps) / dt

        self.log_step(t_sec=t_sec, speed_mps=speed_mps, acceleration_mps2=acceleration_mps2, lap_id=lap_id)
        self._prev_t_sec = t_sec
        self._prev_speed_mps = speed_mps

    # time step edits - nik
    def reset_step(self):
        self._prev_t_sec = None
        self._prev_speed_mps = None

    def close(self):
        self.csv_file.close()
        # time step edits - nik
        if self.timestep_csv_file is not None:
            self.timestep_csv_file.close()