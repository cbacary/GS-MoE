import os
import json
import shutil
from datetime import datetime
from typing import Dict, Optional


class MetricsManager:
    _all_metrics: Dict[str, dict] = {}

    @staticmethod
    def load_metrics(path: str = "./metrics/metrics.json"):
        pass
        # if not MetricsManager._all_metrics and os.path.exists(path):
        #     with open(path, "r") as f:
        #         MetricsManager._all_metrics = json.load(f)

    @staticmethod
    def build_metrics() -> dict:
        run_name = datetime.now().strftime("%Y_%m_%d--%H_%M_%S")
        metrics = {
            "RunName": run_name,
            "RunDate": datetime.now().strftime("%b %d, %Y %I:%M %p"),
            "latent_moe_metrics": {},
        }
        MetricsManager._all_metrics[run_name] = metrics
        return metrics

    @staticmethod
    def save_metrics_static(
        path: str = "./metrics/metrics.json",
        backup_dir: str = "./metrics/backups/",
        jsons_path: str = "./metrics/jsons/",
        run_metrics: Optional[dict] = None,
    ):
        pass
        # os.makedirs(backup_dir, exist_ok=True)

        # with open(path, "w") as f:
        #     json.dump(MetricsManager._all_metrics, f, indent=4)

        # if run_metrics:
        #     os.makedirs(jsons_path, exist_ok=True)
        #     filename_date = run_metrics.get(
        #         "RunName", datetime.now().strftime("%Y_%m_%d--%H_%M_%S")
        #     )
        #     filename_note = ""
        #     if "note" in run_metrics and run_metrics["note"]:
        #         filename_note = run_metrics["note"][:15]

        #     run_metrics_path = os.path.join(jsons_path, f"{filename_date}--{filename_note}.json")

        #     with open(run_metrics_path, "w") as f:
        #         json.dump(run_metrics, f, indent=4)
