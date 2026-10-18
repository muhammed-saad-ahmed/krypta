"""Simple, real experiment log — appends to experiments/log.json. No experiment has been run
yet, so this file does not exist until the first log_experiment() call; list_experiments()
returns [] honestly rather than fabricated rows until then."""
import json
import os
import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "log.json")


def log_experiment(name, model, dataset_version, params, metrics, notes=""):
    entry = {"name": name, "date": datetime.datetime.now().isoformat(), "model": model,
             "dataset_version": dataset_version, "params": params, "metrics": metrics, "notes": notes}
    log = list_experiments()
    log.append(entry)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)
    return entry


def list_experiments():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        return json.load(f)
