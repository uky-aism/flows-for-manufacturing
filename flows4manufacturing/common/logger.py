import os
from datetime import datetime
from glob import glob
from typing import Any, Optional

import neptune.new as neptune
import torch
import yaml
from matplotlib.figure import Figure


class GenericLogger:
    def log(self, key: str, value: Any):
        pass

    def set(self, key: str, value: Any):
        pass

    def upload(self, key: str, path: str):
        pass


class LocalLogger(GenericLogger):
    def __init__(self, dir: str = ".", cache_limit: int = 500):
        super().__init__()
        self._path = os.path.join(dir, datetime.now().isoformat()[:-5].replace(":", ""))
        os.makedirs(self._path)
        self._key_counters = {}
        self._cache = []
        self._cache_limit = cache_limit

    def log(self, key: str, value: Any):
        if isinstance(value, Figure):
            clean_key = key.replace("/", "-")
            counter = self._key_counters.get(key, 0)
            os.makedirs(os.path.join(self._path, clean_key), exist_ok=True)
            value.savefig(
                os.path.join(self._path, clean_key, f"{clean_key}-{counter:05d}.jpg"),
                dpi=150,
            )
            self._key_counters[key] = counter + 1
        else:
            self._cache.append((key, value))
            if len(self._cache) > self._cache_limit:
                keys = set([x[0] for x in self._cache])
                for key in keys:
                    counter = self._key_counters.get(key, 0)
                    values = [x[1] for x in self._cache if x[0] == key]
                    clean_key = key.replace("/", "-")
                    out_path = os.path.join(self._path, f"{clean_key}.csv")
                    lines = []

                    if isinstance(values[0], torch.Tensor):
                        lines = [
                            f"{counter+i},{v.item()}\n" for i, v in enumerate(values)
                        ]

                    with open(out_path, "a") as out_file:
                        out_file.writelines(lines)
                    self._key_counters[key] = counter + len(values)
                self._cache.clear()

    def set(self, key: str, value: Any):
        hparam_path = os.path.join(self._path, "hparams.yml")
        content = {}
        if os.path.exists(hparam_path):
            with open(hparam_path, "r") as hparams_file:
                content = yaml.load(hparams_file, yaml.SafeLoader)
        content[key] = value
        with open(hparam_path, "w") as hparams_file:
            yaml.dump(content, hparams_file, yaml.SafeDumper)


class SafeLogger(GenericLogger):
    def __init__(
        self,
        project: Optional[str],
        source_files_glob="*.py",
        backup: Optional[GenericLogger] = None,
    ):
        super().__init__()
        if project is not None:
            self._run = neptune.init_run(project, source_files=glob(source_files_glob))
        else:
            self._run = None
        self._backup = backup

    def log(self, key: str, value: Any):
        if self._run is not None:
            self._run[key].log(value)
        elif self._backup is not None:
            self._backup.log(key, value)

    def set(self, key: str, value: Any):
        if self._run is not None:
            self._run[key] = value
        elif self._backup is not None:
            self._backup.set(key, value)

    def upload(self, key: str, path: str):
        if self._run is not None:
            self._run[key].upload(path)
        elif self._backup is not None:
            self._backup.upload(key, path)
