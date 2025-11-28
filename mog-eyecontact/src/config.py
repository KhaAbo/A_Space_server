from pathlib import Path
import yaml
from typing import Any, Dict


class Config:
    def __init__(self, data: Dict[str, Any], root: Path):
        self.data = data
        self.root = root

    def get(self, key: str, default: Any=None) -> Any:
        return self.data.get(key, default)

    def section(self, name: str) -> Dict[str, Any]:
        return self.data.get(name, {})

    def path(self, key: str) -> Path:
        raw = self.data["paths"][key]
        p = Path(raw)
        if p.is_absolute():
            return p
        return self.root / p


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    with open(p, "r") as f:
        data = yaml.safe_load(f)

    root = p.parent.parent  # project root
    return Config(data, root)
