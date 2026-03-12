
import importlib.util
import sys
import logging
from pathlib import Path

log = logging.getLogger(__name__)

PHASE4_DIR = Path(__file__).resolve().parent.parent / "Phase_4"


def _load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_phase4_modules():
    p4_config_path = PHASE4_DIR / "config.py"
    p4_model_path  = PHASE4_DIR / "model.py"

    p5_config = sys.modules.get("config")

    p4_config = _load_module_from_file("config", str(p4_config_path))

    p4_model = _load_module_from_file("phase4_model", str(p4_model_path))

    if p5_config is not None:
        sys.modules["config"] = p5_config
    else:
        del sys.modules["config"]

    return p4_config, p4_model

