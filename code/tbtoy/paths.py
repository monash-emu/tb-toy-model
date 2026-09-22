from pathlib import Path

CODE_FOLDER = Path(__file__).resolve().parent.parent
REPO_ROOT_PATH = CODE_FOLDER.parent

DATA_FOLDER = REPO_ROOT_PATH / "data"
OUTPUT_PARENT_FOLDER = REPO_ROOT_PATH / "outputs"

PARAMS_PATH = DATA_FOLDER / "parameters.csv"
TV_PARAMS_PATH = DATA_FOLDER / "time_variant_parameters.csv"
TARGETS_PATH = DATA_FOLDER / "calibration_targets.csv"
