import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.append(os.path.abspath('.'))
from comm_utils.log_util import ic, logger

root_out = Path('/mnt/nas1/models/llama/quantized_models')
TASK_NAME_PREFIX = 'llama2-7b-ner-chem_gene'
events_dir = Path(f'runs/{TASK_NAME_PREFIX}')
events_dir.mkdir(parents=True, exist_ok=True)

for path in root_out.iterdir():
    if path.name.startswith(TASK_NAME_PREFIX) and path.is_dir():
        run_dir = path / 'runs'
        if not run_dir.exists():
            continue
        for run in run_dir.iterdir():
            out_dir = events_dir / run.name
            shutil.copytree(run, out_dir, dirs_exist_ok=True, symlinks=True)
            