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
out_events_dir = Path(f'runs/{TASK_NAME_PREFIX}')
out_events_dir.mkdir(parents=True, exist_ok=True)

for path in root_out.iterdir():
    if path.name.startswith(TASK_NAME_PREFIX) and path.is_dir():
        run_dir = path / 'runs'
        if not run_dir.exists():
            continue
        event_dir = list(run_dir.iterdir())[0]
        out_dir = out_events_dir / event_dir.name
        for file in event_dir.iterdir():
            if file.name.startswith('events.out.tfevents.'):
                out_file = out_dir / file.name
                if not out_file.exists():
                    os.symlink(file, out_file)
        readme_file = out_dir / f'{path.name}.md'
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(f'# {path.name}\n')
        for file in path.iterdir():
            if file.name.endswith(('.log', '.json')):
                out_file = out_dir / file.name
                if not out_file.exists():
                    os.symlink(file, out_file)
