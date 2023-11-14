import json
import logging
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from icecream import ic

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
sys.path.append(os.path.abspath('.'))

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
root_dir = Path('/mnt/nas1/corpus-bio-nlp/NER/bionlp-st-2013-cg/en_core_sci_lg')


def check_bionlp_st_2013_cg_sanity(data):
    """  """
    for info in data:
        sent = info['sentence']
        spans = info['span']
        entities = info['entity']
        for span, entity_text in zip(spans, entities):
            start, end = span[:2]
            entity_type = span[2]
            assert entity_text == sent[start:end]
            
            # ic(start, end, entity_type, entity_text)


def read_bionlp_st_2013_cg():
    """  """
    for file in root_dir.glob('*.json'):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ic(file.name, len(data))
        # check_bionlp_st_2013_cg_sanity(data)
        # break


def main():
    read_bionlp_st_2013_cg()


if __name__ == '__main__':
    main()
