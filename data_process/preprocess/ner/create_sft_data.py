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


def read_chem_x_gene():
    """  """
    file = Path('/mnt/nas1/corpus-bio-nlp/NER/PGx_CTD_chem_x_gene.csv')
    df = pd.read_csv(file, dtype=str)
    ic(df.columns)


def main():
    read_chem_x_gene()


if __name__ == '__main__':
    main()
