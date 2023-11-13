import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

sys.path.append(os.path.abspath('.'))
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120


# one line string
S = (
    ' (2) After treatment with ATRA, the fusion protein disappeared and PML protein resumed '
    'in NB4 cells, while in HL-60 and K562 cells there was no difference from control cells.'
)

def split_with_span_index(sentence: str):
    words = sentence.split()
    search_start = 0
    out = []
    for word in words:
        start = sentence.find(word, search_start)
        end = start + len(word)
        out.append((word, start, end))
        search_start = end
    return out


if __name__ == '__main__':
    print(split_with_span_index(S))
