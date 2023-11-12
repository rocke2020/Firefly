import os
import math
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.append(os.path.abspath('.'))
from comm_utils.log_util import ic


train_params = {
    "data_count": 1655,
    "num_train_epochs": 3,
    "num_gpus": 2,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
}


def calc_epoch_data(params):
    """ max_steps is used to calculate the number of warmsup. """
    data_count = params["data_count"]
    num_train_epochs = params["num_train_epochs"]
    num_gpus = params["num_gpus"]
    per_device_train_batch_size = params["per_device_train_batch_size"]
    gradient_accumulation_steps = params["gradient_accumulation_steps"]


    len_dataloader = math.ceil(data_count / (per_device_train_batch_size * num_gpus))
    num_update_steps_per_epoch = len_dataloader // gradient_accumulation_steps
    max_steps = num_update_steps_per_epoch * num_train_epochs
    # num_train_samples = len_dataloader * per_device_train_batch_size * num_train_epochs
    ic(max_steps)


if __name__ == "__main__":
    calc_epoch_data(train_params)
