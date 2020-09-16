# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import collections
import numpy as np
import os
import re
import string

import mindspore._c_dataengine as de_map
import mindspore.dataset as ds
from mindspore._c_dataengine import InterpolationMode

from write_mindrecord import write_mindrecord_tutorial

MINDRECORD_FILE_NAME = "./imagenet.mindrecord"

def create_dataset_using_mindrecord_tutorial():
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    data_set = ds.MindDataset(MINDRECORD_FILE_NAME, columns_list, num_readers)

    # add your data enhance code here

    assert data_set.get_dataset_size() == 20
    data_set = data_set.repeat(2)

    num_iter = 0
    for item in data_set.create_dict_iterator(output_numpy=True):
        print("-------------- index {} -----------------".format(num_iter))
        # print("-------------- item[label]: {} ---------------------".format(item["label"]))
        # print("-------------- item[data]: {} ----------------------".format(item["data"]))
        num_iter += 1
    assert num_iter == 40

if __name__ == '__main__':
    write_mindrecord_tutorial()

    create_dataset_using_mindrecord_tutorial()

    os.remove(MINDRECORD_FILE_NAME)
    os.remove(MINDRECORD_FILE_NAME + ".db")
