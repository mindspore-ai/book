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
# ============================================================================

import collections
import json
import os
import re
import string
import numpy as np
import urllib
import urllib.request

def get_data(dir_name):
    """
    Get data from imagenet as dict.

    Yields:
        data (dict of list): imagenet data list which contains dict.
    """
    map_file = os.path.join(dir_name, "labels_map.txt")

    if not os.path.exists(map_file):
        raise Exception("map file {} not exists".format(map_file))

    label_dict = {}
    with open(map_file) as fp:
        line = fp.readline()
        while line:
            labels = line.split(" ")
            label_dict[labels[1]] = labels[0]
            line = fp.readline()

    # get all the dir which are n02087046, n02094114, n02109525, ...
    dir_paths = {}
    image_dir = os.path.join(dir_name, "images")
    for item in label_dict:
        real_path = os.path.join(image_dir, label_dict[item])
        if not os.path.isdir(real_path):
            print("warning: {} dir is not exist".format(real_path))
            continue
        dir_paths[item] = real_path

    if not dir_paths:
        raise Exception("not valid image dir in {}".format(image_dir))

    # get the filename, label and image binary as a dict
    data_list = []
    for label in dir_paths:
        for item in os.listdir(dir_paths[label]):
            file_name = os.path.join(dir_paths[label], item)
            if not item.endswith("JPEG") and not item.endswith("jpg"):
                print("warning: {} file is not suffix with JPEG/jpg, skip it.".format(file_name))
                continue
            data = {}
            data["file_name"] = str(file_name)
            data["label"] = int(label)

            # get the image data
            image_file = open(file_name, "rb")
            image_bytes = image_file.read()
            image_file.close()
            data["data"] = image_bytes

            data_list.append(data)
    return data_list

def create_data_cache_dir():
    cwd = os.getcwd()
    target_directory = os.path.join(cwd, "data_cache")
    try:
        if not os.path.exists(target_directory):
            os.mkdir(target_directory)
    except OSError:
        print("Creation of the directory %s failed" % target_directory)
    return target_directory

def download_and_uncompress(files, source_url, target_directory, is_tar=False):
    for f in files:
        url = source_url + f
        target_file = os.path.join(target_directory, f)

        ##check if file already downloaded
        if not (os.path.exists(target_file) or os.path.exists(target_file[:-3])):
            urllib.request.urlretrieve(url, target_file)
            if is_tar:
                print("extracting from local tar file " + target_file)
                rc = os.system("tar -C " + target_directory + " -xvf " + target_file)
            else:
                print("unzipping " + target_file)
                rc = os.system("gunzip -f " + target_file)
            if rc != 0:
                print("Failed to uncompress ", target_file, " removing")
                os.system("rm " + target_file)
                ##exit with error so that build script will fail
                raise SystemError
        else:
            print("Using cached dataset at ", target_file)
