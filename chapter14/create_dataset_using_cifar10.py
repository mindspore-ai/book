"""cifar10"""
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

import os

import mindspore.dataset as de
from mindspore.dataset.transforms import c_transforms as C
from mindspore.dataset.vision import c_transforms as vision
from mindspore import dtype as mstype
import utils

CIFAR_URL = "http://www.cs.toronto.edu/~kriz/"

def download_cifar(target_directory, files, directory_from_tar):
    """download cifar"""
    if target_directory is None:
        target_directory = utils.create_data_cache_dir()

    utils.download_and_uncompress([files], CIFAR_URL, target_directory, is_tar=True)

    ##if target dir was specify move data from directory created by tar
    ##and put data into target dir
    if target_directory is not None:
        tar_dir_full_path = os.path.join(target_directory, directory_from_tar)
        all_files = os.path.join(tar_dir_full_path, "*")
        cmd = "mv " + all_files + " " + target_directory
        if os.path.exists(tar_dir_full_path):
            print("copy files back to target_directory")
            print("Executing: ", cmd)
            rc1 = os.system(cmd)
            rc2 = os.system("rm -r " + tar_dir_full_path)
            if rc1 != 0 or rc2 != 0:
                print("error when running command: ", cmd)
                download_file = os.path.join(target_directory, files)
                print("removing " + download_file)
                os.system("rm " + download_file)

                ##exit with error so that build script will fail
                raise SystemError

    ##change target directory to directory after tar
    return target_directory, os.path.join(target_directory, directory_from_tar)

def create_cifar10_dataset(cifar_dir):
    """
    Creat the cifar10 dataset.
    """
    ds = de.Cifar10Dataset(cifar_dir)

    training = True
    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0
    repeat_num = 10
    batch_size = 32

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4)) # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize((resize_height, resize_width)) # interpolation default BILINEAR
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    ds = ds.map(operations=type_cast_op, input_columns="label")
    ds = ds.map(operations=c_trans, input_columns="image")

    # apply repeat operations
    ds = ds.repeat(repeat_num)

    # apply shuffle operations
    ds = ds.shuffle(buffer_size=10)

    # apply batch operations
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)

    return ds

def download_cifar10(target_directory=None):
    return download_cifar(target_directory, "cifar-10-binary.tar.gz", "cifar-10-batches-bin")

if __name__ == "__main__":
    dataset_dir, _ = download_cifar10()
    data_set = create_cifar10_dataset(dataset_dir)
    for data in data_set.create_dict_iterator():
        print(data['image'].shape)
        print(data['label'])
        print('------------')
