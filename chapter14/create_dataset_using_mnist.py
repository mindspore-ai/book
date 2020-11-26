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
from easydict import EasyDict as edict

import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as VC
from mindspore.dataset.vision.utils import Inter
from mindspore import dtype as mstype
import utils

MNIST_URL = "http://yann.lecun.com/exdb/mnist/"

MNIST_CONFIG = edict({
    'num_classes': 10,
    'lr': 0.01,
    'momentum': 0.9,
    'epoch_size': 1,
    'batch_size': 32,
    'repeat_size': 1,
    'buffer_size': 1000,
    'image_height': 32,
    'image_width': 32,
    'save_checkpoint_steps': 1875,
    'keep_checkpoint_max': 10,
})

def download_mnist(target_directory=None):
    if target_directory is None:
        target_directory = utils.create_data_cache_dir()

        ##create mnst directory
        target_directory = os.path.join(target_directory, "mnist")
        try:
            if not os.path.exists(target_directory):
                os.mkdir(target_directory)
        except OSError:
            print("Creation of the directory %s failed" % target_directory)

    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']
    utils.download_and_uncompress(files, MNIST_URL, target_directory, is_tar=False)

    return target_directory, os.path.join(target_directory, "datasetSchema.json")

def create_mnist_dataset(mnist_dir, num_parallel_workers=1):
    ds = de.MnistDataset(mnist_dir)

    # apply map operations on images
    ds = ds.map(operations=C.TypeCast(mstype.int32), input_columns="label")
    ds = ds.map(operations=VC.Resize((MNIST_CONFIG.image_height, MNIST_CONFIG.image_width),
                                     interpolation=Inter.LINEAR),
                input_columns="image",
                num_parallel_workers=num_parallel_workers)
    ds = ds.map(operations=VC.Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081),
                input_columns="image",
                num_parallel_workers=num_parallel_workers)
    ds = ds.map(operations=VC.Rescale(1.0 / 255.0, 0.0),
                input_columns="image",
                num_parallel_workers=num_parallel_workers)
    ds = ds.map(operations=VC.HWC2CHW(),
                input_columns="image",
                num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    ds = ds.shuffle(buffer_size=MNIST_CONFIG.buffer_size)  # 10000 as in LeNet train script
    ds = ds.batch(MNIST_CONFIG.batch_size, drop_remainder=True)
    ds = ds.repeat(MNIST_CONFIG.repeat_size)

    return ds

if __name__ == "__main__":
    mnistDir, _ = download_mnist()
    data_set = create_mnist_dataset(mnistDir, 2)
    for data in data_set.create_dict_iterator():
        print(data['image'].shape)
        print(data['label'])
        print('------------')
