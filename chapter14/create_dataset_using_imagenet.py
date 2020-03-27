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

import mindspore.dataset as de
import mindspore.dataset.transforms.vision.py_transforms as F

def create_imagenet_dataset(imagenet_dir):
    ds = de.ImageFolderDatasetV2(imagenet_dir)

    transform = F.ComposeOp([F.Decode(),
                             F.RandomHorizontalFlip(0.5),
                             F.ToTensor(),
                             F.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
                             F.RandomErasing()])
    ds = ds.map(input_columns="image", operations=transform())
    ds = ds.shuffle(buffer_size=5)
    ds = ds.repeat(3)
    return ds

if __name__ == "__main__":
    data_set = create_imagenet_dataset('ImageNetDataSimulation/images')
    count = 0
    for data in data_set.create_dict_iterator():
        print(data['image'].shape)
        print('------------')
        count += 1
    print(count)
