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
"""
train and evaluate resnet example for cifar10 dataset
1.The sample can only be run on Ascend 910 AI processor.
2.Aroud 30s per epoch and about 90% accuracy when the number of epoch reaches 34.
"""
import os
import random
import argparse
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.ops.functional as F
import mindspore.dataset as de
import mindspore.dataset.transforms.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model, ParallelMode
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from resnet import resnet50
random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

parser = argparse.ArgumentParser(description='Image classification.')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distributei.')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--do_train', type=bool, default=True, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
parser.add_argument('--epoch_size', type=int, default=1, help='Epoch size.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--num_classes', type=int, default=10, help='Num classes.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path.')
parser.add_argument('--dataset_path', type=str, default="./datasets/cifar/cifar-10-batches-bin",
                    help='Dataset path.')
args_opt = parser.parse_args()

#The path of the data.
data_home = args_opt.dataset_path

#Choose the graph_mode as mode, the env is Ascend and save graphs like ir
context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, save_graphs=True)
if args_opt.device_target == "Ascend":
    #Choose one availabe Device to use on users' env.
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)

def create_dataset(repeat_num=1, training=True):
    """create the dataset of cifar10"""
    ds = de.Cifar10Dataset(data_home)

    if args_opt.run_distribute:
        rank_id = int(os.getenv('RANK_ID'))
        rank_size = int(os.getenv('RANK_SIZE'))
        ds = de.Cifar10Dataset(data_home, num_shards=rank_size, shard_id=rank_id)

    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = C.RandomCrop((32, 32), (4, 4, 4, 4)) # padding_mode default CONSTANT
    random_horizontal_op = C.RandomHorizontalFlip()
    resize_op = C.Resize((resize_height, resize_width)) # interpolation default BILINEAR
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = C.HWC2CHW()
    type_cast_op = C2.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    ds = ds.map(input_columns="label", operations=type_cast_op)
    ds = ds.map(input_columns="image", operations=c_trans)

    # apply repeat operations
    ds = ds.repeat(repeat_num)

    # apply shuffle operations
    ds = ds.shuffle(buffer_size=10)

    # apply batch operations
    ds = ds.batch(batch_size=args_opt.batch_size, drop_remainder=True)

    return ds

if __name__ == '__main__':
    if not args_opt.do_eval and args_opt.run_distribute:
        context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL, mirror_mean=True)
        auto_parallel_context().set_all_reduce_fusion_split_indices([140])
        init()

    epoch_size = args_opt.epoch_size
    net = resnet50(args_opt.num_classes)
    ls = SoftmaxCrossEntropyWithLogits(sparse=True, is_grad=False, reduction="mean")
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)

    model = Model(net, loss_fn=ls, optimizer=opt, metrics={'acc'})

    if args_opt.do_train:
        dataset = create_dataset()
        batch_num = dataset.get_dataset_size()
        config_ck = CheckpointConfig(save_checkpoint_steps=batch_num, keep_checkpoint_max=10)
        checkpoint_path = args_opt.checkpoint_path if args_opt.checkpoint_path is not None else "./"
        ckpoint_cb = ModelCheckpoint(prefix="train_resnet_cifar10", directory=checkpoint_path, config=config_ck)
        loss_cb = LossMonitor()
        model.train(epoch_size, dataset, callbacks=[ckpoint_cb, loss_cb])

    if args_opt.do_eval:
        if args_opt.checkpoint_path:
            param_dict = load_checkpoint(args_opt.checkpoint_path)
            load_param_into_net(net, param_dict)
        net.set_train(False)
        eval_dataset = create_dataset(1, training=False)
        res = model.eval(eval_dataset)
        print("result: ", res)
