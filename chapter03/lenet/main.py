# Copyright 2020 Huawei Technologies Co., Ltd
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
######################## train and test lenet example ########################
1. train lenet and get network model files(.ckpt) :
python main.py --data_path /home/workspace/mindspore_dataset/Tutorial_Network/Lenet/MNIST_Data

2. test lenet according to model file:
python main.py --data_path /home/workspace/mindspore_dataset/Tutorial_Network/Lenet/MNIST_Data
               --mode test --ckpt_path checkpoint_lenet_1-1_1875.ckpt
"""
import os
import argparse
from config import mnist_cfg as cfg
from lenet import LeNet5
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
import mindspore.dataset.transforms.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.transforms.vision import Inter
from mindspore.nn.metrics import Accuracy
from mindspore.common import dtype as mstype


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=resize_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_nml_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore MNIST Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'test'],
                        help='implement phase, set to train or test')
    parser.add_argument('--data_path', type=str, default="./MNIST_Data",
                        help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="./", help='if mode is test, must provide\
                        path where the trained ckpt file')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='dataset_sink_mode is False or True')

    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    network = LeNet5(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
    repeat_size = 1
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    if args.mode == 'train':  # train
        ds_train = create_dataset(os.path.join(args.data_path, args.mode), batch_size=cfg.batch_size,
                                  repeat_size=repeat_size)
        print("============== Starting Training ==============")
        model.train(cfg['epoch_size'], ds_train, callbacks=[ckpoint_cb, LossMonitor()],
                    dataset_sink_mode=args.dataset_sink_mode)
    elif args.mode == 'test':  # test
        print("============== Starting Testing ==============")
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(network, param_dict)
        ds_eval = create_dataset(os.path.join(args.data_path, "test"), 32, 1)
        acc = model.eval(ds_eval, dataset_sink_mode=args.dataset_sink_mode)
        print("============== Accuracy:{} ==============".format(acc))
    else:
        raise RuntimeError('mode should be train or test, rather than {}'.format(args.mode))
