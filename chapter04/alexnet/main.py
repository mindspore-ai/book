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
AlexNet example tutorial
Usage:
     python alexnet.py
with --device_target=GPU: After 20 epoch training, the accuracy is up to 80%
with --device_target=Ascend: After 10 epoch training, the accuracy is up to 81%
"""

import argparse
from config import alexnet_cfg as cfg
from alexnet import AlexNet
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.vision.c_transforms as CV
from mindspore.nn.metrics import Accuracy
from mindspore.common import dtype as mstype


def create_dataset(data_path, batch_size=32, repeat_size=1, mode="train"):
    """
    create dataset for train or test
    """
    cifar_ds = ds.Cifar10Dataset(data_path)
    rescale = 1.0 / 255.0
    shift = 0.0

    resize_op = CV.Resize((cfg.image_height, cfg.image_width))
    rescale_op = CV.Rescale(rescale, shift)
    normalize_op = CV.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if mode == "train":
        random_crop_op = CV.RandomCrop([32, 32], [4, 4, 4, 4])
        random_horizontal_op = CV.RandomHorizontalFlip()
    channel_swap_op = CV.HWC2CHW()
    typecast_op = C.TypeCast(mstype.int32)
    cifar_ds = cifar_ds.map(input_columns="label", operations=typecast_op)
    if mode == "train":
        cifar_ds = cifar_ds.map(input_columns="image", operations=random_crop_op)
        cifar_ds = cifar_ds.map(input_columns="image", operations=random_horizontal_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=resize_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=rescale_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=normalize_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=channel_swap_op)

    cifar_ds = cifar_ds.shuffle(buffer_size=cfg.buffer_size)
    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True)
    cifar_ds = cifar_ds.repeat(repeat_size)
    return cifar_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore AlexNet Example')
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'test'],
                        help='implement phase, set to train or test')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="./", help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if mode is test, must provide\
                        path where the trained ckpt file')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='dataset_sink_mode is False or True')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    network = AlexNet(cfg.num_classes)
    loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
    repeat_size = cfg.epoch_size
    opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
    model = Model(network, loss, opt, metrics={"Accuracy": Accuracy()})  # test

    if args.mode == 'train':
        print("============== Starting Training ==============")
        ds_train = create_dataset(args.data_path,
                                  cfg.batch_size,
                                  repeat_size,
                                  args.mode)
        config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                     keep_checkpoint_max=cfg.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix="checkpoint_alexnet",  directory=args.ckpt_path, config=config_ck)
        model.train(cfg.epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()],
                    dataset_sink_mode=args.dataset_sink_mode)
    elif args.mode == 'test':
        print("============== Starting Testing ==============")
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(network, param_dict)
        ds_eval = create_dataset(args.data_path, mode=args.mode)
        acc = model.eval(ds_eval, dataset_sink_mode=args.dataset_sink_mode)
        print("============== Accuracy:{} ==============".format(acc))
    else:
        raise RuntimeError('mode should be train or test, rather than {}'.format(args.mode))
