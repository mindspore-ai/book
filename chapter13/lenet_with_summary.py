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

import os
import argparse

from mindspore import dataset
from mindspore import nn, context, Tensor, Model
import mindspore.ops as ops
from mindspore.dataset.vision import c_transforms as transforms
from mindspore.dataset.vision import Inter
from mindspore.dataset.transforms import c_transforms as C
from mindspore import dtype as mstype
from mindspore.train.callback import SummaryCollector

from .lenet import LeNet5


class CrossEntropyLoss(nn.Cell):
    """Define loss function for network."""
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.sm_scalar = ops.ScalarSummary()
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()
        self.mean = ops.ReduceMean()
        self.one_hot = ops.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

    def construct(self, logits, label):
        label = self.one_hot(label, ops.shape(logits)[1], self.on_value, self.off_value)
        loss = self.cross_entropy(logits, label)[0]
        loss = self.mean(loss, (-1,))
        self.sm_scalar("loss", loss)
        return loss


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1):
    """Create dataset for train or test."""
    # define dataset
    mnist_ds = dataset.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    resize_op = transforms.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_op = transforms.Rescale(rescale, shift)
    hwc2chw_op = transforms.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


def main(data_path, device_target='Ascend', summary_dir='./summary_dir', learning_rate=0.01):
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)

    momentum = 0.9
    epoch_size = 1
    batch_size = 32

    network = LeNet5()
    network.set_train()
    net_loss = CrossEntropyLoss()
    net_opt = nn.Momentum(network.trainable_params(), learning_rate, momentum)
    model = Model(network, net_loss, net_opt)

    # Init SummaryCollector callback to record summary data in model.train or model.eval
    summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=10)

    ds = create_dataset(os.path.join(data_path, "train"), batch_size=batch_size)

    print("============== Starting Training ==============")
    model.train(epoch_size, ds, callbacks=[summary_collector], dataset_sink_mode=False)
    print("============== Train End =====================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore LeNet5 with summary Example')
    parser.add_argument('--device_target', type=str, default="Ascend",
                        choices=['Ascend', 'GPU', 'CPU'],
                        help='Device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="./MNIST_Data",
                        help='Path where the dataset is saved')
    parser.add_argument('--summary_dir', type=str, default='./summary_dir',
                        help='Summaries log directory.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')

    args = parser.parse_args()

    main(data_path=args.data_path,
         device_target=args.device_target,
         summary_dir=args.summary_dir,
         learning_rate=args.learning_rate)
