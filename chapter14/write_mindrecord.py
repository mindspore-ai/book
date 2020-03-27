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

import os
import uuid
from mindspore.mindrecord import FileWriter, FileReader
from utils import get_data

MINDRECORD_FILE_NAME = "./imagenet.mindrecord"

def write_mindrecord_tutorial():
    writer = FileWriter(MINDRECORD_FILE_NAME)
    data = get_data("./ImageNetDataSimulation")
    schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"},
                      "data": {"type": "bytes"}}
    writer.add_schema(schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(MINDRECORD_FILE_NAME)
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        # print("#item {}: {}".format(index, x))
    assert count == 20
    reader.close()

if __name__ == '__main__':
    write_mindrecord_tutorial()

    os.remove(MINDRECORD_FILE_NAME)
    os.remove(MINDRECORD_FILE_NAME + ".db")
