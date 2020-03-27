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
from mindspore.mindrecord import MindPage, SUCCESS

from write_mindrecord import write_mindrecord_tutorial

MINDRECORD_FILE_NAME = "./imagenet.mindrecord"

def search_mindrecord_tutorial():
    reader = MindPage(MINDRECORD_FILE_NAME)
    fields = reader.get_category_fields()
    assert fields == ['file_name', 'label'], \
        'failed on getting candidate category fields.'

    ret = reader.set_category_field("label")
    assert ret == SUCCESS, 'failed on setting category field.'

    info = reader.read_category_info()
    # print("category info: {}".format(info))

    row = reader.read_at_page_by_id(0, 0, 1)
    assert len(row) == 1
    assert len(row[0]) == 3
    # print("row[0]: {}".format(row[0]))

    row1 = reader.read_at_page_by_name("2", 0, 2)
    assert len(row1) == 2
    assert len(row1[0]) == 3
    # print("row1[0]: {}".format(row1[0]))
    # print("row1[1]: {}".format(row1[1]))

if __name__ == '__main__':
    write_mindrecord_tutorial()

    search_mindrecord_tutorial()

    os.remove(MINDRECORD_FILE_NAME)
    os.remove(MINDRECORD_FILE_NAME + ".db")
