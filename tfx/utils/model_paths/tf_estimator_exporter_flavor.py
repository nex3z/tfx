# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modules for TensorFlow Estimator flavor model path.

TensorFlow Estimator
[Exporter](https://www.tensorflow.org/api_docs/python/tf/estimator/Exporter)
export the model under `{export_path}/export/{exporter_name}/{timestamp}`
directory. We call this a *TF-Estimator-flavored model path*.

Example:

```
gs://your_bucket_name/foo/bar/  # An `export_path`
  export/                       # Constant name "export"
    my_exporter/                # An `exporter_name`
      1582072718/               # UTC `timestamp` in seconds
        (Your exported SavedModel)
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List, Text

import tensorflow as tf

EXPORT_SUB_DIR_NAME = 'export'


def lookup_model_paths(export_path: Text) -> List[Text]:
  """Lookup all model paths in an export path.

  Args:
    export_path: An export_path as defined from the module docstring.

  Returns:
    A list of model_path.
  """
  export_sub_dir = os.path.join(export_path, EXPORT_SUB_DIR_NAME)
  result = []
  if not tf.io.gfile.isdir(export_sub_dir):
    return result
  for exporter_name in tf.io.gfile.listdir(export_sub_dir):
    model_sub_dir = os.path.join(export_sub_dir, exporter_name)
    if not tf.io.gfile.isdir(model_sub_dir):
      continue
    for timestamp in tf.io.gfile.listdir(model_sub_dir):
      if not timestamp.isdigit():
        continue
      model_path = os.path.join(model_sub_dir, timestamp)
      if tf.io.gfile.isdir(model_path):
        result.append(model_path)

  return result
