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
"""Module for TFMA EvalSavedModel flavor model path.

TensorFlow Model Analysis (TFMA) export a model's evaluation graph to a special
[EvalSavedModel](https://www.tensorflow.org/tfx/model_analysis/eval_saved_model)
format under the directory {export_dir_base}/{timestamp}. We call this a
*TFMA-EvalSavedModel-flavored model path*.

Example:

```
gs://your_bucket_name/eval/   # An `export_dir_base`
  1582072718/                 # UTC `timestamp` in seconds
    (Your exported EvalSavedModel)
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List, Text

import tensorflow as tf


def lookup_model_paths(export_dir_base: Text) -> List[Text]:
  """Lookup all model paths in an export_dir_base.

  Args:
    export_dir_base: An export_dir_base as defined from the module docstring.

  Returns:
    A list of model_path.
  """
  result = []
  if not tf.io.gfile.isdir(export_dir_base):
    return result
  for timestamp in tf.io.gfile.listdir(export_dir_base):
    if not timestamp.isdigit():
      continue
    model_path = os.path.join(export_dir_base, timestamp)
    if tf.io.gfile.isdir(model_path):
      result.append(model_path)
  return result
