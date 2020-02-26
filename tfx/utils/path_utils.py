# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Utilities for retrieving paths for various types of artifacts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from typing import List, Text

from tfx.utils.model_paths import tf_estimator_exporter_flavor
from tfx.utils.model_paths import tfma_eval_saved_model_flavor

EVAL_MODEL_DIR = 'eval_model_dir'
SERVING_MODEL_DIR = 'serving_model_dir'


def _only_path(paths: List[Text]) -> Text:
  assert len(paths) == 1
  return paths[0]


# TODO(b/127149760): simplify this PPP-esque structure.
#
# Directory structure of exported model for estimator based trainer:
#   |-- <ModelExportPath>
#       |-- EVAL_MODEL_DIR  <- eval_model_dir
#           |-- <timestamped model>  <- eval_model_path
#               |-- saved_model.pb
#               |-- ...
#       |-- SERVING_MODEL_DIR  <- serving_model_dir
#           |-- export
#               |-- <exporter name>
#                   |-- <timestamped model>  <- serving_model_path
#                       |-- saved_model.pb
#                       |-- ...
#           |-- ...
#
# For generic trainer with Keras, there won't be eval model:
#   |-- <ModelExportPath>
#       |-- SERVING_MODEL_DIR  <- serving_model_dir
#           |-- saved_model.pb
#           |-- ...


def eval_model_dir(output_uri: Text) -> Text:
  """Returns directory for exported model for evaluation purpose."""
  return os.path.join(output_uri, EVAL_MODEL_DIR)


def eval_model_path(output_uri: Text) -> Text:
  """Returns path to timestamped exported model for evaluation purpose."""
  model_dir = eval_model_dir(output_uri)
  paths = tfma_eval_saved_model_flavor.lookup_model_paths(
      export_dir_base=model_dir)
  if not paths:
    # If eval model doesn't exist, use serving model for eval.
    return serving_model_path(output_uri)
  return _only_path(paths)


def serving_model_dir(output_uri: Text) -> Text:
  """Returns directory for exported model for serving purpose."""
  return os.path.join(output_uri, SERVING_MODEL_DIR)


def serving_model_path(output_uri: Text) -> Text:
  """Returns path for timestamped and named serving model exported."""
  model_dir = serving_model_dir(output_uri)
  paths = tf_estimator_exporter_flavor.lookup_model_paths(export_path=model_dir)
  if not paths:
    # If dir doesn't match estimator structure, use serving model root directly.
    return model_dir
  return _only_path(paths)


def get_serving_model_version(output_uri: Text) -> Text:
  """Returns version of the serving model."""
  # Note: TFX doesn't have a logical model version right now, use timestamp as
  # version. For estimator serving model, directly use the timestamp name of the
  # exported folder. For keras serving model, returns the current timestamp.
  version = os.path.basename(serving_model_path(output_uri))
  if not version.isdigit():
    version = str(int(time.time()))
  return version
