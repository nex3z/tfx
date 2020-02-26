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
"""Tests for tfx.utils.model_paths.tfma_eval_saved_model_flavor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from typing import Text

from tfx.utils.model_paths import tfma_eval_saved_model_flavor


def _make_model_path(export_dir_base: Text, timestamp: int) -> Text:
  return os.path.join(export_dir_base, str(timestamp))


class TFMAEvalSavedModelFlavorTest(tf.test.TestCase):

  def _SetupSingleModel(self):
    export_dir_base = self.get_temp_dir()
    tf.io.gfile.makedirs(
        _make_model_path(
            export_dir_base=export_dir_base,
            timestamp=1582798459))
    return export_dir_base

  def _SetupMultipleModels(self):
    export_dir_base = self.get_temp_dir()
    tf.io.gfile.makedirs(
        _make_model_path(
            export_dir_base=export_dir_base,
            timestamp=1582798459))
    tf.io.gfile.makedirs(
        _make_model_path(
            export_dir_base=export_dir_base,
            timestamp=1582858365))
    return export_dir_base

  def testLookupModelPaths_ForSingleModel(self):
    export_dir_base = self._SetupSingleModel()

    model_paths = tfma_eval_saved_model_flavor.lookup_model_paths(
        export_dir_base=export_dir_base)

    self.assertEqual(len(model_paths), 1)
    self.assertEqual(os.path.relpath(model_paths[0], export_dir_base),
                     '1582798459')

  def testLookupModelPaths_ForMultipleModels(self):
    export_dir_base = self._SetupMultipleModels()

    model_paths = tfma_eval_saved_model_flavor.lookup_model_paths(
        export_dir_base=export_dir_base)

    self.assertEqual(len(model_paths), 2)
    mp1, mp2 = sorted(model_paths)
    self.assertEqual(os.path.relpath(mp1, export_dir_base), '1582798459')
    self.assertEqual(os.path.relpath(mp2, export_dir_base), '1582858365')

  def testLookupModelPaths_InvalidPattern(self):
    export_dir_base = self.get_temp_dir()
    tf.io.gfile.makedirs(os.path.join(export_dir_base, 'foo'))

    result = tfma_eval_saved_model_flavor.lookup_model_paths(
        export_dir_base=export_dir_base)

    self.assertEqual(len(result), 0)


if __name__ == '__main__':
  tf.test.main()
