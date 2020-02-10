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
"""Tests for tfx.orchestration.data_types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from typing import Dict, List, Text
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from google.protobuf.json_format import ParseError
from tfx.orchestration import data_types
from tfx.proto import example_gen_pb2
from tfx.types.artifact import Artifact
from tfx.types.channel import Channel
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter


class _InputArtifact(Artifact):
  TYPE_NAME = 'InputArtifact'


class _OutputArtifact(Artifact):
  TYPE_NAME = 'OutputArtifact'


class _BasicComponentSpec(ComponentSpec):

  PARAMETERS = {
      'folds': ExecutionParameter(type=int),
      'proto': ExecutionParameter(type=example_gen_pb2.Input, optional=True),
  }
  INPUTS = {
      'input': ChannelParameter(type=_InputArtifact),
  }
  OUTPUTS = {
      'output': ChannelParameter(type=_OutputArtifact),
  }
  _INPUT_COMPATIBILITY_ALIASES = {
      'future_input_name': 'input',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {
      'future_output_name': 'output',
  }


class DataTypesTest(tf.test.TestCase):

  def testComponentSpecWithRuntimeParam(self):
    param = data_types.RuntimeParameter(name='split-1', ptype=Text)
    serialized_param = str(param)
    # Dict representation of a example_gen_pb2.Input proto message.
    proto = dict(splits=[
        dict(name=param, pattern='pattern1'),
        dict(name='name2', pattern='pattern2'),
        dict(name='name3', pattern='pattern3'),
    ])
    input_channel = Channel(type=_InputArtifact)
    output_channel = Channel(type=_OutputArtifact)
    spec = _BasicComponentSpec(
        folds=10, proto=proto, input=input_channel, output=output_channel)
    # Verify proto property.
    self.assertIsInstance(spec.exec_properties['proto'], str)
    decoded_proto = json.loads(spec.exec_properties['proto'])
    self.assertCountEqual(['splits'], decoded_proto.keys())
    self.assertEqual(3, len(decoded_proto['splits']))
    self.assertCountEqual([serialized_param, 'name2', 'name3'],
                          list(s['name'] for s in decoded_proto['splits']))
    self.assertCountEqual(['pattern1', 'pattern2', 'pattern3'],
                          list(s['pattern'] for s in decoded_proto['splits']))

  def testProtoTypeCheck(self):
    param = data_types.RuntimeParameter(name='split-1', ptype=Text)
    # Dict representation of a example_gen_pb2.Input proto message.
    # The second split has int-typed pattern, which is wrong.
    proto = dict(splits=[
        dict(name=param, pattern='pattern1'),
        dict(name='name2', pattern=42),
        dict(name='name3', pattern='pattern3'),
    ])
    input_channel = Channel(type=_InputArtifact)
    output_channel = Channel(type=_OutputArtifact)

    with self.assertRaisesRegexp(
        ParseError, 'Failed to parse .* field: expected string or '
        '(bytes-like object|buffer)'):
      _ = _BasicComponentSpec(
          folds=10, proto=proto, input=input_channel, output=output_channel)

  def testTypeCheckWithRuntimeParameter(self):

    class SimpleComponentSpec(ComponentSpec):
      INPUTS = {}
      OUTPUTS = {}
      PARAMETERS = {
          'x': ExecutionParameter(type=int),
          'y': ExecutionParameter(type=int, optional=True),
      }

    parameter_int = data_types.RuntimeParameter(name='int', ptype=int)
    parameter_str = data_types.RuntimeParameter(name='str', ptype=Text)

    _ = SimpleComponentSpec(x=parameter_int)
    with self.assertRaisesRegexp(TypeError, 'Expected type'):
      _ = SimpleComponentSpec(x=42, y=parameter_str)

    class ComponentSpecWithContainer(ComponentSpec):
      INPUTS = {}
      OUTPUTS = {}
      PARAMETERS = {
          'x': ExecutionParameter(type=Dict[Text, Text]),
          'y': ExecutionParameter(type=List[int]),
      }

    _ = ComponentSpecWithContainer(x={u'key': parameter_str}, y=[parameter_int])
    with self.assertRaisesRegexp(TypeError, 'Expecting value type'):
      _ = ComponentSpecWithContainer(x={u'key': parameter_int}, y=[])


if __name__ == '__main__':
  tf.test.main()
