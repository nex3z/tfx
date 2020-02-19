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
"""Tests for tfx.components.infra_validator.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import mock
import tensorflow as tf
from typing import Any, Dict, Text

from google.protobuf import json_format
from tfx.components.infra_validator import executor
from tfx.components.infra_validator import request_builder
from tfx.components.infra_validator import serving_binary_lib
from tfx.proto import infra_validator_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


def _make_serving_spec(
    payload: Dict[Text, Any]) -> infra_validator_pb2.ServingSpec:
  result = infra_validator_pb2.ServingSpec()
  json_format.ParseDict(payload, result)
  return result


def _make_validation_spec(
    payload: Dict[Text, Any]) -> infra_validator_pb2.ValidationSpec:
  result = infra_validator_pb2.ValidationSpec()
  json_format.ParseDict(payload, result)
  return result


def _make_request_spec(
    payload: Dict[Text, Any]) -> infra_validator_pb2.RequestSpec:
  result = infra_validator_pb2.RequestSpec()
  json_format.ParseDict(payload, result)
  return result


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    # Setup Mocks

    patcher = mock.patch.object(request_builder, 'build_requests')
    self.build_requests_mock = patcher.start()
    self.addCleanup(patcher.stop)

    # Setup directories

    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    base_output_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                                     self.get_temp_dir())
    output_data_dir = os.path.join(base_output_dir, self._testMethodName)

    # Setup input_dict.

    self._model = standard_artifacts.Model()
    self._model.uri = os.path.join(source_data_dir, 'trainer', 'current')
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'transform',
                                'transformed_examples', 'eval')
    examples.split_names = artifact_utils.encode_split_names(['eval'])

    self._input_dict = {
        'model': [self._model],
        'examples': [examples],
    }
    self._blessing = standard_artifacts.InfraBlessing()
    self._blessing.uri = os.path.join(output_data_dir, 'blessing')
    self._output_dict = {'blessing': [self._blessing]}
    temp_dir = os.path.join(output_data_dir, '.temp')
    self._context = executor.Executor.Context(tmp_dir=temp_dir, unique_id='1')
    self._serving_spec = _make_serving_spec({
        'tensorflow_serving': {
            'model_name': 'chicago-taxi',
            'tags': ['1.15.0']
        },
        'local_docker': {}
    })
    self._serving_binary = serving_binary_lib.parse_serving_binaries(
        self._serving_spec)[0]
    self._validation_spec = _make_validation_spec({
        'max_loading_time_seconds': 10,
        'num_tries': 3
    })
    self._request_spec = _make_request_spec({
        'tensorflow_serving': {
            'rpc_kind': 'CLASSIFY'
        },
        'max_examples': 1
    })
    self._exec_properties = {
        'serving_spec': json_format.MessageToJson(self._serving_spec),
        'validation_spec': json_format.MessageToJson(self._validation_spec),
        'request_spec': json_format.MessageToJson(self._request_spec),
    }

  def testDo_BlessedIfNoError(self):
    # Run executor.
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(infra_validator, '_ValidateOnce'):
      infra_validator.Do(self._input_dict, self._output_dict,
                         self._exec_properties)

    # Check blessed.
    self.assertBlessed()

  def testDo_NotBlessedIfError(self):
    # Run executor.
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(infra_validator, '_ValidateOnce') as validate_mock:
      # Validation will raise error.
      validate_mock.side_effect = ValueError
      infra_validator.Do(self._input_dict, self._output_dict,
                         self._exec_properties)

    # Check not blessed.
    self.assertNotBlessed()

  def testDo_BlessedIfEventuallyNoError(self):
    # Run executor.
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(infra_validator, '_ValidateOnce') as validate_mock:
      # Validation will raise error at first, succeeded at the following.
      # Infra validation will be tried 3 times, so 2 failures are tolerable.
      validate_mock.side_effect = [ValueError, ValueError, None]
      infra_validator.Do(self._input_dict, self._output_dict,
                         self._exec_properties)

    # Check blessed.
    self.assertBlessed()

  def testDo_NotBlessedIfErrorContinues(self):
    # Run executor.
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(infra_validator, '_ValidateOnce') as validate_mock:
      # 3 Errors are not tolerable.
      validate_mock.side_effect = [ValueError, ValueError, ValueError, None]
      infra_validator.Do(self._input_dict, self._output_dict,
                         self._exec_properties)

    # Check not blessed.
    self.assertNotBlessed()

  def testValidateOnce_LoadOnly_Succeed(self):
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(self._serving_binary, 'MakeClient'):
      with mock.patch.object(executor, '_create_model_server_runner'):
        try:
          infra_validator._ValidateOnce(
              model=self._model,
              serving_binary=self._serving_binary,
              serving_spec=self._serving_spec,
              validation_spec=self._validation_spec,
              requests=[])
        except Exception:  # pylint: disable=broad-except
          self.fail()

  def testValidateOnce_LoadOnly_FailIfRunnerWaitRaises(self):
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(self._serving_binary, 'MakeClient'):
      with mock.patch.object(
          executor, '_create_model_server_runner') as mock_runner_factory:
        mock_runner = mock_runner_factory.return_value
        mock_runner.WaitUntilRunning.side_effect = ValueError
        with self.assertRaises(ValueError):
          infra_validator._ValidateOnce(
              model=self._model,
              serving_binary=self._serving_binary,
              serving_spec=self._serving_spec,
              validation_spec=self._validation_spec,
              requests=[])

  def testValidateOnce_LoadOnly_FailIfClientWaitRaises(self):
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(self._serving_binary,
                           'MakeClient') as mock_client_factory:
      mock_client = mock_client_factory.return_value
      with mock.patch.object(
          executor, '_create_model_server_runner') as mock_runner_factory:
        mock_client.WaitUntilModelLoaded.side_effect = ValueError
        with self.assertRaises(ValueError):
          infra_validator._ValidateOnce(
              model=self._model,
              serving_binary=self._serving_binary,
              serving_spec=self._serving_spec,
              validation_spec=self._validation_spec,
              requests=[])
        mock_runner_factory.return_value.WaitUntilRunning.assert_called()

  def testValidateOnce_LoadAndQuery_Succeed(self):
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(self._serving_binary,
                           'MakeClient') as mock_client_factory:
      mock_client = mock_client_factory.return_value
      with mock.patch.object(
          executor, '_create_model_server_runner') as mock_runner_factory:
        try:
          infra_validator._ValidateOnce(
              model=self._model,
              serving_binary=self._serving_binary,
              serving_spec=self._serving_spec,
              validation_spec=self._validation_spec,
              requests=['my_request'])
        except Exception:  # pylint: disable=broad-except
          self.fail()
        mock_runner_factory.return_value.WaitUntilRunning.assert_called()
        mock_client.WaitUntilModelLoaded.assert_called()
        mock_client.SendRequests.assert_called()

  def testValidateOnce_LoadAndQuery_FailIfSendRequestsRaises(self):
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(self._serving_binary,
                           'MakeClient') as mock_client_factory:
      mock_client = mock_client_factory.return_value
      with mock.patch.object(
          executor, '_create_model_server_runner') as mock_runner_factory:
        mock_client.SendRequests.side_effect = ValueError
        with self.assertRaises(ValueError):
          infra_validator._ValidateOnce(
              model=self._model,
              serving_binary=self._serving_binary,
              serving_spec=self._serving_spec,
              validation_spec=self._validation_spec,
              requests=['my_request'])
        mock_runner_factory.return_value.WaitUntilRunning.assert_called()
        mock_client.WaitUntilModelLoaded.assert_called()

  def assertBlessed(self):
    self.assertFileExists(os.path.join(self._blessing.uri, 'INFRA_BLESSED'))
    self.assertEqual(1, self._blessing.get_int_custom_property('blessed'))

  def assertNotBlessed(self):
    self.assertFileExists(os.path.join(self._blessing.uri, 'INFRA_NOT_BLESSED'))
    self.assertEqual(0, self._blessing.get_int_custom_property('blessed'))

  def assertFileExists(self, path: Text):
    self.assertTrue(tf.io.gfile.exists(path))


if __name__ == '__main__':
  tf.test.main()
