# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
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

"""Default logger."""

import logging
from typing import Any, Callable, Mapping, Optional
import nanoid

from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal


class WandBLogger(base.Logger):
    """Weights & Biases logger."""

    def __init__(self, wandb) -> None:
        self.wandb = wandb
        super().__init__()

    def write(self, data: base.LoggingData):
        self.wandb.log(data)

    def close(self):
        self.wandb.finish()

    def flush(self):
        pass


def make_default_logger(
    label: str,
    save_data: bool = True,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
    steps_key: str = "steps",
    wandb_project: Optional[bool] = False,
    config: dict = None,
    identity: str = "",
    task_name: str = "",
) -> base.Logger:
    """Makes a default Acme logger.

    Args:
      label: Name to give to the logger.
      save_data: Whether to persist data.
      time_delta: Time (in seconds) between logging events.
      asynchronous: Whether the write function should block or not.
      print_fn: How to print to terminal (defaults to print).
      serialize_fn: An optional function to apply to the write inputs before
        passing them to the various loggers.
      steps_key: Ignored.
      identity: the identity of the logger, either evaluator or learner
      task_name: specify the task of logger, useful in training a generalist model,
        when evaluating the performance of the model under multiple environments.

    Returns:
      A logger object that responds to logger.write(some_dict).
    """
    del steps_key
    if not print_fn:
        print_fn = logging.info
    terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

    loggers = [terminal_logger]

    if save_data:
        loggers.append(csv.CSVLogger(label=label))

    if wandb_project:
        # initialize wandb logging
        trail_name = f"{config['run_config']['run_name']}-{identity}"
        if task_name != "":
            trail_name += f"-{task_name}"
        wandb = setup_wandb(
            config=config,
            project="rodent-four-tasks",
            rank_zero_only=False,
            trial_name=trail_name,
            trial_id=nanoid.generate(),
        )  # with unique uuid
        loggers.append(WandBLogger(wandb=wandb))

    # Dispatch to all writers and filter Nones and by time.
    logger = aggregators.Dispatcher(loggers, serialize_fn)
    logger = filters.NoneFilter(logger)
    if asynchronous:
        logger = async_logger.AsyncLogger(logger)
    logger = filters.TimeFilter(logger, time_delta)

    return logger
