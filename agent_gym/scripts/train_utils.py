# import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm

from stable_baselines3.common.callbacks import BaseCallback

from typing import Callable


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - 1e-4) + 1e-4

    return func


class CallbackEpisodeMetrics(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, model_save_path, model_save_name, model_save_freq=1000, num_cpu=1, verbose=0):
        super(CallbackEpisodeMetrics, self).__init__(verbose)
        self.model_save_path = model_save_path
        self.model_save_name = model_save_name
        self.model_save_freq = model_save_freq
        self.pbar = None
        self.step_count = 0
        self.num_cpu = num_cpu

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)

        for info in self.locals['infos']:
            if len(info) > 1:
                for key in info.keys():
                    self.logger.record(key, info[key])

        if self.pbar is None:
            self.pbar = tqdm(total=self.locals['total_timesteps'])

        self.step_count += 1
        self.pbar.update(self.num_cpu)

        if self.step_count % (self.model_save_freq // self.num_cpu) == 0:
            self.model.save(self.model_save_path + '/' + self.model_save_name + '_at_step_{}'.format(
                self.step_count * self.num_cpu))
        return True

    def _on_training_end(self) -> None:
        self.model.save(self.model_save_path + '/' + self.model_save_name + '_at_the_end')


import os


def create_folder(path):
    print("searching folder from path:\t", path)
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print('folder not founded, new folder created')
        return True
    else:
        print('folder exists！')
        return False
