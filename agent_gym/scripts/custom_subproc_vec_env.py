import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np
import torch

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)


def _worker(
        remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            ################## my code
            elif cmd == "sample_action":
                sampled_action = env.sample_action()
                remote.send(sampled_action)

            elif cmd == "update_prediction":
                env.update_prediction(data)
            #################
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class CustomSubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.
    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.
    .. warning::
        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.
    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        ####!!!!!!!!!!!!!!!!!!!!!!!!!!!! my code  ##############################################3
        self.prediction_model = None
        self.prediction_model_loaded = False

    ####!!!!!!!!!!!!!!!!!!!!!!!!!!!! my func  ##############################################3
    def sample_action(self):
        for remote in self.remotes:
            remote.send(("sample_action", None))
        acts = [remote.recv() for remote in self.remotes]

        return np.stack(acts)

    def load_prediction_model(self, model, input_type=None, output_type=None):
        self.prediction_model = model
        assert input_type == self.prediction_model.obs_type
        assert output_type == self.prediction_model.cost_type

        input_features, cost_features, mask_features = self.prediction_model.get_input_and_output()

        self.input_shape = len(input_features)
        self.cost_shape = len(cost_features)
        self.mask_shape = len(mask_features)

        self.prediction_model_loaded = True
        return self.prediction_model_loaded

    def predict_cost_and_mask(self, obs):
        assert self.prediction_model_loaded

        prediction_inputs = obs["prediction_inputs"]
        unpred_cost = obs["coop_edge_cost"]
        unpred_mask = obs["coop_edge_mask"]

        prediction_inputs_shape = prediction_inputs.shape
        #
        assert prediction_inputs_shape[-1] == self.input_shape

        X = torch.tensor(prediction_inputs).to(self.prediction_model.device)
        pred_cost = self.prediction_model.predict_cost(X).cpu().detach()  # .numpy()
        pred_mask = self.prediction_model.predict_mask(X).cpu().detach()  # .numpy()
        pred_mask[:, :, -1] = 1

        # pred_cost = torch.reshape(pred_cost, pred_cost.shape[:-1])

        pred_cost = torch.sum(pred_cost, 1).numpy().reshape(unpred_cost.shape)
        pred_mask = torch.multiply(pred_mask[:, 0, :], pred_mask[:, 1, :]).numpy().reshape(unpred_mask.shape)

        # print("pred_cost.shape", pred_cost.shape)
        # print("pred_mask.shape", pred_mask.shape)

        obs["coop_edge_mask"] = np.multiply(pred_mask, unpred_mask)
        obs["coop_edge_cost"] = np.multiply(pred_cost, obs["coop_edge_mask"]) + np.multiply(unpred_cost,
                                                                                            1 - obs["coop_edge_mask"])

        return obs

    def update_cost_and_mask(self, obs):
        for remote,cost,mask in zip(self.remotes, obs['coop_edge_cost'], obs['coop_edge_mask']):
            remote.send(("update_prediction", [cost,mask]))
        return True

    ####!!!!!!!!!!!!!!!!!!!!!!!!!!! my func end ###############################################
    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        ######## my code ######
        obs = _flatten_obs(obs, self.observation_space)
        obs = self.predict_cost_and_mask(obs)
        self.update_cost_and_mask(obs)

        #######################3
        return obs, np.stack(rews), np.stack(dones), infos

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            seed = np.random.randint(0, 2 ** 32 - 1)
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self) -> VecEnvObs:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]

        ######## my code ######
        obs = _flatten_obs(obs, self.observation_space)
        obs = self.predict_cost_and_mask(obs)
        self.update_cost_and_mask(obs)

        #######################3

        return obs

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.
        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: gym.spaces.Space) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.
    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))
    else:
        return np.stack(obs)
