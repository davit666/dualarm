import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np
import torch

from load_pseudo_task_data import extract_pseudo_task_data

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
            elif cmd == "reset_with_task_data":
                observation = env.reset(data)
                remote.send(observation)
            elif cmd == "sample_action":
                sampled_action = env.sample_action()
                remote.send(sampled_action)

            elif cmd == "update_prediction":
                obs = env.update_prediction(data)
                remote.send(obs)
            elif cmd == "get_data_for_offline_planning":
                c, m, p = env.get_data_for_offline_planning()
                d = {}
                d['c'] = c
                d['m'] = m
                d['p'] = p
                remote.send(d)
            elif cmd == "get_current_task_data":
                data = env.get_current_task_data()
                remote.send(data)
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

        self.use_prediction_model = True

    ####!!!!!!!!!!!!!!!!!!!!!!!!!!!! my func  ##############################################3
    def sample_action(self):
        for remote in self.remotes:
            remote.send(("sample_action", None))
        acts = [remote.recv() for remote in self.remotes]

        return np.stack(acts)

    def load_prediction_model(self, model, input_type=None, output_type=None, use_prediction_model=True, predict_content = "both"):
        self.predict_content = predict_content
        self.use_prediction_model = use_prediction_model
        if use_prediction_model:
            self.prediction_model_loaded = True
            self.prediction_model = model
            assert input_type == self.prediction_model.obs_type
            assert output_type == self.prediction_model.cost_type

            input_features, cost_features, mask_features = self.prediction_model.get_input_and_output()

            self.input_shape = len(input_features)
            self.cost_shape = len(cost_features)
            self.mask_shape = len(mask_features)

        return self.prediction_model_loaded

    def predict_cost_and_mask(self, obs):
        assert self.prediction_model_loaded

        prediction_inputs = obs["prediction_inputs"]
        unpred_cost = obs["coop_edges"]
        unpred_mask = obs["coop_edge_mask"]
        edge_shape = unpred_cost.shape[:-1]

        prediction_inputs_shape = prediction_inputs.shape
        #
        assert prediction_inputs_shape[-1] == self.input_shape

        X = torch.tensor(prediction_inputs).to(self.prediction_model.device)
        pred_cost = self.prediction_model.predict_cost(X).cpu().detach()  # .numpy()
        pred_mask = self.prediction_model.predict_mask(X).cpu().detach()  # .numpy()
        pred_mask[:, :, -1] = 1

        if self.predict_content in ["cost_only", "only_cost", "cost"]:
            # do not update mask
            pred_mask1 = unpred_mask.copy()
            pred_mask2 = unpred_mask.copy()
            pred_mask0 = unpred_mask.copy()
        else:
            # mask update
            pred_mask1 = pred_mask[:, 0, :].reshape(edge_shape).numpy().astype(np.float32)
            pred_mask2 = pred_mask[:, 1, :].reshape(edge_shape).numpy().astype(np.float32)
            pred_mask0 = torch.multiply(pred_mask[:, 0, :], pred_mask[:, 1, :]).reshape(edge_shape).numpy().astype(
                np.float32)

            pred_mask1 = np.multiply(pred_mask1, unpred_mask)
            pred_mask2 = np.multiply(pred_mask2, unpred_mask)
            pred_mask0 = np.multiply(pred_mask0, unpred_mask)

            mask_terminate = pred_mask0[:, :-1, :-1].sum(axis=-2).sum(axis=-1)
            mask_terminate = mask_terminate < 1

            pred_mask1[:, -1, -1] = mask_terminate
            pred_mask2[:, -1, -1] = mask_terminate
            pred_mask0[:, -1, -1] = mask_terminate

        obs["coop_edge_mask"] = pred_mask0
        # print("m1", pred_mask1)
        # print("m2", pred_mask2)
        # print("m0", pred_mask0)


        if self.predict_content in ["mask_only", "only_mask", "mask"]:
            # donot update cost
            pred_cost1 = unpred_cost[:, :, :, 1]
            pred_cost2 = unpred_cost[:, :, :, 2]
            pred_cost0 = unpred_cost[:, :, :, 0]
        else:
            # cost update
            pred_cost1 = pred_cost[:, 0, :].reshape(edge_shape).numpy().astype(np.float32) / 500
            pred_cost2 = pred_cost[:, 1, :].reshape(edge_shape).numpy().astype(np.float32) / 500
            pred_cost2[:,-1,-1] = 0
            pred_cost0 = (pred_cost1 + pred_cost2) / 2

        pred_cost1 = np.multiply(pred_cost1, pred_mask1) + np.multiply(np.ones(edge_shape), 1 - pred_mask1)
        pred_cost2 = np.multiply(pred_cost2, pred_mask2) + np.multiply(np.ones(edge_shape), 1 - pred_mask2)
        pred_cost0 = np.multiply(pred_cost0, pred_mask0) + np.multiply(np.ones(edge_shape), 1 - pred_mask0)
        # print(pred_cost2)
        ############################  experiment: coop edge
        pred_cost = np.stack([pred_cost0, pred_cost1, pred_cost2,  pred_mask0, pred_mask1, pred_mask2], axis=-1)
        # pred_cost = np.stack([pred_cost0, pred_cost1, pred_cost2], axis=-1)
        # pred_cost = np.stack([pred_cost0, pred_mask0], axis=-1)
        # pred_cost = np.stack([pred_cost0], axis=-1)
        ############################  experiment: coop edge
        # print(pred_mask0, pred_cost)
        obs["coop_edges"] = pred_cost
        # print("1", pred_cost1)
        # print("2", pred_cost2)
        # print("0", pred_cost0)

        return obs

    def update_cost_and_mask(self, obs):
        for remote, cost, mask in zip(self.remotes, obs['coop_edges'], obs['coop_edge_mask']):
            remote.send(("update_prediction", [cost, mask]))

        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_obs(obs, self.observation_space)

        return obs

    def get_data_for_offline_planning(self):
        for remote in self.remotes:
            remote.send(("get_data_for_offline_planning", None))
        data = [remote.recv() for remote in self.remotes]
        return data

    ####!!!!!!!!!!!!!!!!!!!!!!!!!!! my func end ###############################################
    def step_async(self, actions: np.ndarray) -> None:
        # print("subproc:", actions)
        for remote, action in zip(self.remotes, actions):
            # print("a",action)
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        ######## my code ######
        obs = _flatten_obs(obs, self.observation_space)
        if self.use_prediction_model:
            obs = self.predict_cost_and_mask(obs)
            obs = self.update_cost_and_mask(obs)

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
        if self.use_prediction_model:
            obs = self.predict_cost_and_mask(obs)
            obs = self.update_cost_and_mask(obs)

        #######################3

        return obs
    def reset_with_task_data(self, load_task_data = None) -> VecEnvObs:
        if load_task_data is not None:
            assert load_task_data.shape[0] == len(self.remotes)
            for remote, data in zip(self.remotes, load_task_data):
                remote.send(("reset_with_task_data", data))
            obs = [remote.recv() for remote in self.remotes]
        else:
            for remote in self.remotes:
                remote.send(("reset", None))
            obs = [remote.recv() for remote in self.remotes]

        ######## my code ######
        obs = _flatten_obs(obs, self.observation_space)
        if self.use_prediction_model:
            obs = self.predict_cost_and_mask(obs)
            obs = self.update_cost_and_mask(obs)

        #######################3

        return obs
    def get_current_task_data(self):

        for remote in self.remotes:
            remote.send(("get_current_task_data", None))
        data = [remote.recv() for remote in self.remotes]
        data = np.stack(data)

        return data
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
