from typing import Callable, Any, Dict, List, Optional, Tuple, Type, Union

import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
from gym import spaces
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, is_vectorized_observation, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy


from stable_baselines3.dqn.dqn import DQN, QNetwork
from stable_baselines3.dqn.policies import DQNPolicy
from model_utils_task import CustomFeatureExtractor


class Custom_DQN(DQN):
    #### change action sample rules(add mask)
    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.
        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            # if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
            #     if isinstance(observation, dict):
            #         n_batch = observation[list(observation.keys())[0]].shape[0]
            #     else:
            #         n_batch = observation.shape[0]
            #     action = np.array([self.action_space.sample() for _ in range(n_batch)])
            # else:
            #     action = np.array(self.action_space.sample())
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        else:
            action, state = self.policy.predict(observation, state, episode_start, True)
        return action, state


class Custom_DQNPolicy(DQNPolicy):
    ###### load my Q net
    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return Custom_QNetwork(**net_args).to(self.device)

class Custom_QNetwork(QNetwork):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            features_extractor: nn.Module,
            features_dim: int,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = CustomFeatureExtractor(self.observation_space)
        self.mask_dim = self.features_extractor.mask_dim
        self.features_dim = features_dim
        action_dim = self.action_space.n  # number of actions



        self.q_net = CustomDQNNetwork_EdgeAttentionWithNodeEncoding(self.node_features_dim, self.mask_dim,)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.
        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.q_net(self.extract_features(obs, self.features_extractor))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        # Greedy action
        if deterministic:
            action = q_values.argmax(dim=1).reshape(-1)
        else:
            action = q_values.argmax(dim=1).reshape(-1)
        return action



from multi_head_attention import MultiHeadSelfCrossAttentionWithNodeAndEdge2, MultiHeadAttention
class CustomDQNNetwork_EdgeAttentionWithNodeEncoding(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            mask_dim: int,
    ):
        super(CustomDQNNetwork_EdgeAttentionWithNodeEncoding, self).__init__()

        # IMPORTANT:
        self.feature_dim = feature_dim
        self.mask_dim = mask_dim
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = feature_dim
        self.head_num = 4 // 2

        self.setup()

    def setup(self):
        # node decoder
        self.node_decoder1_1 = MultiHeadSelfCrossAttentionWithNodeAndEdge2(self.feature_dim, self.head_num,
                                                                           self.feature_dim)
        self.node_decoder2_1 = MultiHeadSelfCrossAttentionWithNodeAndEdge2(self.feature_dim, self.head_num,
                                                                           self.latent_dim_pi)

        self.node_decoder1_2 = MultiHeadSelfCrossAttentionWithNodeAndEdge2(self.feature_dim, self.head_num,
                                                                           self.feature_dim)
        self.node_decoder2_2 = MultiHeadSelfCrossAttentionWithNodeAndEdge2(self.feature_dim, self.head_num,
                                                                           self.latent_dim_pi)

        self.coop_edge_decoder1 = nn.Linear(self.feature_dim * 3, self.feature_dim)
        self.coop_edge_decoder2 = nn.Linear(self.feature_dim * 3, self.feature_dim)

        self.task_edge_decoder1_1 = nn.Linear(self.feature_dim * 3, self.feature_dim)
        self.task_edge_decoder2_1 = nn.Linear(self.feature_dim * 3, self.feature_dim)
        self.task_edge_decoder1_2 = nn.Linear(self.feature_dim * 3, self.feature_dim)
        self.task_edge_decoder2_2 = nn.Linear(self.feature_dim * 3, self.feature_dim)

        # Policy network
        self.q_net = nn.Linear(self.latent_dim_pi * 3, 1)

    def decode_feature(self, features):
        f_r1 = features["robot_1_node"]
        f_r2 = features["robot_2_node"]
        # return f_r1, f_r2

        edge_r1 = features["robot_1_task_edge_dist"]
        edge_r2 = features["robot_2_task_edge_dist"]

        mask1 = th.unsqueeze(features["robot_1_task_edge_mask"], 1)
        mask2 = th.unsqueeze(features["robot_2_task_edge_mask"], 1)

        edge_coop = features["coop_edge_cost"]
        coop_mask = features["coop_edge_mask"]

        ###############################
        edge_coop = self.decode_coop_edge(edge_coop, coop_mask, f_r1, f_r2, self.coop_edge_decoder1)
        edge_r1 = self.decode_task_edge(edge_r1, mask1, f_r1, self.task_edge_decoder1_1)
        edge_r2 = self.decode_task_edge(edge_r2, mask2, f_r2, self.task_edge_decoder1_2)

        f_r1_1 = self.node_decoder1_1(f_r1, f_r2, edge_r1, edge_coop, masks=mask1, maskc=coop_mask)
        f_r2_1 = self.node_decoder1_2(f_r2, f_r1, edge_r2, edge_coop.transpose(-3, -2), masks=mask2,
                                      maskc=coop_mask.transpose(-2, -1))
        edge_coop = self.decode_coop_edge(edge_coop, coop_mask, f_r1_1, f_r2_1, self.coop_edge_decoder2)

        edge_r1 = self.decode_task_edge(edge_r1, mask1, f_r1_1, self.task_edge_decoder2_1)
        edge_r2 = self.decode_task_edge(edge_r2, mask2, f_r2_1, self.task_edge_decoder2_2)
        f_r1_2 = self.node_decoder2_1(f_r1_1, f_r2_1, edge_r1, edge_coop, masks=mask1, maskc=coop_mask)
        f_r2_2 = self.node_decoder2_2(f_r2_1, f_r1_1, edge_r2, edge_coop.transpose(-3, -2), masks=mask2,
                                      maskc=coop_mask.transpose(-2, -1))

        edge_coop = th.flatten(edge_coop, -3, -2)
        # coop_mask = th.flatten(coop_mask, -2, -1)
        return edge_coop, coop_mask, f_r1_2, f_r2_2

    def decode_coop_edge(self, edge_coop, coop_mask, f_r1, f_r2, layer, last_layer=False):
        edge_coop = edge_coop.reshape(
            (-1, self.mask_dim ** 2, self.latent_dim_pi))
        f_r1 = th.unsqueeze(f_r1, -2).repeat(1, 1, self.mask_dim, 1).reshape(
            (-1, self.mask_dim ** 2, self.latent_dim_pi))
        f_r2 = th.unsqueeze(f_r2, -3).repeat(1, self.mask_dim, 1, 1).reshape(
            (-1, self.mask_dim ** 2, self.latent_dim_pi))
        f = th.cat([f_r1, f_r2, edge_coop], dim=-1)

        edge_coop = layer(f)
        if not last_layer:
            edge_coop = edge_coop.reshape((-1, self.mask_dim, self.mask_dim, self.latent_dim_pi))
        return edge_coop

    def decode_task_edge(self, edge_task, mask, f, layer, last_layer=False):
        edge_task = edge_task.reshape(
            (-1, self.mask_dim ** 2, self.latent_dim_pi))

        f_r1 = th.unsqueeze(f, -2).repeat(1, 1, self.mask_dim, 1).reshape(
            (-1, self.mask_dim ** 2, self.latent_dim_pi))
        f_r2 = th.unsqueeze(f, -3).repeat(1, self.mask_dim, 1, 1).reshape(
            (-1, self.mask_dim ** 2, self.latent_dim_pi))
        f = th.cat([f_r1, f_r2, edge_task], dim=-1)

        edge_task = layer(f)
        if not last_layer:
            edge_task = edge_task.reshape((-1, self.mask_dim, self.mask_dim, self.latent_dim_pi))
        return edge_task

    def calcul_action(self, edge_coop, coop_mask, f1, f2):
        action = th.flatten(self.decode_coop_edge(edge_coop, coop_mask, f1, f2, self.policy_net, last_layer = True), -2, -1)
        return action


    def mask_action(self, action, coop_mask):
        assert action.shape[-1] == self.mask_dim ** 2
        coop_mask = th.flatten(coop_mask, -2, -1)
        action = F.softmax(action.masked_fill(coop_mask < 1, -1e9), dim=-1)
        return action

    def forward(self, features: th.Tensor) -> th.Tensor:
        edge_coop, coop_mask, f1, f2 = self.decode_feature(features)

        action = self.calcul_action(edge_coop, coop_mask, f1, f2)
        masked_action = self.mask_action(action, coop_mask)

        return masked_action

