from typing import Callable, Any, Dict, List, Optional, Tuple, Type, Union

from gym import spaces
import gym
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn.functional as F


############################# custom AC policy

class CustomActorCriticPolicy0220(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):
        self.mask_dim = 0
        self.node_features_dim = 0
        self.discrete_action_space = False

        super(CustomActorCriticPolicy0220, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        # Feature extractor
        self.features_extractor = CustomFeatureExtractor(self.observation_space, **self.features_extractor_kwargs)
        self.node_features_dim = self.features_extractor._features_dim
        self.mask_dim = self.features_extractor.mask_dim
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
        print(self.node_features_dim, self.mask_dim)

        # Action distribution
        self.action_dist, discrete_action_space = make_custom_proba_distribution(action_space, use_sde=False,
                                                                                 dist_kwargs=None)
        self.discrete_action_space = discrete_action_space
        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork_0220(self.node_features_dim, self.mask_dim,
                                                discrete_action_space=self.discrete_action_space)


####################################3 custom actor critic network

import time

latent_dim = 64
from multi_head_attention import MultiHeadAttention, MultiHeadAttentionWithEdge


class CustomNetwork_0220(nn.Module):
    """
    this network flatten all node observations and use mlp to decode, action mask is added

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            mask_dim: int,
            last_layer_dim_pi: int = latent_dim,
            last_layer_dim_vf: int = latent_dim,
            discrete_action_space: bool = False,
    ):
        super(CustomNetwork_0220, self).__init__()

        # IMPORTANT:
        self.feature_dim = feature_dim
        self.mask_dim = mask_dim
        self.discrete_action_space = discrete_action_space
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = self.mask_dim * 2
        self.latent_dim_vf = 64
        self.head_num = 8
        self.setup()

    def setup(self):
        self.robot1_node_encoder_1 = MultiHeadAttentionWithEdge(self.feature_dim, self.head_num, self.feature_dim)
        self.robot1_node_encoder_2 = MultiHeadAttentionWithEdge(self.feature_dim, self.head_num, self.feature_dim)
        # self.robot1_task_edge_encoder_1 = nn.Linear(self.feature_dim * 3, self.feature_dim)

        self.robot2_node_encoder_1 = MultiHeadAttentionWithEdge(self.feature_dim, self.head_num, self.feature_dim)
        self.robot2_node_encoder_2 = MultiHeadAttentionWithEdge(self.feature_dim, self.head_num, self.feature_dim)
        # self.robot2_task_edge_encoder_1 = nn.Linear(self.feature_dim * 3, self.feature_dim)

        ##########################################33
        self.action_decoder_1 = MultiHeadAttention(self.feature_dim * 3, self.head_num, self.feature_dim)
        # self.action_decoder_1 = MultiHeadAttention(self.feature_dim , self.head_num, self.feature_dim)
        ####################################

        self.action_decoder_2 = MultiHeadAttention(self.feature_dim, self.head_num, 16)

        self.policy_net = nn.Linear(16, 1)

        self.value_net_1 = nn.Linear(16, 4)
        self.value_net_2 = nn.Sequential(
            nn.Linear(self.mask_dim ** 2 * 4, self.latent_dim_vf)
        )

    def encode(self, features):
        f_r1 = features["robot_1_node"]
        f_r2 = features["robot_2_node"]
        # return f_r1, f_r2

        edge_r1 = features["robot_1_task_edges"]
        edge_r2 = features["robot_2_task_edges"]

        mask1 = features["robot_1_task_edge_mask"]
        mask2 = features["robot_2_task_edge_mask"]

        edge_coop = features["coop_edges"]
        coop_mask = features["coop_edge_mask"]

        ###############################
        f_r1 = self.robot1_node_encoder_1(f_r1, edge_r1, f_r1, mask=mask1)
        f_r1 = self.robot1_node_encoder_2(f_r1, edge_r1, f_r1, mask=mask1)

        f_r2 = self.robot2_node_encoder_1(f_r2, edge_r2, f_r2, mask=mask2)
        f_r2 = self.robot2_node_encoder_2(f_r2, edge_r2, f_r2, mask=mask2)
        ###############################
        edge_coop = edge_coop.reshape(
            (-1, self.mask_dim ** 2, self.feature_dim))
        f_r1 = th.unsqueeze(f_r1, -2).repeat(1, 1, self.mask_dim, 1).reshape(
            (-1, self.mask_dim ** 2, self.feature_dim))
        f_r2 = th.unsqueeze(f_r2, -3).repeat(1, self.mask_dim, 1, 1).reshape(
            (-1, self.mask_dim ** 2, self.feature_dim))

        ############ cat or add #############
        f = th.cat([f_r1, f_r2, edge_coop], dim=-1)
        # f = f_r1 + f_r2 + edge_coop
        ############ cat or add #############
        c_mask = coop_mask.reshape(
            (-1, self.mask_dim ** 2, 1))
        # print(f.shape, c_mask.shape)
        # print(f_r1.shape, edge_r1.shape, mask1.shape)
        return f, c_mask

    def decode(self, features, mask):
        features = self.action_decoder_1(features, features, features, mask=mask)
        features = self.action_decoder_2(features, features, features, mask=mask)
        return features

    def calcul_action(self, features, mask):
        action = th.flatten(self.policy_net(features), -2, -1)
        mask = th.flatten(mask, -2, -1)
        # print(action.shape, mask.shape)
        masked_action = F.softmax(action.masked_fill(mask == 0, -1e9), dim=-1)
        return masked_action

    def calcul_critic(self, features):
        critic = th.flatten(self.value_net_1(features), -2, -1)
        # print(critic.shape)
        return self.value_net_2(critic)

    def forward(self, features):
        features, action_mask = self.encode(features)
        features = self.decode(features, action_mask)

        action = self.calcul_action(features, action_mask)
        critic = self.calcul_critic(features)
        return action, critic

    def forward_actor(self, features):
        features, action_mask = self.encode(features)
        features = self.decode(features, action_mask)
        action = self.calcul_action(features, action_mask)
        return action

    def forward_critic(self, features):
        features, action_mask = self.encode(features)
        features = self.decode(features, action_mask)
        critic = self.calcul_critic(features)
        return critic


#################################### custom feature extractor
############################  experiment: node state obs
part_node_feature_dim = 22
# part_node_feature_dim = 12
############################  experiment: node state obs
############################  experiment: rest node
reset_node_feature_dim = 15
# reset_node_feature_dim = 9
############################  experiment: rest node
############################  experiment: task edge
task_edges_feature_dim = 3
# task_edges_feature_dim = 2
############################  experiment: task edge
############################  experiment: coop edge
coop_edges_feature_dim = 6
# coop_edges_feature_dim = 3
# coop_edges_feature_dim = 2
# coop_edges_feature_dim = 1
############################  experiment: coop edge
encode_latent_num_node = 64
encode_latent_num_edge = 64

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):

        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=encode_latent_num_node)

        self.node_feature_dim = 0
        self.mask_dim = 0
        self._features_dim = encode_latent_num_node
        for key, subspace in observation_space.spaces.items():
            if "node" in key:
                self.node_feature_dim = subspace.shape[0]
            elif "task_edge_mask" in key:
                self.mask_dim = subspace.shape[0]

        extractors = {}
        extractors['robot_1_part_node_encoder'] = nn.Linear(part_node_feature_dim, encode_latent_num_node)
        extractors['robot_2_part_node_encoder'] = nn.Linear(part_node_feature_dim, encode_latent_num_node)

        extractors['robot_1_reset_node_encoder'] = nn.Linear(reset_node_feature_dim, encode_latent_num_node)
        extractors['robot_2_reset_node_encoder'] = nn.Linear(reset_node_feature_dim, encode_latent_num_node)

        extractors['task_edge_encoder'] = nn.Linear(task_edges_feature_dim, encode_latent_num_edge)
        extractors['coop_edge_encoder'] = nn.Linear(coop_edges_feature_dim, encode_latent_num_edge)

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = encode_latent_num_node

    def forward(self, observations) -> th.Tensor:
        encoded_observation = {}
        robot_1_part_node_tensor_list = []
        robot_2_part_node_tensor_list = []

        for key, obs in observations.items():

            if "robot_1_part" in key:
                node_obs = obs.reshape([obs.shape[0], 1, obs.shape[1]])
                robot_1_part_node_tensor_list.append(node_obs)
            elif "robot_1_reset" in key:
                reset_node_obs1 = obs.reshape([obs.shape[0], 1, obs.shape[1]])
                reset_node_obs1 = self.extractors['robot_1_reset_node_encoder'](reset_node_obs1)
            elif "robot_2_part" in key:
                node_obs = obs.reshape([obs.shape[0], 1, obs.shape[1]])
                robot_2_part_node_tensor_list.append(node_obs)
            elif "robot_2_reset" in key:
                reset_node_obs2 = obs.reshape([obs.shape[0], 1, obs.shape[1]])
                reset_node_obs2 = self.extractors['robot_2_reset_node_encoder'](reset_node_obs2)
            elif "mask" in key:
                encoded_observation[key] = obs
            elif "task_edges" in key:
                task_edge_obs = obs.reshape([obs.shape[0], obs.shape[1], obs.shape[2], task_edges_feature_dim])
                encoded_observation[key] = self.extractors['task_edge_encoder'](task_edge_obs)
            elif "coop_edges" in key:
                coop_edge_obs = obs.reshape([obs.shape[0], obs.shape[1], obs.shape[2], coop_edges_feature_dim])
                encoded_observation[key] = self.extractors['coop_edge_encoder'](coop_edge_obs)

        robot_1_part_node_tensor = th.cat(robot_1_part_node_tensor_list, dim=1)
        robot_2_part_node_tensor = th.cat(robot_2_part_node_tensor_list, dim=1)

        part_node_obs1 = self.extractors['robot_1_part_node_encoder'](robot_1_part_node_tensor)
        part_node_obs2 = self.extractors['robot_1_part_node_encoder'](robot_2_part_node_tensor)

        encoded_observation["robot_1_node"] = th.cat([part_node_obs1, reset_node_obs1], dim=1)
        encoded_observation["robot_2_node"] = th.cat([part_node_obs2, reset_node_obs2], dim=1)

        return encoded_observation


########################### custom distribution


from stable_baselines3.common.distributions import Distribution, CategoricalDistribution, MultiCategoricalDistribution, \
    DiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from torch.distributions import Bernoulli, Categorical, Normal


class FakeNet(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(FakeNet, self).__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.linear = None

    def forward(self, x):
        # print(x)
        return x


class CustomMultiCategoricalDistribution(MultiCategoricalDistribution):
    def __init__(self, action_dim: int):
        super(CustomMultiCategoricalDistribution, self).__init__(action_dim)

        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        directly return the output of policy network, which is masked action logits
        """
        action_logits = FakeNet(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "MultiCategoricalDistribution":
        action_dim_num = action_logits.shape[-1] // 2
        action_dim = [action_dim_num, action_dim_num]
        ##############################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        # print(action_logits)

        self.distribution = [Categorical(probs=split) for split in
                             th.split(action_logits, tuple(action_dim), dim=1)]

        # print("action logit:\t", action_logits)
        # for split in th.split(action_logits, tuple(self.action_dims), dim=1):
        #     for i in range(5):
        #         print("split,   ", split, Categorical(logits=split).sample())
        # print(d1.sample(),d2.sample())
        return self


class CustomCategoricalDistribution(CategoricalDistribution):
    def __init__(self, action_dim: int):
        super(CustomCategoricalDistribution, self).__init__(action_dim)

        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        directly return the output of policy network, which is masked action logits
        """
        action_logits = FakeNet(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "CategoricalDistribution":
        self.distribution = Categorical(probs=action_logits)
        return self


class CustomDiagGaussianDistribution(DiagGaussianDistribution):
    def __init__(self, action_dim: int):
        super(CustomDiagGaussianDistribution, self).__init__(action_dim)

        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)
        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = FakeNet(latent_dim, self.action_dim)
        # TODO: allow action dependent std
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(
            self, mean_actions: th.Tensor, log_std: th.Tensor
    ):
        """
        Create the distribution given its parameters (mean, std)
        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self


def make_custom_proba_distribution(
        action_space: gym.spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space
    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.MultiDiscrete):
        return CustomMultiCategoricalDistribution(action_space.nvec, **dist_kwargs), False
    elif isinstance(action_space, spaces.Discrete):
        return CustomCategoricalDistribution(action_space.n, **dist_kwargs), True
    elif isinstance(action_space, spaces.Box):
        assert len(action_space.shape) == 1, "Error: the action space must be a vector"
        cls = CustomDiagGaussianDistribution
        return cls(get_action_dim(action_space), **dist_kwargs), False
        # return None, False
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )
