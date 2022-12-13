from typing import Callable, Any, Dict, List, Optional, Tuple, Type, Union

from gym import spaces
import gym
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


############################# custom AC policy

class CustomActorCriticPolicy(ActorCriticPolicy):
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

        super(CustomActorCriticPolicy, self).__init__(
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
        self.action_dist = make_custom_proba_distribution_multidiscrete(action_space, use_sde=False, dist_kwargs=None)
        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        # self.mlp_extractor = CustomNetwork_FlattenNodes(self.node_features_dim, self.mask_dim)
        # self.mlp_extractor = CustomNetwork_FlattenNodesAndEdges(self.node_features_dim, self.mask_dim)
        # self.mlp_extractor = CustomNetwork_SelfAttention(self.node_features_dim, self.mask_dim)
        # self.mlp_extractor = CustomNetwork_SelfCrossAttention(self.node_features_dim, self.mask_dim)
        self.mlp_extractor = CustomNetwork_SelfAttentionWithTaskEdge(self.node_features_dim, self.mask_dim)


####################################3 custom actor critic network

import time

latent_dim = 128


class CustomNetwork_FlattenNodes(nn.Module):
    """
    this network flatten all node observations and use mlp to decode, action mask is added

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            node_feature_dim: int,
            mask_dim: int,
            last_layer_dim_pi: int = latent_dim,
            last_layer_dim_vf: int = latent_dim,
    ):
        super(CustomNetwork_FlattenNodes, self).__init__()

        # IMPORTANT:
        self.node_feature_dim = node_feature_dim
        self.mask_dim = mask_dim
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = self.mask_dim * 2
        self.latent_dim_vf = 64

        flatten_node_dim = self.node_feature_dim * self.mask_dim * 2
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(flatten_node_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, self.mask_dim * 2)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(flatten_node_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, self.latent_dim_vf), nn.ReLU(),
        )

    def flatten(self, features: th.Tensor) -> th.Tensor:
        f_r1 = th.flatten(features["robot_1_node"], 1, 2)
        f_r2 = th.flatten(features["robot_2_node"], 1, 2)

        return th.cat([f_r1, f_r2], dim=1)

    def mask_action(self, features, action):
        assert action.shape[1] == self.mask_dim * 2

        mask = th.cat([features["robot_1_task_edge_mask"], features["robot_2_task_edge_mask"]], dim=1)
        masked_action = action.masked_fill(mask == 0, -1e9)

        return masked_action

    def forward(self, features: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        flatten_feature = self.flatten(features)

        unmasked_action = self.policy_net(flatten_feature)
        masked_action = self.mask_action(features, unmasked_action)

        # print(np.round_(masked_action.cpu().detach().numpy(), decimals = 5))
        # time.sleep(1)

        return masked_action, self.value_net(flatten_feature)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        flatten_feature = self.flatten(features)

        unmasked_action = self.policy_net(flatten_feature)
        masked_action = self.mask_action(features, unmasked_action)
        return masked_action

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        flatten_feature = self.flatten(features)
        return self.value_net(flatten_feature)


######
class CustomNetwork_FlattenNodesAndEdges(nn.Module):
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
    ):
        super(CustomNetwork_FlattenNodesAndEdges, self).__init__()

        # IMPORTANT:
        self.feature_dim = feature_dim
        self.mask_dim = mask_dim
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = self.mask_dim * 2
        self.latent_dim_vf = 64

        flatten_node_dim = self.feature_dim * self.mask_dim * 2 + self.feature_dim * self.mask_dim * self.mask_dim * 3 + self.mask_dim * self.mask_dim
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(flatten_node_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, self.mask_dim * 2)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(flatten_node_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, self.latent_dim_vf), nn.ReLU(),
        )

    def flatten(self, features: th.Tensor) -> th.Tensor:
        f_r1_n = th.flatten(features["robot_1_node"], 1, 2)
        f_r2_n = th.flatten(features["robot_2_node"], 1, 2)
        f_r1_e = th.flatten(features["robot_1_task_edge_dist"], 1, 3)
        f_r2_e = th.flatten(features["robot_2_task_edge_dist"], 1, 3)
        f_c_e = th.flatten(features["coop_edge_cost"], 1, 3)
        f_c_m = th.flatten(features["coop_edge_mask"], 1, 2)

        return th.cat([f_r1_n, f_r2_n, f_r1_e, f_r2_e, f_c_e, f_c_m], dim=1)

    def mask_action(self, features, action):
        assert action.shape[1] == self.mask_dim * 2

        mask = th.cat([features["robot_1_task_edge_mask"], features["robot_2_task_edge_mask"]], dim=1)
        masked_action = action.masked_fill(mask == 0, -1e9)
        return masked_action

    def forward(self, features: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        flatten_feature = self.flatten(features)
        unmasked_action = self.policy_net(flatten_feature)
        masked_action = self.mask_action(features, unmasked_action)
        return masked_action, self.value_net(flatten_feature)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        flatten_feature = self.flatten(features)

        unmasked_action = self.policy_net(flatten_feature)
        masked_action = self.mask_action(features, unmasked_action)
        return masked_action

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        flatten_feature = self.flatten(features)
        return self.value_net(flatten_feature)


##### attention nodes

from multi_head_attention import ScaledDotProductAttention, MultiHeadAttention


class CustomNetwork_SelfAttention(nn.Module):
    """
    this network use self attention to decode node and task edge features

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
    ):
        super(CustomNetwork_SelfAttention, self).__init__()

        # IMPORTANT:
        self.feature_dim = feature_dim
        self.mask_dim = mask_dim
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = self.mask_dim * 2
        self.latent_dim_vf = 64
        self.head_num = 8

        # node decoder
        self.robot1_decoder1 = MultiHeadAttention(feature_dim, self.head_num, feature_dim)
        self.robot1_decoder2 = MultiHeadAttention(feature_dim, self.head_num, 16)

        self.robot2_decoder1 = MultiHeadAttention(feature_dim, self.head_num, feature_dim)
        self.robot2_decoder2 = MultiHeadAttention(feature_dim, self.head_num, 16)

        # Policy network
        self.policy_net_r1 = nn.Sequential(
            nn.Linear(16, 1)
        )
        self.policy_net_r2 = nn.Sequential(
            nn.Linear(16, 1)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(16 * 2 * mask_dim, self.latent_dim_vf)
        )

    def mask_action(self, features, action):
        assert action.shape[1] == self.mask_dim * 2
        # print("!!!!!!!!!!!!!!!!!!!!1")
        # print(action.detach().numpy())
        mask = th.cat([features["robot_1_task_edge_mask"], features["robot_2_task_edge_mask"]], dim=1)
        masked_action = action.masked_fill(mask == 0, -1e9)
        return masked_action

    def decode_feature(self, features):
        f_r1 = features["robot_1_node"]
        f_r2 = features["robot_2_node"]

        # print("111111111111111111111111")
        # print(f_r1[:,:,:5].detach().numpy())

        mask1 = th.unsqueeze(features["robot_1_task_edge_mask"], 1)
        mask2 = th.unsqueeze(features["robot_2_task_edge_mask"], 1)

        f_r1 = self.robot1_decoder1(f_r1, f_r1, f_r1, mask=mask1)

        # print("?????????????????????????????")
        # print(mask1)
        # print(f_r1[:, :, :5].detach().numpy())

        f_r1 = self.robot1_decoder2(f_r1, f_r1, f_r1, mask=mask1)

        # print("2222222222222222222222222222")
        # print(f_r1[:,:,:5].detach().numpy())

        f_r2 = self.robot2_decoder1(f_r2, f_r2, f_r2, mask=mask2)
        f_r2 = self.robot2_decoder2(f_r2, f_r2, f_r2, mask=mask2)

        # print("3333333333333333333333333333333")
        # print(f_r1[:, :, :5].detach().numpy())

        return f_r1, f_r2

    def forward(self, features: th.Tensor) -> th.Tensor:
        f_r1, f_r2 = self.decode_feature(features)

        act_r1 = self.policy_net_r1(f_r1)
        act_r2 = self.policy_net_r2(f_r1)

        # print("@@@@@@@@@@@@@@@@@@@@@@")
        # print(f_r1.detach().numpy())

        action = th.cat([th.flatten(act_r1, 1, 2), th.flatten(act_r2, 1, 2)], dim=1)
        masked_action = self.mask_action(features, action)

        f_value = th.cat([th.flatten(f_r1, 1, 2), th.flatten(f_r2, 1, 2)], dim=1)

        return masked_action, self.value_net(f_value)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        f_r1, f_r2 = self.decode_feature(features)

        act_r1 = self.policy_net_r1(f_r1)
        act_r2 = self.policy_net_r2(f_r1)

        action = th.cat([th.flatten(act_r1, 1, 2), th.flatten(act_r2, 1, 2)], dim=1)
        masked_action = self.mask_action(features, action)
        return masked_action

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        f_r1, f_r2 = self.decode_feature(features)
        f_value = th.cat([th.flatten(f_r1, 1, 2), th.flatten(f_r2, 1, 2)], dim=1)
        return self.value_net(f_value)


######## self + cross attention

from multi_head_attention import MultiHeadSelfCrossAttention


class CustomNetwork_SelfCrossAttention(CustomNetwork_SelfAttention):
    """
    this network use self attention to decode node and task edge features

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
    ):
        super(CustomNetwork_SelfCrossAttention, self).__init__(
            feature_dim=feature_dim,
            mask_dim=mask_dim,
            last_layer_dim_pi=latent_dim,
            last_layer_dim_vf=latent_dim,
        )

        # IMPORTANT:
        self.feature_dim = feature_dim
        self.mask_dim = mask_dim
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = self.mask_dim * 2
        self.latent_dim_vf = 64
        self.head_num = 8 // 2

        # node decoder
        self.robot1_decoder1 = MultiHeadSelfCrossAttention(self.feature_dim, self.head_num, self.feature_dim)
        self.robot1_decoder2 = MultiHeadSelfCrossAttention(self.feature_dim, self.head_num, 16)

        self.robot2_decoder1 = MultiHeadSelfCrossAttention(self.feature_dim, self.head_num, self.feature_dim)
        self.robot2_decoder2 = MultiHeadSelfCrossAttention(self.feature_dim, self.head_num, 16)

        # Policy network
        self.policy_net_r1 = nn.Sequential(
            nn.Linear(16, 1)
        )
        self.policy_net_r2 = nn.Sequential(
            nn.Linear(16, 1)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(16 * 2 * mask_dim, self.latent_dim_vf)
        )

    def decode_feature(self, features):
        f_r1 = features["robot_1_node"]
        f_r2 = features["robot_2_node"]

        mask1 = th.unsqueeze(features["robot_1_task_edge_mask"], 1)
        mask2 = th.unsqueeze(features["robot_2_task_edge_mask"], 1)

        coop_mask = features["coop_edge_mask"]
        # print(coop_mask.shape, "!!!!!!!!!!!", mask1.shape)

        f_r1_1 = self.robot1_decoder1(f_r1, f_r2, masks=mask1, maskc=coop_mask)
        f_r2_1 = self.robot2_decoder1(f_r2, f_r1, masks=mask2, maskc=coop_mask.transpose(-2, -1))
        # print("!!!!!!!!!!!!!!!!!!!11111!!!")
        # print(f_r1.shape, f_r2.shape)
        f_r1_2 = self.robot1_decoder2(f_r1_1, f_r2_1, masks=mask1, maskc=coop_mask)
        f_r2_2 = self.robot2_decoder2(f_r2_1, f_r1_1, masks=mask2, maskc=coop_mask.transpose(-2, -1))

        # print("3333333333333333333333333333333")
        # print(f_r1[:, :, :5].detach().numpy())

        return f_r1_2, f_r2_2


######## self attention with edge

from multi_head_attention import MultiHeadAttentionWithEdge


class CustomNetwork_SelfAttentionWithTaskEdge(CustomNetwork_SelfAttention):
    """
    this network use self attention to decode node and task edge features

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
    ):
        super(CustomNetwork_SelfAttentionWithTaskEdge, self).__init__(
            feature_dim=feature_dim,
            mask_dim=mask_dim,
            last_layer_dim_pi=latent_dim,
            last_layer_dim_vf=latent_dim,
        )

        # IMPORTANT:
        self.feature_dim = feature_dim
        self.mask_dim = mask_dim
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = self.mask_dim * 2
        self.latent_dim_vf = 64
        self.head_num = 8

        # node decoder
        self.robot1_decoder1 = MultiHeadAttentionWithEdge(feature_dim, self.head_num, feature_dim)
        self.robot1_decoder2 = MultiHeadAttentionWithEdge(feature_dim, self.head_num, 16)

        self.robot2_decoder1 = MultiHeadAttentionWithEdge(feature_dim, self.head_num, feature_dim)
        self.robot2_decoder2 = MultiHeadAttentionWithEdge(feature_dim, self.head_num, 16)

        # Policy network
        self.policy_net_r1 = nn.Sequential(
            nn.Linear(16, 1)
        )
        self.policy_net_r2 = nn.Sequential(
            nn.Linear(16, 1)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(16 * 2 * mask_dim, self.latent_dim_vf)
        )

    def decode_feature(self, features):
        f_r1 = features["robot_1_node"]
        f_r2 = features["robot_2_node"]

        edge_r1 = features["robot_1_task_edge_dist"]
        edge_r2 = features["robot_2_task_edge_dist"]

        mask1 = th.unsqueeze(features["robot_1_task_edge_mask"], 1)
        mask2 = th.unsqueeze(features["robot_2_task_edge_mask"], 1)

        f_r1 = self.robot1_decoder1(f_r1, edge_r1, f_r1, mask=mask1)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!")
        f_r1 = self.robot1_decoder2(f_r1, edge_r1, f_r1, mask=mask1)

        f_r2 = self.robot2_decoder1(f_r2, edge_r2, f_r2, mask=mask2)
        f_r2 = self.robot2_decoder2(f_r2, edge_r2, f_r2, mask=mask2)

        return f_r1, f_r2


########### self and cross attention with edges

from multi_head_attention import MultiHeadSelfCrossAttentionWithEdge


class CustomNetwork_SelfCrossAttentionWithEdge(CustomNetwork_SelfAttention):
    """
    this network use self attention to decode node and task edge features

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
    ):
        super(CustomNetwork_SelfCrossAttentionWithEdge, self).__init__(
            feature_dim=feature_dim,
            mask_dim=mask_dim,
            last_layer_dim_pi=latent_dim,
            last_layer_dim_vf=latent_dim,
        )

        # IMPORTANT:
        self.feature_dim = feature_dim
        self.mask_dim = mask_dim
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = self.mask_dim * 2
        self.latent_dim_vf = 64
        self.head_num = 8 // 2

        # node decoder
        self.robot1_decoder1 = MultiHeadSelfCrossAttentionWithEdge(self.feature_dim, self.head_num, self.feature_dim)
        self.robot1_decoder2 = MultiHeadSelfCrossAttentionWithEdge(self.feature_dim, self.head_num, 16)

        self.robot2_decoder1 = MultiHeadSelfCrossAttentionWithEdge(self.feature_dim, self.head_num, self.feature_dim)
        self.robot2_decoder2 = MultiHeadSelfCrossAttentionWithEdge(self.feature_dim, self.head_num, 16)

        # Policy network
        self.policy_net_r1 = nn.Sequential(
            nn.Linear(16, 1)
        )
        self.policy_net_r2 = nn.Sequential(
            nn.Linear(16, 1)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(16 * 2 * mask_dim, self.latent_dim_vf)
        )

    def decode_feature(self, features):
        f_r1 = features["robot_1_node"]
        f_r2 = features["robot_2_node"]

        edge_r1 = features["robot_1_task_edge_dist"]
        edge_r2 = features["robot_2_task_edge_dist"]
        edge_coop = features["coop_edge_cost"]

        mask1 = th.unsqueeze(features["robot_1_task_edge_mask"], 1)
        mask2 = th.unsqueeze(features["robot_2_task_edge_mask"], 1)

        coop_mask = features["coop_edge_mask"]
        # print(coop_mask.shape, "!!!!!!!!!!!", mask1.shape)

        f_r1_1 = self.robot1_decoder1(f_r1, f_r2, edge_r1, edge_coop, masks=mask1, maskc=coop_mask)
        f_r2_1 = self.robot2_decoder1(f_r2, f_r1, edge_r2, edge_coop, masks=mask2, maskc=coop_mask.transpose(-2, -1))
        # print("!!!!!!!!!!!!!!!!!!!11111!!!")
        # print(f_r1.shape, f_r2.shape)
        f_r1_2 = self.robot1_decoder2(f_r1_1, f_r2_1, edge_r1, edge_coop, masks=mask1, maskc=coop_mask)
        f_r2_2 = self.robot2_decoder2(f_r2_1, f_r1_1, edge_r2, edge_coop, masks=mask2,
                                      maskc=coop_mask.transpose(-2, -1))

        # print("3333333333333333333333333333333")
        # print(f_r1[:, :, :5].detach().numpy())

        return f_r1_2, f_r2_2


#################################### custom feature extractor
node_feature_dim = 24
encode_latent_num_node = 128
encode_latent_num_edge = 128

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):

        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=node_feature_dim)

        self.node_feature_dim = 0
        self.mask_dim = 0
        self._features_dim = encode_latent_num_node
        for key, subspace in observation_space.spaces.items():
            if "node" in key:
                self.node_feature_dim = subspace.shape[0]
            elif "task_edge_mask" in key:
                self.mask_dim = subspace.shape[0]

        extractors = {}
        extractors['robot_1_node_encoder'] = nn.Linear(self.node_feature_dim, encode_latent_num_node)
        extractors['robot_2_node_encoder'] = nn.Linear(self.node_feature_dim, encode_latent_num_node)

        extractors['task_edge_encoder'] = nn.Linear(1, encode_latent_num_edge)
        extractors['coop_edge_encoder'] = nn.Linear(1, encode_latent_num_edge)

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = encode_latent_num_node

    def forward(self, observations) -> th.Tensor:
        encoded_observation = {}
        robot_1_node_tensor_list = []
        robot_2_node_tensor_list = []

        for key, obs in observations.items():

            if "robot_1_node" in key:
                node_obs = obs.reshape([obs.shape[0], 1, obs.shape[1]])
                robot_1_node_tensor_list.append(node_obs)
            elif "robot_2_node" in key:
                node_obs = obs.reshape([obs.shape[0], 1, obs.shape[1]])
                robot_2_node_tensor_list.append(node_obs)
            elif "mask" in key:
                encoded_observation[key] = obs
            elif "dist" in key:
                task_edge_obs = obs.reshape([obs.shape[0], obs.shape[1], obs.shape[2], 1])
                encoded_observation[key] = self.extractors['task_edge_encoder'](task_edge_obs)
            elif "cost" in key:
                coop_edge_obs = obs.reshape([obs.shape[0], obs.shape[1], obs.shape[2], 1])
                encoded_observation[key] = self.extractors['coop_edge_encoder'](coop_edge_obs)

        robot_1_node_tensor = th.cat(robot_1_node_tensor_list, dim=1)
        robot_2_node_tensor = th.cat(robot_2_node_tensor_list, dim=1)

        encoded_observation["robot_1_node"] = self.extractors['robot_1_node_encoder'](robot_1_node_tensor)
        encoded_observation["robot_2_node"] = self.extractors['robot_2_node_encoder'](robot_2_node_tensor)

        return encoded_observation


########################### custom distribution


from stable_baselines3.common.distributions import Distribution, MultiCategoricalDistribution, DiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from torch.distributions import Bernoulli, Categorical, Normal
import numpy as np


### multidiscrete output

# class CustomMaskedLinear(nn.Module):
#     def __init__(self, latent_dim, action_dim):
#         super(CustomMaskedLinear, self).__init__()
#         assert latent_dim - action_dim == latent_num, "latent_dim:{} and action_dim:{} must diff latent_num".format(
#             latent_dim, action_dim)
#         self.latent_dim = latent_dim
#         self.action_dim = action_dim
#         self.linear = nn.Linear(self.latent_dim - self.action_dim, self.action_dim)
#
#     def forward(self, x):
#         latent, mask = th.split(x, [self.latent_dim - self.action_dim, self.action_dim], dim=1)
#
#         output = self.linear(latent)
#         mask = th.add(mask, -1)
#         mask = mask.mul(10000)
#         output = th.add(output, mask)
#
#
#         # logits1, logits2 = th.split(logits, [self.action_dim // 2, self.action_dim // 2], dim=1)
#         # masks = th.split(mask, [self.action_dim // 2, self.action_dim // 2], dim=1)
#         # #
#         # # logits1 = logits1.mul(mask1)
#         # # logits2 = logits2.mul(mask2)
#         # #
#         # logits1 = nn.Softmax(dim=1)(logits1)
#         # logits2 = nn.Softmax(dim=1)(logits2)
#         #
#         # logits1 = logits1.mul(mask1)
#         # logits2 = logits2.mul(mask2)
#         #
#         # output = th.cat((logits1, logits2), 1)
#
#         # print("latent\t", logits[:, :])
#         # print("mask\t", mask[:, :])
#         # print("output\t", output[:, :])
#         return output
class FakeNet(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(FakeNet, self).__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.linear = None

    def forward(self, x):
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

        self.distribution = [Categorical(logits=split) for split in
                             th.split(action_logits, tuple(action_dim), dim=1)]

        # print("action logit:\t", action_logits)
        # for split in th.split(action_logits, tuple(self.action_dims), dim=1):
        #     for i in range(5):
        #         print("split,   ", split, Categorical(logits=split).sample())
        # print(d1.sample(),d2.sample())
        return self


### box output
#
# #
# class CustomMaskedLinearBox(nn.Module):
#     def __init__(self, latent_dim, action_dim):
#         super(CustomMaskedLinearBox, self).__init__()
#         assert latent_dim - action_dim == latent_num, "latent_dim:{} and action_dim:{} must diff latent_num".format(
#             latent_dim, action_dim)
#         self.latent_dim = latent_dim
#         self.action_dim = action_dim
#         self.linear = nn.Linear(self.latent_dim - self.action_dim, self.action_dim)
#
#     def forward(self, x):
#         latent, mask = th.split(x, [self.latent_dim - self.action_dim, self.action_dim], dim=1)
#
#         output = self.linear(latent)
#         mask = th.add(mask, -1)
#         mask = mask.mul(10000)
#         logits = th.add(output, mask)
#
#         logits1, logits2 = th.split(logits, [self.action_dim // 2, self.action_dim // 2], dim=1)
#         logits1 = nn.Softmax(dim=1)(logits1)
#         logits2 = nn.Softmax(dim=1)(logits2)
#
#         output = th.cat((logits1, logits2), 1)
#         return output
#
# class CustomDiagGaussianDistribution(DiagGaussianDistribution):
#     def __init__(self, action_dim: int):
#         super(CustomDiagGaussianDistribution, self).__init__(action_dim)
#
#         self.action_dim = action_dim
#         self.mean_actions = None
#         self.log_std = None
#
#     def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
#         """
#         Create the layers and parameter that represent the distribution:
#         one output will be the mean of the Gaussian, the other parameter will be the
#         standard deviation (log std in fact to allow negative values)
#         :param latent_dim: Dimension of the last layer of the policy (before the action layer)
#         :param log_std_init: Initial value for the log standard deviation
#         :return:
#         """
#         mean_actions = latent_dim#CustomMaskedLinearBox(latent_dim, self.action_dim)
#         # TODO: allow action dependent std
#         log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
#         return mean_actions, log_std

def make_custom_proba_distribution_multidiscrete(
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
        return CustomMultiCategoricalDistribution(action_space.nvec, **dist_kwargs)
    elif isinstance(action_space, spaces.Box):
        assert len(action_space.shape) == 1, "Error: the action space must be a vector"
        cls = CustomDiagGaussianDistribution
        # print(get_action_dim(action_space), cls(get_action_dim(action_space)))
        return cls(get_action_dim(action_space))
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )
