from typing import Callable, Any, Dict, List, Optional, Tuple, Type, Union

from gym import spaces
import gym
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy



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
        self.mask_dim = sum(action_space.nvec)
        # self.mask_dim = get_action_dim(action_space)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@", self.mask_dim)
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

        # Action distribution
        self.action_dist = make_custom_proba_distribution_multidiscrete(action_space, use_sde=False, dist_kwargs=None)
        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork1(self.features_dim, self.mask_dim)


latent_num = 128


class CustomNetwork1(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            mask_dim: int,
            last_layer_dim_pi: int = latent_num,
            last_layer_dim_vf: int = latent_num,
    ):
        super(CustomNetwork1, self).__init__()

        # IMPORTANT:
        self.feature_dim = feature_dim
        self.mask_dim = mask_dim
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi + self.mask_dim
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, latent_num), nn.ReLU(),
            nn.Linear(latent_num, last_layer_dim_pi), nn.ReLU(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, latent_num), nn.ReLU(),
            nn.Linear(latent_num, last_layer_dim_vf), nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        _, mask = th.split(features, [self.feature_dim - self.mask_dim, self.mask_dim], dim=1)
        output_pi = self.policy_net(features)
        output_pi = th.cat((output_pi, mask), 1)
        return output_pi, self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        _, mask = th.split(features, [self.feature_dim - self.mask_dim, self.mask_dim], dim=1)
        output_pi = self.policy_net(features)
        output_pi = th.cat((output_pi, mask), 1)
        return output_pi

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


from stable_baselines3.common.distributions import Distribution, MultiCategoricalDistribution, DiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from torch.distributions import Bernoulli, Categorical, Normal
import numpy as np


class CustomMaskedLinear(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(CustomMaskedLinear, self).__init__()
        assert latent_dim - action_dim == latent_num, "latent_dim:{} and action_dim:{} must diff latent_num".format(
            latent_dim, action_dim)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.linear = nn.Linear(self.latent_dim - self.action_dim, self.action_dim)

    def forward(self, x):
        latent, mask = th.split(x, [self.latent_dim - self.action_dim, self.action_dim], dim=1)

        output = self.linear(latent)
        mask = th.add(mask, -1)
        mask = mask.mul(10000)
        output = th.add(output, mask)


        # logits1, logits2 = th.split(logits, [self.action_dim // 2, self.action_dim // 2], dim=1)
        # masks = th.split(mask, [self.action_dim // 2, self.action_dim // 2], dim=1)
        # #
        # # logits1 = logits1.mul(mask1)
        # # logits2 = logits2.mul(mask2)
        # #
        # logits1 = nn.Softmax(dim=1)(logits1)
        # logits2 = nn.Softmax(dim=1)(logits2)
        #
        # logits1 = logits1.mul(mask1)
        # logits2 = logits2.mul(mask2)
        #
        # output = th.cat((logits1, logits2), 1)

        # print("latent\t", logits[:, :])
        # print("mask\t", mask[:, :])
        # print("output\t", output[:, :])
        return output


class CustomMultiCategoricalDistribution(MultiCategoricalDistribution):
    def __init__(self, action_dim: int):
        super(CustomMultiCategoricalDistribution, self).__init__(action_dim)

        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.
        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = CustomMaskedLinear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "MultiCategoricalDistribution":
        self.distribution = [Categorical(logits=split) for split in
                             th.split(action_logits, tuple(self.action_dims), dim=1)]

        # print("action logit:\t", action_logits)
        # for split in th.split(action_logits, tuple(self.action_dims), dim=1):
        #     for i in range(5):
        #         print("split,   ", split, Categorical(logits=split).sample())
                # print(d1.sample(),d2.sample())
        return self







class CustomMaskedLinearBox(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(CustomMaskedLinearBox, self).__init__()
        assert latent_dim - action_dim == latent_num, "latent_dim:{} and action_dim:{} must diff latent_num".format(
            latent_dim, action_dim)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.linear = nn.Linear(self.latent_dim - self.action_dim, self.action_dim)

    def forward(self, x):
        latent, mask = th.split(x, [self.latent_dim - self.action_dim, self.action_dim], dim=1)

        output = self.linear(latent)
        mask = th.add(mask, -1)
        mask = mask.mul(10000)
        logits = th.add(output, mask)

        logits1, logits2 = th.split(logits, [self.action_dim // 2, self.action_dim // 2], dim=1)
        logits1 = nn.Softmax(dim=1)(logits1)
        logits2 = nn.Softmax(dim=1)(logits2)

        output = th.cat((logits1, logits2), 1)
        return output

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
        mean_actions = CustomMaskedLinearBox(latent_dim, self.action_dim)
        # TODO: allow action dependent std
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

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
