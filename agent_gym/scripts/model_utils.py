from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
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
            last_layer_dim_pi: int = 256,
            last_layer_dim_vf: int = 512,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi * 2
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net1 = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(),
            nn.Linear(512, last_layer_dim_pi), nn.ReLU(),
        )
        self.policy_net2 = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(),
            nn.Linear(512, last_layer_dim_pi), nn.ReLU(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(),
            nn.Linear(256, last_layer_dim_vf), nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        output1 = self.policy_net1(features)
        output2 = self.policy_net2(features)
        output_pi = th.cat((output1, output2), 1)
        return output_pi, self.value_net(features)
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        output1 = self.policy_net1(features)
        output2 = self.policy_net2(features)
        output_pi = th.cat((output1, output2), 1)
        return output_pi

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


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
        self.action_dist = make_custom_proba_distribution(action_space, use_sde=False, dist_kwargs=None)
        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


# model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1)
# model.learn(5000)


from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.distributions import Distribution, DiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from gym import spaces

class CustomSquashedLinear(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(CustomSquashedLinear, self).__init__()
        assert (latent_dim % 2) == 0 and (action_dim % 2) == 0, "latent_dim:{} and action_dim:{} must be even".format(
            latent_dim, action_dim)
        self.half_latent = latent_dim // 2
        self.half_action = action_dim // 2
        self.linear_left = nn.Linear(self.half_latent, self.half_action)
        self.linear_right = nn.Linear(self.half_latent, self.half_action)

    def forward(self, x):
        # print(x.size())
        act_left = self.linear_left(x[:,:self.half_latent])
        act_right = self.linear_right(x[:,self.half_latent:])
        act = th.cat((act_left, act_right), 1)
        # print(act.size())
        output = nn.Tanh()(act)
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
        # print("hhhhhhhhhhhhhh")
        mean_actions = CustomSquashedLinear(latent_dim, self.action_dim)
        # from torchinfo import summary
        # summary(mean_actions)
        # TODO: allow action dependent std
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std


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

    if isinstance(action_space, spaces.Box):
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




from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):

        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            # print("total_concat_size:\t",key, subspace,total_concat_size)
            if "img" in key:
                n_input_channels = subspace.shape[0]
                # cnn = nn.Sequential(
                #     nn.Conv2d(n_input_channels, 4, kernel_size=8, stride=4, padding=0),
                #     nn.ReLU(),
                #     nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=0),
                #     nn.ReLU(),
                #     nn.Flatten(),
                # )
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 4, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(4, 16, kernel_size=4, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                with th.no_grad():
                    n_flatten = cnn(
                        th.as_tensor(subspace.sample()[None]).float()
                    ).shape[1]
                mlp = nn.Sequential(nn.Linear(n_flatten, 128), nn.ReLU())

                extractors[key] = nn.Sequential(cnn, mlp)
                total_concat_size += 128

            elif "robot" in key:

                extractors[key] = nn.Linear(subspace.shape[0], 128)
                total_concat_size += 128

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        return th.cat(encoded_tensor_list, dim=1)


class CustomLinkExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):

        super(CustomLinkExtractor, self).__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            # print("total_concat_size:\t",key, subspace,total_concat_size)
            if "link" in key:
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16
            else:
                extractors[key] = nn.Linear(subspace.shape[0], 64)
                total_concat_size += 64
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
            # if "link" in key:
            #     encoded_tensor_list.append(extractor(observations[key]))
            # else:
            #     encoded_tensor_list.append(observations[key])

        return th.cat(encoded_tensor_list, dim=1)