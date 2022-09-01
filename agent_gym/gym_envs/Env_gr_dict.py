import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '../../')

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import random
import pybullet_data
from pkg_resources import normalize_path, parse_version

from bullet_env.robot import Robot
from bullet_env.part import Part
from bullet_env.p_utils import (
    create_cube_object,
    draw_sphere_body,
    draw_cube_body,
    remove_markers,
    remove_obstacles
)
from bullet_env import tools
from bullet_env.plot import Robot_Triangle_Plot

import Env_gr

largeValObservation = np.inf  ###############


class Env_gr_dict(Env_gr.Env_gr):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 env_config,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 useInverseKinematics=None,
                 renders=False,
                 evaluate=False,
                 showBallMarkers=False,
                 isDiscrete=False,
                 freezeAction=False,
                 maxSteps=500,
                 reward_type="delta_dist_with_sparse_reward",
                 obs_type="common_obs",
                 action_type="js",
                 in_task=False,
                 # use_plot=True,
                 image_width=128,
                 image_height=128,
                 share_image=True
                 ):

        self.image_width = image_width
        self.image_height = image_height
        self._share_image = share_image
        if self._share_image:
            self.image_num = 1
            self.image_names = ["img_{}".format(0)]
        else:
            self.image_num = len(self.robots_num)
            self.image_names = ["img_{}".format(i) for i in range(self.image_num)]

        super(Env_gr_dict, self).__init__(env_config,
                                          urdfRoot=urdfRoot,
                                          actionRepeat=actionRepeat,
                                          isEnableSelfCollision=isEnableSelfCollision,
                                          useInverseKinematics=useInverseKinematics,
                                          renders=renders,
                                          evaluate=evaluate,
                                          showBallMarkers=showBallMarkers,
                                          isDiscrete=isDiscrete,
                                          freezeAction=freezeAction,
                                          maxSteps=maxSteps,
                                          reward_type=reward_type,
                                          obs_type=obs_type,
                                          action_type=action_type,
                                          in_task=in_task,
                                          use_plot=True)

    def get_observation_space(self):
        observation_space = None
        observation = self._observation()

        obs_space_dict = dict(
            # robot_1=spaces.Box(-np.inf, np.inf, shape=observation['robot_1'].shape),
            # robot_2=spaces.Box(-np.inf, np.inf, shape=observation['robot_2'].shape),
            base=spaces.Box(-np.inf, np.inf, shape=observation['base'].shape), )

        # imgs_obs_space_dict = {img_name: spaces.Box(-np.inf, np.inf, shape=observation[img_name].shape) for img_name in
        #                        self.image_names}
        # obs_space_dict.update(imgs_obs_space_dict)

        link_dist_obs_dict = dict(link_dist=spaces.Box(-np.inf, np.inf, shape=observation['link_dist'].shape))
        obs_space_dict.update(link_dist_obs_dict)

        link_pose_obs_dict = dict(link_pose=spaces.Box(-np.inf, np.inf, shape=observation['link_pose'].shape))
        obs_space_dict.update(link_pose_obs_dict)

        observation_space = spaces.Dict(obs_space_dict)
        return observation_space

    def _observation(self):
        observation = {}

        obs_list = self.get_states()
        base_obs = []
        link_dist_obs = []
        link_pose_obs = []
        for i, robot in enumerate(self.robots):
            base_obs.append(self.obs_info["base_obs_{}".format(i + 1)])
            link_dist_obs.append(self.obs_info["link_dist_obs_{}".format(i + 1)])
            link_pose_obs.append(self.obs_info["link_pose_obs_{}".format(i + 1)])

        base_obs = np.concatenate(base_obs)
        observation['base'] = base_obs
        link_dist_obs = np.concatenate(link_dist_obs)
        observation['link_dist'] = link_dist_obs
        link_pose_obs = np.concatenate(link_pose_obs)
        observation['link_pose'] = link_pose_obs


        # obs_list = self.get_states()
        # robot_states = np.concatenate([robot_obs for robot_obs in obs_list])
        #
        # if self._use_plot:
        #     tri_img = self.plot.output_img_matrix()
        #     if self._renders:
        #         self.plot.show_img_matrix()
        #
        # for img_name in self.image_names:
        #     img = self.get_image_matrix()
        #     observation[img_name] = img
        # observation['robots'] = robot_states

        return observation

    def get_image_matrix(self):
        img = np.expand_dims(self.plot.img_matrix, axis=0)
        img /= 255
        return img

    def create_plot(self):
        plot = Robot_Triangle_Plot(img_width=self.image_width, img_height=self.image_height,
                                   fill_triangle=self._fill_triangle)
        for i, robot in enumerate(self.robots):
            plot.create_robot_triangle(i)
        plot.create_plot()
        return plot
