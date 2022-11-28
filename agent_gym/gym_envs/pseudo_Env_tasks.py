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

from Env_gr import Env_gr
from bullet_env import tools
from bullet_env.pseudo_robot import PseudoRobot, PseudoPart





maximum_cost = 10000
largeValObservation = np.inf  ###############


class Env_tasks(gym.GoalEnv):
    metadata = {}

    def __init__(self,
                 env_config,
                 urdfRoot=pybullet_data.getDataPath(),
                 useInverseKinematics=None,
                 renders=False,
                 task_allocator_action_type="MultiDiscrete",
                 task_allocator_reward_type=None,
                 task_allocator_obs_type=None,
                 # motion_planner_reward_type="delta_dist_with_sparse_reward",
                 # motion_planner_obs_type="common_obs",
                 # motion_planner_action_type="ee",
                 parts_num=6,
                 maxSteps=50):

        self._urdfRoot = urdfRoot
        self._useInverseKinematics = useInverseKinematics

        self._renders = renders
        self._maxSteps = maxSteps
        self.parts_num = parts_num

        self.action_type = task_allocator_action_type
        self.task_allocator_reward_type = task_allocator_reward_type
        self.task_allocator_obs_type = task_allocator_obs_type
        # self.motion_planner_reward_type = motion_planner_reward_type
        # self.motion_planner_obs_type = motion_planner_obs_type
        # self.motion_planner_action_type = motion_planner_action_type
        # self.success_dist_threshold = env_config['success_dist_threshold'] + 0.1
        # self.dead_lock_count_threshold = 400 / fragment_length
        # self.use_reset = False

        self.lift_height = 0.15
        self.sync_execution = True
        self.selected_planner = None#self.go_straight_planner


        # load configs
        self._partsBaseSize = env_config['partsBaseSize']
        self._partsBasePos = env_config['partsBasePos']
        self._partsBaseOrn = env_config['partsBaseOrn']
        self._partsBaseColor = env_config['partsBaseColor']

        self._beltBaseSize = env_config['beltBaseSize']
        self._beltBasePos = env_config['beltBasePos']
        self._beltBaseOrn = env_config['beltBaseOrn']
        self._beltBaseColor = env_config['beltBaseColor']




        ######### define gym conditions

        # pass
        self.seed()
        self.setup()

        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

    def get_observation_space(self):
        observation_dim = len(self._observation())
        observation_high = np.array([largeValObservation] * observation_dim)
        observation_space = spaces.Box(-observation_high, observation_high)
        return observation_space

    def get_action_space(self):
        if self.action_type == "MultiDiscrete":
            action_space = spaces.MultiDiscrete([self.action_dim] * self.robots_num)
        elif self.action_type == "Box":
            lower_bound = np.array([0] * (self.action_dim * self.robots_num))
            upper_bound = np.array([1] * (self.action_dim * self.robots_num))
            action_space = spaces.Box(lower_bound, upper_bound)
        else:
            print("action_type goes wrong: {}".format(self.action_type))
            action_space = None
        return action_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def setup(self):
        # load pseudo robots
        self.robots = []
        self._robot1 = PseudoRobot(robotName="abb", useInverseKinematics=self._useInverseKinematics)
        self.robots.append(self._robot1)
        self._robot2 = PseudoRobot(robotName="kawasaki", useInverseKinematics=self._useInverseKinematics)
        self.robots.append(self._robot2)

        self.global_workspace = np.array([[0., 0.], [0., 0.], [0.5, 0.5]])
        for i, robot in enumerate(self.robots):
            self.update_workspace(robot)

        # load pseudo parts
        self.parts = []
        for k in range(self.parts_num):
            part = PseudoPart()
            self.parts.append(part)

        self.robots_num = len(self.robots)
        self.parts_num = len(self.parts)

        self.reset()

    def update_workspace(self, robot=None):
        if robot is not None:
            for k in range(len(self.global_workspace)):
                self.global_workspace[k, 0] = min(self.global_workspace[k, 0], robot.workspace[k, 0])
                self.global_workspace[k, 1] = max(self.global_workspace[k, 1], robot.workspace[k, 1])
        return self.global_workspace

    def reset(self):
        self.terminated = 0
        self.success = 0
        self.fail = 0
        self.env_step_counter = 0
        self.state_info = {}
        self.prev_state_info = {}

        self.accumulated_reward = 0
        self.accumulated_reward_dict = {}
        self.accumulated_reward_dict['robots'] = [0] * self.robots_num
        self.accumulated_reward_dict['parts'] = [0] * self.parts_num
        self.accumulated_reward_dict['task'] = 0

        ######## robot reset#########################################################################################
        for robot in self.robots:
            robot.mode.initialize_mode()
            self.apply_placing_item(robot)
        self.robots_env.reset()
        for robot in self.robots:
            robot.resetGoalPose(default_pose=False)
            robot.resetInitPose(default_pose=False)
            robot.success_count = 0
            robot.task_switched = 0
            robot.mode_switched = 0
            robot.slack_step = 0
        ######## parts reset
        goal_poses_base = [[-0.7, -0.5], [-0.7, 0], [-0.7, 0.5], [0.7, -0.5], [0.7, 0], [0.7, 0.5]]
        init_poses_base = [[-0.25, -0.5], [-0.25, 0], [-0.25, 0.5], [0.25, -0.5], [0.25, 0], [0.25, 0.5]]
        random.shuffle(goal_poses_base)
        random.shuffle(init_poses_base)
        ######################################################
        if np.random.random() < 0.5:
            all_poses_base = goal_poses_base + init_poses_base
            random.shuffle(all_poses_base)
            init_poses_base = all_poses_base[:self.parts_num]
            goal_poses_base = all_poses_base[self.parts_num:]
        ######################################################
        for j, part in enumerate(self.parts):
            #### set part init pose
            init_pose_base = init_poses_base[j]
            init_x = init_pose_base[0] + (np.random.random() - 0.5) / 10.
            init_y = init_pose_base[1] + (np.random.random() - 0.5) / 10.
            init_pos = [init_x, init_y, self.base_height]
            init_rz = (np.random.random() - 0.5) * math.pi / 10
            init_orn = p.getQuaternionFromEuler([0, 0, init_rz])
            part.resetInitPose(init_pos, init_orn)
            #### set parts goal pose
            goal_pose_base = goal_poses_base[j]
            goal_x = goal_pose_base[0] + (np.random.random() - 0.5) / 100.
            goal_y = goal_pose_base[1] + (np.random.random() - 0.5) / 100.
            goal_pos = [goal_x, goal_y, self.base_height]
            goal_orn = p.getQuaternionFromEuler([0, 0, 0])
            part.resetGoalPose(goal_pos, goal_orn)

            part.reset()
        ######## visualize goal
        for i, part in enumerate(self.parts):
            if self._renders:
                goal_pos_orn = part.getGoalPose(useInverseKinematics=False)
                target = draw_cube_body(goal_pos_orn[:3], goal_pos_orn[3:], [0.03, 0.02, 0.0025], part.color)
                self.markers.append(target)

        ######## get obs and update state history
        self.observation_history = []
        self.state_info = {}
        obs = self._observation()
        self.prev_state_info = self.state_info



        return obs



    def get_states(self):

        return observation

    def step(self, allocator_action):

        return observation, reward, done, episode_info

    def _observation(self):
        obs_dict = self.get_states()
        observation = np.concatenate(obs_dict)

        return observation

    def _reward(self):

        return reward

    def _termination(self):

        return None

    def sample_allocator_action(self, allocator_action):
        assert len(allocator_action) == self.action_dim * self.robots_num
        sampled_allocator_action = np.array([0] * self.robots_num)
        for i in range(self.robots_num):
            robot_action = allocator_action[i:i + self.action_dim]
            if sum(robot_action) > 0:
                idx = random.choices(np.arange(self.action_dim), weights=robot_action, k=1)[0]
                sampled_allocator_action[i] = idx

        return sampled_allocator_action

    def allocate_task(self, allocator_action):



    def assign_robot_goal(self, robot, task_part=None):



    def execute_planner(self, customize_fragment_length=None):



    def assign_planners(self):

        return robots_modes, robots_planners




    ####################################### task allocators ################################
    def distance_based_allocator(self, obs=None):
        ######## allocate closest tasks to robots
        selectable_part_idx = list(range(self.parts_num))
        for j in selectable_part_idx[::-1]:
            part = self.parts[j]
            if part.is_success or part.is_in_process:
                selectable_part_idx.remove(selectable_part_idx[j])
        # print(selectable_part_idx)
        if obs is None:
            return np.array([0] * self.parts_num * self.robots_num)
        dist_cost = obs
        dist_dict = {}
        for i, robot in enumerate(self.robots):
            if robot.is_in_process:
                # robot have target task and is in process, keep target
                idx = robot.task_idx_allocated
                dist_dict[i] = idx
        while len(dist_dict) < self.robots_num and len(selectable_part_idx) > 0:
            # print("selectable idx:\t", selectable_part_idx)
            # print("dist dict:\t", dist_dict)
            for i, robot in enumerate(self.robots):
                if i in dist_dict:
                    continue
                else:
                    idx_with_min_dist = selectable_part_idx[0]
                    min_dist = 10000
                    for idx in selectable_part_idx:
                        dist = dist_cost["dist_robot_{}_to_part_{}".format(i + 1, idx + 1)]
                        if dist <= min_dist:
                            min_dist = dist
                            idx_with_min_dist = idx
                    if idx_with_min_dist in dist_dict.values():
                        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        other_robot_idx = [k for k, v in dist_dict.items() if v == idx_with_min_dist][0]

                        robot_dist = dist_cost["dist_robot_{}_to_part_{}".format(i + 1, idx_with_min_dist + 1)]
                        other_robot_dist = dist_cost[
                            "dist_robot_{}_to_part_{}".format(other_robot_idx + 1, idx_with_min_dist + 1)]
                        if robot_dist < other_robot_dist:
                            # print("switch!")
                            dist_dict.pop(other_robot_idx)
                            dist_dict[i] = idx_with_min_dist
                        selectable_part_idx.remove(idx_with_min_dist)
                    else:
                        dist_dict[i] = idx_with_min_dist

        allocator_action = []
        for i, robot in enumerate(self.robots):
            if i in dist_dict:
                idx = dist_dict[i]
            else:
                idx = -1
            allocator_action.append(idx + 1)
        # print("allocator action:\t", allocator_action)
        return allocator_action

    def random_allocator(self, obs=None):
        ######## random allocate tasks to robots
        selectable_part_idx = list(range(self.parts_num))
        for j in selectable_part_idx[::-1]:
            part = self.parts[j]
            if part.is_success or part.is_in_process:
                selectable_part_idx.remove(selectable_part_idx[j])
        # print(selectable_part_idx)
        allocator_action = []
        for i, robot in enumerate(self.robots):
            act = [0] * self.parts_num
            if robot.is_in_process:
                # robot have target task and is in process, keep target
                idx = robot.task_idx_allocated
            elif robot.task_idx_allocated in selectable_part_idx and random.random() < 0.75:
                # robot have target selectable task but not in process, keep target
                idx = robot.task_idx_allocated
            elif len(selectable_part_idx) > 0:
                # robot do not have selectable target task and not in process, random choose a selectable task as target
                idx = random.choice(selectable_part_idx)
                selectable_part_idx.remove(idx)
            else:
                idx = -1
            allocator_action.append(idx + 1)
        # print("allocator action:\t", allocator_action)
        return allocator_action


    def __del__(self):
        p.disconnect()
