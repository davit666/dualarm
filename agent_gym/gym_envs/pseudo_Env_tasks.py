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
                 useInverseKinematics=True,
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

        self.sync_execution = True
        self.selected_planner = None  # self.go_straight_planner

        # load configs
        self._partsBaseSize = env_config['partsBaseSize']
        self._partsBasePos = env_config['partsBasePos']
        self._partsBaseOrn = env_config['partsBaseOrn']
        self._partsBaseColor = env_config['partsBaseColor']

        self._beltBaseSize = env_config['beltBaseSize']
        self._beltBasePos = env_config['beltBasePos']
        self._beltBaseOrn = env_config['beltBaseOrn']
        self._beltBaseColor = env_config['beltBaseColor']

        self.max_cost_const = env_config['max_cost_const'] if env_config['use_prediction_model'] else 2
        self.global_success_bonus = env_config['global_success_bonus']
        self.reward_scale = env_config['reward_scale']

        self.task_type = env_config['task_type']
        self.dynamic_task = env_config['dynamic_task']

        self.robot_done_freeze = env_config['robot_done_freeze']
        self.use_prediction_model = env_config['use_prediction_model']

        ######### define gym conditions

        # pass
        self.base_height = self._partsBaseSize[2] * 2
        self.action_dim = self.parts_num + 1

        self.seed()
        self.setup()

        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

    def get_observation_space(self):
        obs_space_dict = dict()

        observation = self._observation()
        for key in observation.keys():
            space_dict = dict()
            space_dict[key] = spaces.Box(-np.inf, np.inf, shape=observation[key].shape)
            obs_space_dict.update(space_dict)

        observation_space = spaces.Dict(obs_space_dict)
        return observation_space

    def get_action_space(self):
        if self.action_type == "MultiDiscrete":
            action_space = spaces.MultiDiscrete([self.action_dim] * self.robots_num)
        elif self.action_type == "Discrete":
            action_space = spaces.Discrete((self.action_dim ** self.robots_num))
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
        self._robot1 = PseudoRobot(robotName="abb")
        self.robots.append(self._robot1)
        self._robot2 = PseudoRobot(robotName="kawasaki")
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
        self.terminated = False
        self.success = False
        self.failed = False
        self.succ_parts_num = 0
        self.env_step_counter = 0
        self.state_info = {}
        self.prev_state_info = {}

        self.accumulated_reward = 0
        self.accumulated_cost = 0
        # self.accumulated_reward_dict['robots'] = [0] * self.robots_num
        # self.accumulated_reward_dict['parts'] = [0] * self.parts_num
        # self.accumulated_reward_dict['task'] = 0
        self.prediction_updated = False

        ######## robot reset#########################################################################################
        for robot in self.robots:
            robot.reset()
            robot.tasks_done = 0
            robot.wrong_allocation = 0
            robot.freeze_step = 0
            robot.is_done = False

        ######## parts reset
        goal_poses_base = [[-0.7, -0.5], [-0.7, 0], [-0.7, 0.5], [0.7, -0.5], [0.7, 0], [0.7, 0.5]]
        init_poses_base = [[-0.25, -0.5], [-0.25, 0], [-0.25, 0.5], [0.25, -0.5], [0.25, 0], [0.25, 0.5]]

        # global_poses_base = goal_poses_base + init_poses_base
        # random.shuffle(global_poses_base)
        # init_poses_base = global_poses_base[:self.parts_num]
        # goal_poses_base = global_poses_base[self.parts_num:]

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

        ######## get obs and update state history
        self.observation_history = []
        self.state_info = {}
        obs = self._observation()
        self.prev_state_info = self.state_info.copy()

        return obs

    def _observation(self):
        state_dict = self.get_states()
        robot_nodes = state_dict["robots_nodes"]
        robots_task_edge_dist = state_dict['robots_task_edge_xyz']
        robots_task_edge_mask = state_dict['robots_task_edge_mask']
        coop_edge_cost = state_dict["coop_edge_cost"]
        coop_edge_mask = state_dict['coop_edge_mask']
        prediction_inputs = state_dict['prediction_inputs']

        observation = {}
        for i in range(self.robots_num):
            for j in range(self.parts_num):
                observation['robot_{}_node_{}'.format(i + 1, j + 1)] = robot_nodes[i][j].astype(np.float32)
            observation['robot_{}_node_reset'.format(i + 1)] = robot_nodes[i][self.parts_num].astype(np.float32)
            observation['robot_{}_task_edge_dist'.format(i + 1)] = robots_task_edge_dist[i].astype(np.float32)
            observation['robot_{}_task_edge_mask'.format(i + 1)] = robots_task_edge_mask[i].astype(np.float32)
        observation['coop_edge_cost'] = coop_edge_cost.astype(np.float32)
        observation['coop_edge_mask'] = coop_edge_mask.astype(np.float32)
        observation['prediction_inputs'] = prediction_inputs.astype(np.float32)

        return observation

    def get_states(self):
        self.state_info = {}
        # robot obs
        robots_ee = []
        robots_goal = []
        robots_mask = []
        for i, robot in enumerate(self.robots):
            robot_ee = robot.getObservation_EE()
            robot_goal = robot.goal_pose
            robot_done = robot.is_done

            self.state_info["robot_ee_{}".format(i + 1)] = robot_ee
            self.state_info["robot_goal_{}".format(i + 1)] = robot_goal
            self.state_info["robot_done_{}".format(i + 1)] = robot_done

            # normalize
            norm_robot_ee = self.normalize_cartesian_pose(robot_ee)
            norm_robot_goal = self.normalize_cartesian_pose(robot_goal)

            robots_ee.append(norm_robot_ee)
            robots_goal.append(norm_robot_goal)
            robots_mask.append(1 - robot_done)

        # part obs
        parts_init = []
        parts_goal = []
        parts_mask = []
        for j, part in enumerate(self.parts):
            part_init = part.getPose()
            part_goal = part.getGoalPose()
            part_done = part.is_success

            # part_init[2] += (np.random.random() - 0.5) / 50
            part_goal[2] += (np.random.random() - 0.5) / 50

            self.state_info["part_init_{}".format(j + 1)] = part_init
            self.state_info["part_goal_{}".format(j + 1)] = part_goal
            self.state_info["part_done_{}".format(j + 1)] = part_done

            # normalize
            norm_part_init = self.normalize_cartesian_pose(part_init)
            norm_part_goal = self.normalize_cartesian_pose(part_goal)

            parts_init.append(norm_part_init)
            parts_goal.append(norm_part_goal)
            parts_mask.append(1 - part_done)

        # node_obs
        robots_nodes = []
        for i in range(self.robots_num):
            robot_nodes = []
            # part node
            for j in range(self.parts_num):
                # robot 2 part init
                init_ee1 = robots_ee[i]
                goal_ee1 = parts_init[j]
                dist_xyz1 = np.linalg.norm(init_ee1[:3] - goal_ee1[:3])
                dist_rz1 = np.linalg.norm(init_ee1[3:] - goal_ee1[3:])
                obs1 = np.concatenate([init_ee1, goal_ee1, [dist_xyz1], [dist_rz1]])
                # part init 2 part goal
                init_ee2 = parts_init[j]
                goal_ee2 = parts_goal[j]
                dist_xyz2 = np.linalg.norm(init_ee2[:3] - goal_ee2[:3])
                dist_rz2 = np.linalg.norm(init_ee2[3:] - goal_ee2[3:])
                obs2 = np.concatenate([init_ee2, goal_ee2, [dist_xyz2], [dist_rz2]])
                # others
                node_type = np.array([0, 1])
                # node_mask = robots_mask[i] * parts_mask[j]
                node_mask = parts_mask[j]

                node_obs = np.concatenate([obs1, obs2, node_type, [node_mask], [robots_mask[i]]])
                self.state_info["robot_{}_node_obs_{}".format(i + 1, j + 1)] = node_obs
                self.state_info["robot_{}_node_obs1_{}".format(i + 1, j + 1)] = obs1
                self.state_info["robot_{}_node_obs2_{}".format(i + 1, j + 1)] = obs2
                self.state_info["robot_{}_node_mask_{}".format(i + 1, j + 1)] = node_mask

                robot_nodes.append(node_obs)

            # robot reset node
            # robot 2 reset_pose
            init_ee1 = robots_ee[i]
            goal_ee1 = robots_goal[i]
            dist_xyz1 = np.linalg.norm(init_ee1[:3] - goal_ee1[:3])
            dist_rz1 = np.linalg.norm(init_ee1[3:] - goal_ee1[3:])
            obs1 = np.concatenate([init_ee1, goal_ee1, [dist_xyz1], [dist_rz1]])
            # freeze
            init_ee2 = robots_goal[i]
            goal_ee2 = robots_goal[i]
            dist_xyz2 = np.linalg.norm(init_ee2[:3] - goal_ee2[:3])
            dist_rz2 = np.linalg.norm(init_ee2[3:] - goal_ee2[3:])
            obs2 = np.concatenate([init_ee2, goal_ee2, [dist_xyz2], [dist_rz2]])
            # others
            node_type = np.array([1, 0])
            node_mask = 1

            reset_node_obs = np.concatenate([obs1, obs2, node_type, [node_mask], [robots_mask[i]]])
            self.state_info["robot_{}_reset_node_obs".format(i + 1)] = reset_node_obs
            self.state_info["robot_{}_node_obs1_{}".format(i + 1, self.parts_num + 1)] = obs1
            self.state_info["robot_{}_node_obs2_{}".format(i + 1, self.parts_num + 1)] = obs2

            robot_nodes.append(reset_node_obs)
            robots_nodes.append(robot_nodes)

        # task_edge_obs
        robots_task_edge_xyz = []
        robots_task_edge_rz = []
        robots_task_edge_mask = []
        for i in range(self.robots_num):
            task_edge_xyz = np.zeros((self.parts_num + 1, self.parts_num + 1))
            task_edge_rz = np.zeros((self.parts_num + 1, self.parts_num + 1))

            for m in range(self.parts_num):
                # edge: to parts
                for n in range(self.parts_num):
                    if n == m:
                        pass
                    else:
                        init_ee1 = parts_goal[m]
                        goal_ee1 = parts_init[n]
                        task_edge_xyz[m, n] = np.linalg.norm(init_ee1[:3] - goal_ee1[:3])
                        task_edge_rz[m, n] = np.linalg.norm(init_ee1[3:] - goal_ee1[3:])

                # edge: to reset
                init_ee1 = parts_goal[m]
                goal_ee1 = robots_goal[i]
                task_edge_xyz[m, self.parts_num] = np.linalg.norm(init_ee1[:3] - goal_ee1[:3])
                task_edge_rz[m, self.parts_num] = np.linalg.norm(init_ee1[3:] - goal_ee1[3:])

            task_edge_mask = np.array([1] * (self.parts_num + 1))
            for n in range(self.parts_num):
                # task_edge_mask[n] = robots_mask[i] * parts_mask[n]
                task_edge_mask[n] = parts_mask[n]

            self.state_info["robot_{}_task_edge_xyz".format(i + 1)] = task_edge_xyz
            self.state_info["robot_{}_task_edge_rz".format(i + 1)] = task_edge_rz
            self.state_info["robot_{}_task_edge_mask".format(i + 1)] = task_edge_mask

            robots_task_edge_xyz.append(task_edge_xyz)
            robots_task_edge_rz.append(task_edge_rz)
            robots_task_edge_mask.append(task_edge_mask)

        # cost_edge_initialization
        coop_edge_cost = np.ones((self.parts_num + 1, self.parts_num + 1)) * self.max_cost_const
        coop_edge_mask = np.ones((self.parts_num + 1, self.parts_num + 1))
        for m in range(self.parts_num + 1):
            for n in range(self.parts_num + 1):
                # if self.parts[m].is_success or self.parts[n].is_success or m == n:
                #     coop_edge_mask[m,n] = 0
                coop_edge_mask[m, n] = robots_task_edge_mask[0][m] * robots_task_edge_mask[1][n]
                if m == n and m < self.parts_num:
                    coop_edge_mask[m, m] = 0
                if self.robot_done_freeze:
                    if not robots_mask[0] and m < self.parts_num:
                        coop_edge_mask[m, n] = 0
                    if not robots_mask[1] and n < self.parts_num:
                        coop_edge_mask[m, n] = 0

                if not self.use_prediction_model and coop_edge_mask[m, n]:
                    dist1 = robots_nodes[0][m][8] + robots_nodes[0][m][18]
                    dist2 = robots_nodes[1][n][8] + robots_nodes[1][n][18]
                    coop_edge_cost[m, n] = max(dist1, dist2)

        self.state_info["robots_nodes"] = robots_nodes
        self.state_info['robots_task_edge_xyz'] = robots_task_edge_xyz
        self.state_info['robots_task_edge_rz'] = robots_task_edge_rz
        self.state_info['robots_task_edge_mask'] = robots_task_edge_mask
        self.state_info["coop_edge_cost"] = coop_edge_cost
        self.state_info['coop_edge_mask'] = coop_edge_mask

        #### prediction model input
        self.state_info["prediction_inputs"] = self.get_prediction_obs()
        self.state_info["prediction_inputs_shape"] = self.state_info["prediction_inputs"].shape
        self.prediction_updated = False

        return self.state_info

    def get_prediction_obs(self):
        prediction_inputs_obs1 = []
        prediction_inputs_obs2 = []
        for m in range(self.parts_num + 1):
            for n in range(self.parts_num + 1):
                robot1_obs1 = self.state_info["robot_{}_node_obs1_{}".format(1, m + 1)]
                robot2_obs1 = self.state_info["robot_{}_node_obs1_{}".format(2, n + 1)]
                input_obs1 = np.concatenate([robot1_obs1, robot2_obs1])
                prediction_inputs_obs1.append(input_obs1)

                robot1_obs2 = self.state_info["robot_{}_node_obs2_{}".format(1, m + 1)]
                robot2_obs2 = self.state_info["robot_{}_node_obs2_{}".format(2, n + 1)]
                input_obs2 = np.concatenate([robot1_obs2, robot2_obs2])
                prediction_inputs_obs2.append(input_obs2)
        # for p in prediction_inputs_obs1:
        #     print(p.shape)
        # print(prediction_inputs_obs1)
        # print(prediction_inputs_obs2)
        prediction_inputs_obs1 = np.stack(prediction_inputs_obs1, axis=0)
        prediction_inputs_obs2 = np.stack(prediction_inputs_obs2, axis=0)

        prediction_inputs = np.stack([prediction_inputs_obs1, prediction_inputs_obs2], axis=0)
        return prediction_inputs.astype(np.float32)

    def update_prediction(self, data):
        if self._renders:
            print("prediction updated")

        cost = data[0]
        mask = data[1]

        # print("cost",cost)
        # print("mask",mask)
        #
        # self.state_info["coop_edge_mask"] = np.multiply(mask, self.state_info["coop_edge_mask"])
        # self.state_info["coop_edge_cost"] = np.multiply(cost, self.state_info["coop_edge_mask"]) + self.max_cost_const * (1 - self.state_info["coop_edge_mask"])

        self.state_info["coop_edge_mask"] = mask
        self.state_info["coop_edge_cost"] = cost
        # self.state_info["coop_edge_cost"][-1, -1] = 1
        self.prediction_updated = True
        return self.prediction_updated

    def normalize_cartesian_pose(self, cartesian_pose, normalize_orn=False):
        if len(cartesian_pose) == 0:
            return None
        normalized_pose = cartesian_pose.copy()
        for k in range(len(normalized_pose)):
            if k < 3:
                mean = (self.global_workspace[k, 0] + self.global_workspace[k, 1]) / 2.
                delta = (self.global_workspace[k, 1] - self.global_workspace[k, 0])
                normalized_pose[k] = (normalized_pose[k] - mean) / delta + mean
            else:
                if self._useInverseKinematics or normalize_orn:
                    normalized_pose[k] /= (math.pi / 2)
        return normalized_pose

    def step(self, allocator_action):
        self.env_step_counter += 1

        # change action into multi discrete type
        allocator_action = self.extract_allocator_action(allocator_action)

        # apply_action
        cost, check = self.apply_action(allocator_action)
        self.accumulated_cost += cost
        # get new observation without cost_updated
        observation = self._observation()
        # get reward
        reward = self._reward(cost, check)
        reward *= self.reward_scale
        self.accumulated_reward += reward
        # check termination
        done = self._termination()
        if done:
            episode_info = {}
            episode_info['1_num_steps'] = self.env_step_counter
            episode_info['2_cost/accumulated_cost'] = self.accumulated_cost
            episode_info['2_cost/average_cost'] = self.accumulated_cost / self.env_step_counter
            episode_info['3_reward/accumulated_reward'] = self.accumulated_reward
            episode_info['3_reward/average_reward'] = self.accumulated_reward / self.env_step_counter
            episode_info['4_succ_task_num/task_num'] = self.succ_parts_num
            for i, robot in enumerate(self.robots):
                episode_info['5_robot_info/robot{}_task'.format(i + 1)] = robot.tasks_done
                episode_info['5_robot_info/robot{}_freeze_step'.format(i + 1)] = robot.freeze_step
                episode_info['5_robot_info/robot{}_wrong_allocation'.format(i + 1)] = robot.wrong_allocation
            episode_info['6_global_success'] = self.success

            # print(episode_info['4_succ_task_num/task_num'])
            # print(allocator_action)
            # print(self.accumulated_reward)
            # print("steps:  ",episode_info['1_num_steps'])
            # print("action:  ", allocator_action)
            # print("check:  ", check)
            # time.sleep(1)
        else:
            episode_info = {}
            episode_info['7_status/correct_check'] = check
            episode_info['state_info'] = self.state_info
        return observation, reward, done, episode_info

    def extract_allocator_action(self, allocator_action):
        if self.action_type == "MultiDiscrete":
            return allocator_action
        elif self.action_type == "Box":
            assert len(allocator_action) == self.action_dim * self.robots_num
            extracted_allocator_action = np.array([0] * self.robots_num)
            for i in range(self.robots_num):
                robot_action = allocator_action[i:i + self.action_dim]
                if sum(robot_action) > 0:
                    idx = random.choices(np.arange(self.action_dim), weights=robot_action, k=1)[0]
                    extracted_allocator_action[i] = idx

            return extracted_allocator_action
        elif self.action_type == "Discrete":

            extracted_allocator_action = np.array([0] * self.robots_num)
            extracted_allocator_action[0] = allocator_action // (self.parts_num + 1)
            extracted_allocator_action[1] = allocator_action % (self.parts_num + 1)

            return extracted_allocator_action
        else:
            return None

    def apply_action(self, allocator_action):
        cost = 0
        check = 0
        if self.use_prediction_model:
            assert self.prediction_updated
        if self._renders:
            print("allocator action:\t", allocator_action)
        coop_mask = self.state_info['coop_edge_mask'][allocator_action[0], allocator_action[1]]
        coop_cost = self.state_info['coop_edge_cost'][allocator_action[0], allocator_action[1]]

        if any(self.state_info['robots_task_edge_mask'][i][allocator_action[i]] == 0 for i in range(self.robots_num)):
            coop_mask = 0
        elif self.robot_done_freeze:
            for i, robot in enumerate(self.robots):
                if robot.is_done and allocator_action[i] != self.parts_num:
                    coop_mask = 0
        if coop_mask == 0:
            check = 0
            if self._renders:
                print("wrong action")
            # freeze action and get max cost
            for i, robot in enumerate(self.robots):
                robot.wrong_allocation += 1
            cost = self.max_cost_const
        else:
            if self._renders:
                print("correct action")
            # assign robot goal and execute action
            for i, robot in enumerate(self.robots):
                robot_action = allocator_action[i]

                if robot.is_done:
                    # done robot wont move
                    if self.robot_done_freeze:
                        assert robot_action == self.parts_num
                        continue
                    # done robot could move
                    else:
                        if robot_action < self.parts_num:
                            check += 1
                            # restart and finish a part task
                            part = self.parts[robot_action]
                            robot_target = part.getGoalPose()
                            robot.applyAction(robot_target)
                            part.pickedAndPlaced()
                            robot.tasks_done += 1
                            robot.is_done = False
                            robot.freeze_step = 0
                else:
                    # not done robots move
                    # print("act{}:".format(i),robot_action)
                    if robot_action < self.parts_num:
                        check += 1
                        # finish a part task
                        part = self.parts[robot_action]
                        robot_target = part.getGoalPose()
                        robot.applyAction(robot_target)
                        part.pickedAndPlaced()
                        robot.tasks_done += 1
                    else:
                        robot.applyAction(robot.goal_pose)
                        robot.is_done = True
                        robot.freeze_step = self.env_step_counter
            # get cost
            cost = coop_cost

        # update succ parts num
        self.succ_parts_num = sum(part.is_success for part in self.parts)

        return cost, check

    def _reward(self, cost, check):
        reward = - cost / self.max_cost_const
        reward += check * 1
        if all([robot.is_done for robot in self.robots]):
            if all([part.is_success for part in self.parts]):
                reward += self.global_success_bonus
            else:
                if self.robot_done_freeze:
                    # give punishment for task unfinished in success_freeze pattern
                    reward -= self.global_success_bonus
                else:
                    reward = - self.max_cost_const / self.max_cost_const
                # reward -= sum([not part.is_success for part in self.parts])
        return reward

    def _termination(self):
        if self.env_step_counter >= self._maxSteps:
            self.terminated = True
            self.failed = True

        if self.robot_done_freeze:
            # terminated when robot done
            if all([robot.is_done for robot in self.robots]):
                self.terminated = True
                if self.succ_parts_num == self.parts_num:
                    self.success = True
                else:
                    self.failed = True

        else:
            # terminate when task and robot done
            if all([robot.is_done for robot in self.robots]) and (self.succ_parts_num == self.parts_num):
                self.terminated = True
                self.success = True

        return self.terminated

    def sample_action(self):
        if self.action_type == "MultiDiscrete":
            action_list = np.arange(self.parts_num + 1)
            mask = self.state_info["robots_task_edge_mask"]
            action = []

            prob1 = np.array(mask[0]) / sum(mask[0])
            act1 = np.random.choice(action_list, p=prob1)
            action.append(act1)

            if act1 < self.parts_num:
                mask[1][act1] = 0
            prob2 = np.array(mask[1]) / sum(mask[1])
            act2 = np.random.choice(action_list, p=prob2)
            action.append(act2)

            return np.array(action)
        elif self.action_type == "Discrete":
            action_list = np.arange((self.parts_num + 1) ** self.robots_num)
            mask = self.state_info["coop_edge_mask"].flatten()
            action = []
            prob = mask / sum(mask)
            act = np.random.choice(action_list, p=prob)
            action.append(act)

            return np.array(action)

        else:
            return None
    ####################################### task allocators ################################
    # def distance_based_allocator(self, obs=None):
    #     ######## allocate closest tasks to robots
    #     selectable_part_idx = list(range(self.parts_num))
    #     for j in selectable_part_idx[::-1]:
    #         part = self.parts[j]
    #         if part.is_success or part.is_in_process:
    #             selectable_part_idx.remove(selectable_part_idx[j])
    #     # print(selectable_part_idx)
    #     if obs is None:
    #         return np.array([0] * self.parts_num * self.robots_num)
    #     dist_cost = obs
    #     dist_dict = {}
    #     for i, robot in enumerate(self.robots):
    #         if robot.is_in_process:
    #             # robot have target task and is in process, keep target
    #             idx = robot.task_idx_allocated
    #             dist_dict[i] = idx
    #     while len(dist_dict) < self.robots_num and len(selectable_part_idx) > 0:
    #         # print("selectable idx:\t", selectable_part_idx)
    #         # print("dist dict:\t", dist_dict)
    #         for i, robot in enumerate(self.robots):
    #             if i in dist_dict:
    #                 continue
    #             else:
    #                 idx_with_min_dist = selectable_part_idx[0]
    #                 min_dist = 10000
    #                 for idx in selectable_part_idx:
    #                     dist = dist_cost["dist_robot_{}_to_part_{}".format(i + 1, idx + 1)]
    #                     if dist <= min_dist:
    #                         min_dist = dist
    #                         idx_with_min_dist = idx
    #                 if idx_with_min_dist in dist_dict.values():
    #                     # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #                     other_robot_idx = [k for k, v in dist_dict.items() if v == idx_with_min_dist][0]
    #
    #                     robot_dist = dist_cost["dist_robot_{}_to_part_{}".format(i + 1, idx_with_min_dist + 1)]
    #                     other_robot_dist = dist_cost[
    #                         "dist_robot_{}_to_part_{}".format(other_robot_idx + 1, idx_with_min_dist + 1)]
    #                     if robot_dist < other_robot_dist:
    #                         # print("switch!")
    #                         dist_dict.pop(other_robot_idx)
    #                         dist_dict[i] = idx_with_min_dist
    #                     selectable_part_idx.remove(idx_with_min_dist)
    #                 else:
    #                     dist_dict[i] = idx_with_min_dist
    #
    #     allocator_action = []
    #     for i, robot in enumerate(self.robots):
    #         if i in dist_dict:
    #             idx = dist_dict[i]
    #         else:
    #             idx = -1
    #         allocator_action.append(idx + 1)
    #     # print("allocator action:\t", allocator_action)
    #     return allocator_action
    #
    # def random_allocator(self, obs=None):
    #     ######## random allocate tasks to robots
    #     selectable_part_idx = list(range(self.parts_num))
    #     for j in selectable_part_idx[::-1]:
    #         part = self.parts[j]
    #         if part.is_success or part.is_in_process:
    #             selectable_part_idx.remove(selectable_part_idx[j])
    #     # print(selectable_part_idx)
    #     allocator_action = []
    #     for i, robot in enumerate(self.robots):
    #         act = [0] * self.parts_num
    #         if robot.is_in_process:
    #             # robot have target task and is in process, keep target
    #             idx = robot.task_idx_allocated
    #         elif robot.task_idx_allocated in selectable_part_idx and random.random() < 0.75:
    #             # robot have target selectable task but not in process, keep target
    #             idx = robot.task_idx_allocated
    #         elif len(selectable_part_idx) > 0:
    #             # robot do not have selectable target task and not in process, random choose a selectable task as target
    #             idx = random.choice(selectable_part_idx)
    #             selectable_part_idx.remove(idx)
    #         else:
    #             idx = -1
    #         allocator_action.append(idx + 1)
    #     # print("allocator action:\t", allocator_action)
    #     return allocator_action
