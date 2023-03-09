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
from bullet_env.pseudo_plot import PseudoPlot
from bullet_env.pseudo_robot import PseudoRobot, PseudoPart

from scripts.load_pseudo_task_data import extract_pseudo_task_data

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
                 # parts_num=6,
                 maxSteps=50):

        self._urdfRoot = urdfRoot
        self._useInverseKinematics = useInverseKinematics

        self._renders = renders
        self._maxSteps = maxSteps
        self.parts_num = env_config['part_num']

        self.action_type = env_config['task_allocator_action_type']
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

        self.default_cost_const = 1.75
        self.max_cost_const = env_config['max_cost_const'] if env_config[
            'use_prediction_model'] else self.default_cost_const

        self.global_success_bonus = env_config['global_success_bonus']
        self.reward_scale = env_config['reward_scale']

        self.task_type = env_config['task_type']
        self.dynamic_task = env_config['dynamic_task']

        self.robot_done_freeze = env_config['robot_done_freeze']
        self.use_prediction_model = env_config['use_prediction_model']
        self.predict_content = env_config['predict_content']
        self.use_mask_constraint = env_config['use_mask_constraint']
        self.mask_constraint = env_config['mask_constraint']
        self.default_rest_pose = env_config['default_rest_pose']
        self.hard_mask = False
        self.mask_termination = False

        self.mask_done_task = env_config['mask_done_task']
        self.fix_box_sample = env_config['fix_box_sample']
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
        print("action_type:", self.action_type)
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

        self.reset_robots()
        self.reset_parts()

        if self._renders:
            self.plot = PseudoPlot(img_width=128, img_height=128, dpi=64)

            self.plot.setup(self.global_workspace, self.robots, self.parts)
            self.plot.imshow()

        self.reset()

    def update_workspace(self, robot=None):
        if robot is not None:
            for k in range(len(self.global_workspace)):
                self.global_workspace[k, 0] = min(self.global_workspace[k, 0], robot.workspace[k, 0])
                self.global_workspace[k, 1] = max(self.global_workspace[k, 1], robot.workspace[k, 1])
        return self.global_workspace

    def reset(self, load_task_data=None):
        self.terminated = False
        self.success = False
        self.failed = False
        self.succ_parts_num = 0
        self.env_step_counter = 0
        self.state_info = {}
        self.prev_state_info = {}

        self.accumulated_reward = 0
        self.accumulated_cost = 0
        self.accumulated_cost2 = 0
        # self.accumulated_reward_dict['robots'] = [0] * self.robots_num
        # self.accumulated_reward_dict['parts'] = [0] * self.parts_num
        # self.accumulated_reward_dict['task'] = 0
        self.prediction_updated = False

        ############################################################################

        if load_task_data is not None:
            load_task_data = extract_pseudo_task_data(load_task_data)

        self.reset_robots(load_task_data=load_task_data)
        self.reset_parts(load_task_data=load_task_data)

        ######## get obs and update state history
        self.observation_history = []
        self.state_info = {}
        obs = self._observation()
        self.prev_state_info = self.state_info.copy()

        if self._renders:
            # time.sleep(1)
            assert self.plot is not None
            self.plot.update_plot(self.robots, self.parts)
            self.plot.save()

        return obs

    def reset_robots(self, load_task_data=None):

        if load_task_data is not None:
            robots_data = load_task_data['robots']
        ######## robot reset#############
        for i, robot in enumerate(self.robots):
            robot.reset(default_goal_pose=self.default_rest_pose)
            if load_task_data is not None:
                r_data = robots_data[i]
                robot.load_robot_data(r_data)
                # print("robot load data:,", robot.robot_name, robot.getObservation_EE(), robot.goal_pose)
            robot.tasks_done = 0
            robot.wrong_allocation = 0
            robot.freeze_step = 0
            robot.is_done = False
            # print("robot reset:,", robot.robot_name, robot.getObservation_EE(), robot.goal_pose)

        return True

    def reset_parts(self, load_task_data=None):

        if load_task_data is not None:
            parts_data = load_task_data['parts']
            assert len(parts_data) == self.parts_num

        ######## parts reset
        # goal_poses_base = [[-0.7, -0.5], [-0.7, 0], [-0.7, 0.5], [0.7, -0.5], [0.7, 0], [0.7, 0.5]]
        # init_poses_base = [[-0.25, -0.5], [-0.25, 0], [-0.25, 0.5], [0.25, -0.5], [0.25, 0], [0.25, 0.5]]

        goal_poses_base = []
        init_poses_base = []
        x_axis = np.linspace(-0.7, 0.7, 15)
        y_axis = np.linspace(-0.5, 0.5, 11)
        for x in x_axis:
            for y in y_axis:
                if abs(x) <= 0.35:
                    init_poses_base.append([x, y])
                else:
                    goal_poses_base.append([x, y])

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
            goal_x = goal_pose_base[0] + (np.random.random() - 0.5) / 10.
            goal_y = goal_pose_base[1] + (np.random.random() - 0.5) / 10.
            goal_pos = [goal_x, goal_y, self.base_height]
            goal_orn = p.getQuaternionFromEuler([0, 0, 0])
            part.resetGoalPose(goal_pos, goal_orn)

            part.reset()
            if load_task_data is not None:
                p_data = parts_data[j]
                part.load_part_data(p_data)
                # print("part load data:", j, part.getInitPose(), part.getGoalPose())
            # print("part reset:", j, part.getInitPose(), part.getGoalPose())

        return True

    def _observation(self):
        state_dict = self.get_states()
        robot_nodes = state_dict["robots_nodes"]
        robots_task_edges = state_dict['task_edges']
        robots_task_edge_mask = state_dict['robots_task_edge_mask']
        coop_edges = state_dict["coop_edges"]
        coop_edge_mask = state_dict['coop_edge_mask']
        prediction_inputs = state_dict['prediction_inputs']

        observation = {}
        for i in range(self.robots_num):
            for j in range(self.parts_num):
                observation['robot_{}_part_node_{}'.format(i + 1, j + 1)] = robot_nodes[i][j].astype(np.float32)
            observation['robot_{}_reset_node'.format(i + 1)] = robot_nodes[i][self.parts_num].astype(np.float32)
            observation['robot_{}_task_edges'.format(i + 1)] = robots_task_edges[i].astype(np.float32)
            observation['robot_{}_task_edge_mask'.format(i + 1)] = robots_task_edge_mask[i].astype(np.float32)
        observation['coop_edges'] = coop_edges.astype(np.float32)
        observation['coop_edge_mask'] = coop_edge_mask.astype(np.float32)
        observation['prediction_inputs'] = prediction_inputs.astype(np.float32)

        return observation

    def get_states(self):
        self.state_info = {}
        # robot obs

        robots_ee = []
        robots_goal = []
        robots_base = []
        robots_mask = []
        for i, robot in enumerate(self.robots):
            robot_ee = robot.getObservation_EE()
            robot_goal = robot.goal_pose
            robot_base = robot.getBase()
            robot_done = robot.is_done

            self.state_info["robot_ee_{}".format(i + 1)] = robot_ee
            self.state_info["robot_goal_{}".format(i + 1)] = robot_goal
            self.state_info["robot_base_{}".format(i + 1)] = robot_base
            self.state_info["robot_done_{}".format(i + 1)] = robot_done

            # normalize
            norm_robot_ee = self.normalize_cartesian_pose(robot_ee)
            norm_robot_goal = self.normalize_cartesian_pose(robot_goal)
            norm_robot_base = self.normalize_cartesian_pose(robot_base)

            robots_ee.append(norm_robot_ee)
            robots_goal.append(norm_robot_goal)
            robots_base.append(norm_robot_base)
            robots_mask.append(1 - robot_done)

        # part obs
        parts_init = []
        parts_goal = []
        parts_mask = []
        for j, part in enumerate(self.parts):
            part_init = part.getInitPose()
            part_goal = part.getGoalPose()
            part_done = part.is_success

            # part_init[2] += (np.random.random() - 0.5) / 50
            # part_goal[2] += (np.random.random() - 0.5) / 50

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
                if self.hard_mask:
                    node_mask = robots_mask[i] * parts_mask[j]
                else:
                    node_mask = parts_mask[j]

                self.state_info["robot_{}_node_obs1_{}".format(i + 1, j + 1)] = obs1
                self.state_info["robot_{}_node_obs2_{}".format(i + 1, j + 1)] = obs2
                self.state_info["robot_{}_node_mask_{}".format(i + 1, j + 1)] = node_mask

                # static node state
                static_init = parts_init[j] - robots_base[i]
                static_goal = parts_goal[j] - robots_base[i]
                # dynamic node state
                dynamic_init = parts_init[j] - robots_ee[i]
                dynamic_goal = parts_goal[j] - robots_ee[i]
                # dist1, dist2
                dist_pick = np.linalg.norm(init_ee1[:3] - goal_ee1[:3])
                dist_place = np.linalg.norm(init_ee2[:3] - goal_ee2[:3])
                dist_s1 = np.linalg.norm(parts_init[j][:3] - robots_base[i][:3])
                dist_s2 = np.linalg.norm(parts_goal[j][:3] - robots_base[i][:3])
                dist_d1 = np.linalg.norm(parts_init[j][:3] - robots_ee[i][:3])
                dist_d2 = np.linalg.norm(parts_goal[j][:3] - robots_ee[i][:3])
                ############################  experiment: node state obs
                dists = np.array([dist_s1, dist_s2, dist_d1, dist_d2, dist_pick, dist_place])
                # dists = np.array([dist_d1, dist_d2, dist_pick, dist_place])
                # dists = np.array([dist_s1, dist_s2, dist_pick, dist_place])
                ############################  experiment: node state obs

                self.state_info["robot_node_static_{}_{}".format(i + 1, j + 1)] = [static_init, static_goal]
                self.state_info["robot_node_dynamic_{}_{}".format(i + 1, j + 1)] = [dynamic_init, dynamic_goal]
                self.state_info["robot_node_dist_{}_{}".format(i + 1, j + 1)] = dists

                ############################  experiment: node state obs
                node_obs = np.concatenate([static_init, static_goal, dynamic_init, dynamic_goal, dists])
                # node_obs = np.concatenate([dynamic_init, dynamic_goal, dists])
                # node_obs = np.concatenate([static_init, static_goal, dists])
                ############################  experiment: node state obs
                self.state_info["robot_{}_node_obs_{}".format(i + 1, j + 1)] = node_obs

                if self.mask_done_task:
                    node_obs *= parts_mask[j]

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
            self.state_info["robot_{}_node_obs1_{}".format(i + 1, self.parts_num + 1)] = obs1
            self.state_info["robot_{}_node_obs2_{}".format(i + 1, self.parts_num + 1)] = obs2

            # reset node state
            # poses
            base = robots_base[i]
            ee = robots_ee[i]
            rest = robots_goal[i]
            # dists
            dist1 = np.linalg.norm(base[:3] - ee[:3])
            dist2 = np.linalg.norm(ee[:3] - rest[:3])
            dist3 = np.linalg.norm(rest[:3] - base[:3])
            ############################  experiment: rest node
            dists2 = np.array([dist1, dist2, dist3])
            # dists2 = np.array([dist2])
            ############################  experiment: rest node
            self.state_info["robot_reset_node_pose_{}".format(i + 1)] = [base, ee, rest]
            self.state_info["robot_reset_node_dist_{}".format(i + 1)] = dists2
            ############################  experiment: rest node
            reset_node_obs = np.concatenate([base, ee, rest, dists2])
            # reset_node_obs = np.concatenate([ee, rest, dists2])
            ############################  experiment: rest node
            self.state_info["robot_{}_reset_node_obs".format(i + 1)] = reset_node_obs

            robot_nodes.append(reset_node_obs)
            robots_nodes.append(robot_nodes)

        # task_edge_obs
        robots_task_edges = []
        robots_node_masks = []

        robots_task_edge_xyz = []
        robots_task_edge_rz = []
        robots_task_edge_mask = []
        for i in range(self.robots_num):
            task_edge_xyz = np.zeros((self.parts_num + 1, self.parts_num + 1))
            task_edge_rz = np.zeros((self.parts_num + 1, self.parts_num + 1))
            task_edge_mask = np.ones((self.parts_num + 1, self.parts_num + 1))
            node_mask = np.array([1] * (self.parts_num + 1))

            for m in range(self.parts_num):
                # edge: to parts
                for n in range(self.parts_num):
                    task_edge_mask[m, n] = parts_mask[m] * parts_mask[n]
                    if n == m:
                        pass
                    else:
                        init_ee1 = parts_goal[m]
                        goal_ee1 = parts_init[n]
                        task_edge_xyz[m, n] = np.linalg.norm(init_ee1[:3] - goal_ee1[:3])
                        task_edge_rz[m, n] = np.linalg.norm(init_ee1[3:] - goal_ee1[3:])
                        if self.mask_done_task:
                            task_edge_xyz[m, n] *= parts_mask[m]
                            task_edge_xyz[m, n] *= parts_mask[n]

                # edge: to reset
                init_ee1 = parts_goal[m]
                goal_ee1 = robots_goal[i]
                task_edge_xyz[m, self.parts_num] = np.linalg.norm(init_ee1[:3] - goal_ee1[:3])
                task_edge_rz[m, self.parts_num] = np.linalg.norm(init_ee1[3:] - goal_ee1[3:])
                task_edge_mask[m, self.parts_num] = parts_mask[m]

            for n in range(self.parts_num):
                # edge: to reset
                init_ee1 = robots_goal[i]
                goal_ee1 = parts_init[n]
                task_edge_xyz[self.parts_num, n] = np.linalg.norm(init_ee1[:3] - goal_ee1[:3])
                task_edge_rz[self.parts_num, n] = np.linalg.norm(init_ee1[3:] - goal_ee1[3:])
                task_edge_mask[self.parts_num, n] = parts_mask[n]
                # node mask
                node_mask[n] = robots_mask[i] * parts_mask[n]
                # node_mask[n] = parts_mask[n]

            self.state_info["robot_{}_task_edge_xyz".format(i + 1)] = task_edge_xyz
            self.state_info["robot_{}_task_edge_rz".format(i + 1)] = task_edge_rz
            self.state_info["robot_{}_task_edge_mask".format(i + 1)] = task_edge_mask
            self.state_info["robot_{}_node_mask".format(i + 1)] = node_mask

            robots_task_edge_xyz.append(task_edge_xyz)
            robots_task_edge_rz.append(task_edge_rz)
            robots_task_edge_mask.append(task_edge_mask)
            robots_node_masks.append(node_mask)
            ############################  experiment: task edge
            task_edges = np.stack([task_edge_xyz, task_edge_rz, task_edge_mask], axis=-1)
            # task_edges = np.stack([task_edge_xyz, task_edge_rz], axis=-1)
            ############################  experiment: task edge
            self.state_info["robot_{}_task_edges".format(i + 1)] = task_edges
            robots_task_edges.append(task_edges)

        # cost_edge_initialization
        coop_edge_mask = np.ones((self.parts_num + 1, self.parts_num + 1))
        coop_edge_cost1 = np.ones((self.parts_num + 1, self.parts_num + 1))
        coop_edge_cost2 = np.ones((self.parts_num + 1, self.parts_num + 1))
        coop_edge_cost_all = (coop_edge_cost1 + coop_edge_cost2) / 2.
        coop_edges = []
        for m in range(self.parts_num + 1):
            for n in range(self.parts_num + 1):
                ##### calculate initial coop mask
                coop_edge_mask[m, n] = robots_node_masks[0][m] * robots_node_masks[1][n]
                if m == n and m < self.parts_num:
                    coop_edge_mask[m, m] = 0
                if self.robot_done_freeze:
                    if not robots_mask[0] and m < self.parts_num:
                        coop_edge_mask[m, n] = 0
                    if not robots_mask[1] and n < self.parts_num:
                        coop_edge_mask[m, n] = 0
                if (m == self.parts_num or n == self.parts_num):
                    coop_edge_mask[m, n] = 0

                ##### add additional mask constraint
                if self.use_mask_constraint and self.add_constraint_mask(m, n, -1, -1, robots_ee, robots_goal,
                                                                      parts_init, parts_goal):
                    coop_edge_mask[m, n] = 0

                ##### calculate initial coop cost value
                # if not self.use_prediction_model and coop_edge_mask[m, n]:
                if coop_edge_mask[m, n]:
                    d1_pick = robots_nodes[0][m][-2]
                    d1_place = robots_nodes[0][m][-1] if m < self.parts_num else 0
                    d2_pick = robots_nodes[1][n][-2]
                    d2_place = robots_nodes[1][n][-1] if n < self.parts_num else 0

                    dist1 = max(d1_pick, d2_pick)
                    dist2 = max(d1_place, d2_place)

                    ### normalize cost
                    c1 = dist1 / self.default_cost_const
                    c2 = dist2 / self.default_cost_const
                    c_all = (c1 + c2) / 2

                    # update edge cost
                    coop_edge_cost1[m, n] = c1
                    coop_edge_cost2[m, n] = c2
                    coop_edge_cost_all[m, n] = c_all
        if coop_edge_mask.sum(axis=-2).sum(axis=-1) < 1:
            coop_edge_mask[-1, -1] = 1
            dist1 = max(robots_nodes[0][-1][-2], robots_nodes[1][-1][-2])
            dist2 = 0

            ### normalize cost
            c1 = dist1 / self.default_cost_const
            c2 = dist2 / self.default_cost_const
            c_all = (c1 + c2) / 2
            # update edge cost

            coop_edge_cost1[-1, -1] = c1
            coop_edge_cost2[-1, -1] = c2
            coop_edge_cost_all[-1, -1] = c_all

        if self.mask_termination:
            for m in range(self.parts_num + 1):
                for n in range(self.parts_num + 1):
                    if m == self.parts_num or n == self.parts_num:
                        coop_edge_mask[m, n] = 0
            if coop_edge_mask[:-1, :-1].sum() < 1:
                coop_edge_mask[-1, -1] = 1

        if self.mask_done_task:
            for m in range(self.parts_num + 1):
                for n in range(self.parts_num + 1):
                    coop_edge_cost_all[m, n] *= coop_edge_mask[m, n]
                    coop_edge_cost1[m, n] *= coop_edge_mask[m, n]
                    coop_edge_cost2[m, n] *= coop_edge_mask[m, n]

        self.state_info["robots_nodes"] = robots_nodes
        self.state_info['task_edges'] = robots_task_edges
        # self.state_info['robots_task_edge_rz'] = robots_task_edge_rz
        self.state_info['robots_task_edge_mask'] = robots_task_edge_mask
        self.state_info['robots_node_masks'] = robots_node_masks
        self.state_info["coop_edge_cost1"] = coop_edge_cost1
        self.state_info["coop_edge_cost2"] = coop_edge_cost2
        self.state_info["coop_edge_cost_all"] = coop_edge_cost_all
        self.state_info['coop_edge_mask'] = coop_edge_mask

        ############################  experiment: coop edge
        coop_edges = np.stack(
            [coop_edge_cost_all, coop_edge_cost1, coop_edge_cost2, coop_edge_mask, coop_edge_mask, coop_edge_mask],
            axis=-1)
        # coop_edges = np.stack([coop_edge_cost_all, coop_edge_cost1, coop_edge_cost2],axis=-1)
        # coop_edges = np.stack([coop_edge_cost_all,coop_edge_mask],axis=-1)
        # coop_edges = np.stack([coop_edge_cost_all],axis=-1)

        ############################  experiment: coop edge
        self.state_info["coop_edges"] = coop_edges

        #### prediction model input
        self.state_info["prediction_inputs"] = self.get_prediction_obs()
        self.state_info["prediction_inputs_shape"] = self.state_info["prediction_inputs"].shape
        if self.use_prediction_model:
            self.prediction_updated = False
        else:
            self.custom_coop_edge()

        return self.state_info

    def get_current_task_data(self):
        data = []
        robots_data = []
        parts_data = []
        for i in range(self.robots_num):
            robot_ee = self.state_info["robot_ee_{}".format(i + 1)]
            robot_goal = self.state_info["robot_goal_{}".format(i + 1)]
            robot_base = self.state_info["robot_base_{}".format(i + 1)]
            robot_done = self.state_info["robot_done_{}".format(i + 1)]

            ####
            robot_data = np.concatenate([robot_ee, robot_goal])
            robot_data_dim = 8
            ####
            robots_data.append(robot_data)
        for j in range(self.parts_num):
            part_init = self.state_info["part_init_{}".format(j + 1)]
            part_goal = self.state_info["part_goal_{}".format(j + 1)]
            part_done = self.state_info["part_done_{}".format(j + 1)]

            ####
            part_data = np.concatenate([part_init, part_goal])
            part_dim = 8
            ####
            parts_data.append(part_data)

        robots_data = np.concatenate(robots_data)
        parts_data = np.concatenate(parts_data)
        data = np.concatenate([robots_data, parts_data])
        # print(data)
        return data

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
        # print(prediction_inputs.shape)
        return prediction_inputs.astype(np.float32)

    def update_prediction(self, data):
        if self._renders:
            print("prediction updated")

        cost = data[0]
        mask = data[1]
        # print(cost)

        self.state_info["data"] = data
        self.state_info["coop_edge_mask"] = mask
        self.state_info["coop_edges"] = cost
        self.prediction_updated = True

        ###### custom coop edge
        self.custom_coop_edge()

        state_dict = self.state_info
        robot_nodes = state_dict["robots_nodes"]
        robots_task_edges = state_dict['task_edges']
        robots_task_edge_mask = state_dict['robots_task_edge_mask']
        coop_edges = state_dict["coop_edges"]
        coop_edge_mask = state_dict['coop_edge_mask']
        prediction_inputs = state_dict['prediction_inputs']
        # print("check\n",coop_edge_mask)
        observation = {}
        for i in range(self.robots_num):
            for j in range(self.parts_num):
                observation['robot_{}_part_node_{}'.format(i + 1, j + 1)] = robot_nodes[i][j].astype(np.float32)
            observation['robot_{}_reset_node'.format(i + 1)] = robot_nodes[i][self.parts_num].astype(np.float32)
            observation['robot_{}_task_edges'.format(i + 1)] = robots_task_edges[i].astype(np.float32)
            observation['robot_{}_task_edge_mask'.format(i + 1)] = robots_task_edge_mask[i].astype(np.float32)
        observation['coop_edges'] = coop_edges.astype(np.float32)
        observation['coop_edge_mask'] = coop_edge_mask.astype(np.float32)
        observation['prediction_inputs'] = prediction_inputs.astype(np.float32)

        return observation

    def custom_coop_edge(self):
        # cost_map = self.state_info["coop_edge_cost"][:self.parts_num, :self.parts_num].flatten()
        # mask_map = self.state_info["coop_edge_mask"][:self.parts_num, :self.parts_num].flatten()
        # # print(cost_map)
        # # print(mask_map)
        #
        # # choice_num = 5
        # # idx = np.argpartition(cost_map, choice_num)[:choice_num]
        # # # print(idx, c)
        # # for k in range(len(mask_map)):
        # #     if k in idx:
        # #         pass
        # #     else:
        # #         mask_map[k] = mask_map[k] * 0.99
        # #
        # # mask_map = mask_map.reshape(self.parts_num, self.parts_num)
        # #
        # # # print(mask_map)
        # # self.state_info["coop_edge_mask"][:self.parts_num, :self.parts_num] = mask_map[:, :]
        #
        # if self.mask_done_task:
        #     for m in range(self.parts_num + 1):
        #         for n in range(self.parts_num + 1):
        #             self.state_info["coop_edge_cost"][m, n] *= self.state_info["coop_edge_mask"][m, n]

        return True

    def normalize_cartesian_pose(self, cartesian_pose, normalize_orn=False):
        if len(cartesian_pose) == 0:
            return None
        normalized_pose = cartesian_pose.copy()
        for k in range(len(normalized_pose)):
            if k < 3:
                if normalized_pose[k] < self.global_workspace[k, 0]:
                    normalized_pose[k] = self.global_workspace[k, 0]
                elif normalized_pose[k] > self.global_workspace[k, 1]:
                    normalized_pose[k] = self.global_workspace[k, 1]

                mean = (self.global_workspace[k, 0] + self.global_workspace[k, 1]) / 2.
                delta = (self.global_workspace[k, 1] - self.global_workspace[k, 0]) / 2.
                normalized_pose[k] = (normalized_pose[k] - mean) / delta + mean
            else:
                if self._useInverseKinematics or normalize_orn:
                    if normalized_pose[k] > (math.pi / 2):
                        normalized_pose[k] = (math.pi / 2)
                    elif normalized_pose[k] < -(math.pi / 2):
                        normalized_pose[k] = -(math.pi / 2)
                    normalized_pose[k] /= (math.pi / 2)
        return normalized_pose

    def step(self, allocator_action):
        # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa", allocator_action)
        self.env_step_counter += 1

        # change to termination actin if no available action
        if self.state_info["coop_edge_mask"].sum() < 1:
            allocator_action = (self.parts_num + 1) ** self.robots_num - 1
        else:
            # change action into multi discrete type
            allocator_action = self.extract_allocator_action(allocator_action)

        # apply_action
        cost, check = self.apply_action(allocator_action)
        # print("action:\t", allocator_action)
        # print("cost:\t",cost)
        # get new observation without cost_updated
        observation = self._observation()
        # get reward
        reward = self._reward(cost, check)
        reward *= self.reward_scale
        self.accumulated_reward += reward
        # check termination
        done = self._termination()

        ## update plot
        if self._renders:
            # time.sleep(1)
            assert self.plot is not None
            self.plot.update_plot(self.robots, self.parts)
            self.plot.save()

        if done:
            episode_info = {}
            episode_info['1_num_steps'] = self.env_step_counter
            # print(self.env_step_counter)
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
            episode_info['allocator_action'] = allocator_action
            episode_info['state_info'] = self.state_info
        return observation, reward, done, episode_info

    def extract_allocator_action(self, allocator_action):
        if self.action_type == "MultiDiscrete":
            return allocator_action
        elif self.action_type == "Box":
            if len(allocator_action) == 1:
                allocator_action = allocator_action[0]
            # print("extract", allocator_action.shape, allocator_action)
            # print(allocator_action)
            assert len(allocator_action) == self.action_dim * self.robots_num

            robots_probs = []
            for i in range(self.robots_num):
                robot_action = allocator_action[i:i + self.action_dim]
                # if sum(robot_action) > 0:
                #     idx = random.choices(np.arange(self.action_dim), weights=robot_action, k=1)[0]
                #     extracted_allocator_action[i] = idx
                if sum(robot_action) <= 0:
                    robot_action[-1] = 1
                action_prob = np.array(robot_action / sum(robot_action))
                robots_probs.append(action_prob)

            prob1 = robots_probs[0].reshape(self.action_dim, 1)
            prob2 = robots_probs[1].reshape(1, self.action_dim)

            joint_probs = prob1 * prob2

            action_mask = self.state_info['coop_edge_mask']
            assert joint_probs.shape == action_mask.shape
            joint_probs = joint_probs * action_mask

            # print(joint_probs)
            joint_probs = joint_probs.reshape(self.action_dim ** 2)

            # print("!!!!!!!!!!!!!!!!!!!!!!1", joint_probs.sum())
            if joint_probs.sum() <= 0:
                joint_action = np.arange(len(joint_probs))[-1]
            else:
                joint_probs /= joint_probs.sum()

                if self.fix_box_sample:
                    joint_action = np.argmax(joint_probs)
                else:
                    joint_action = random.choices(np.arange(len(joint_probs)), weights=joint_probs, k=1)[0]

            # print("joint_action:", joint_action)
            extracted_allocator_action = np.array([0] * self.robots_num)
            extracted_allocator_action[0] = joint_action // (self.parts_num + 1)
            extracted_allocator_action[1] = joint_action % (self.parts_num + 1)

            # print("111",extracted_allocator_action)

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
        ############################  experiment: coop edge
        coop_cost = self.state_info['coop_edges'][allocator_action[0], allocator_action[1]][0]
        # print("coop cost value:", self.state_info['coop_edges'][allocator_action[0], allocator_action[1]][0:3])
        ############################  experiment: coop edge

        if allocator_action[0] == self.parts_num and allocator_action[1] == self.parts_num:
            coop_mask = 1
        elif any(self.state_info['robots_node_masks'][i][allocator_action[i]] == 0 for i in range(self.robots_num)):
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
            cost = self.default_cost_const
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
        # print("check action:", allocator_action)
        # print("ccheck ost:", cost)
        return cost, check

    def _reward(self, cost, check):
        # print("cost",cost)
        self.accumulated_cost += cost
        self.accumulated_cost2 += (1 - cost)

        if check == 0:
            # reward = -1
            reward = 1 - cost
        else:
            reward = 1 - cost
            # reward = - cost
            # reward = 0

        if all([robot.is_done for robot in self.robots]):
            # reward -= self.accumulated_reward
            reward += self.accumulated_cost2
            if all([part.is_success for part in self.parts]):
                # reward += self.global_success_bonus
                # reward += self.accumulated_reward
                pass
            else:
                pass
                reward = - self.global_success_bonus / 5
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
            # # terminate when task and robot done
            # if all([robot.is_done for robot in self.robots]) and (self.succ_parts_num == self.parts_num):
            #     self.terminated = True
            #     self.success = True
            # terminated when robot done
            if all([robot.is_done for robot in self.robots]):
                self.terminated = True
                if self.succ_parts_num == self.parts_num:
                    self.success = True
                else:
                    self.failed = True

        return self.terminated

    def sample_action(self, use_baseline="min_cost_sample"):
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!")
        if self.action_type == "MultiDiscrete":
            action_list = np.arange(self.parts_num + 1)
            mask = self.state_info["robots_node_masks"]
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
            mask = self.state_info["coop_edge_mask"].copy()
            action = []

            if use_baseline == "random_sample":
                mask[self.parts_num, self.parts_num] = 0
                mask0 = np.zeros((self.parts_num + 1, self.parts_num + 1))
                for m in range(self.parts_num):
                    for n in range(self.parts_num):
                        mask0[m, n] = 1
                mask0 = (mask * mask0)
                mask0 = mask0.flatten()
                mask = mask.flatten()

                if sum(mask0) >= 1:
                    prob = mask0 / sum(mask0)
                    act = np.random.choice(action_list, p=prob)
                # elif sum(mask) >= 1:
                #     prob = mask / sum(mask)
                #     act = np.random.choice(action_list, p=prob)
                else:
                    act = (self.parts_num + 1) ** self.robots_num - 1
                action.append(act)
                return np.array(action)

            elif use_baseline == "min_cost_sample":
                mask[self.parts_num, self.parts_num] = 0
                mask0 = np.zeros((self.parts_num + 1, self.parts_num + 1))
                for m in range(self.parts_num):
                    for n in range(self.parts_num):
                        mask0[m, n] = 1
                mask0 = (mask * mask0)
                # print(mask0)
                mask0 = mask0.flatten()
                mask = mask.flatten()
                ############################  experiment: coop edge
                cost = self.state_info["coop_edges"][:, :, 0].copy()
                # cost = self.state_info["coop_edges"][:,:,0].copy()
                ############################  experiment: coop edge
                # print(cost)
                cost = cost.flatten()
                if sum(mask0) >= 1:
                    # print("in A")
                    cost = cost * mask0 + self.default_cost_const * (1 - mask0) * 10
                    act = np.argmin(cost)
                    assert mask0[act]
                # elif sum(mask) >= 1:
                #     # print("in B")
                #     cost = cost * mask + self.default_cost_const * (1 - mask) * 10
                #     act = np.argmin(cost)
                #     assert mask[act]
                else:
                    # print("in C")
                    act = (self.parts_num + 1) ** self.robots_num - 1
                action.append(act)
                # print(act)
                return np.array(action)

            else:
                mask = mask.flatten()
                prob = mask / sum(mask)
                act = np.random.choice(action_list, p=prob)
                action.append(act)
                return np.array(action)
        elif self.action_type == "Box":
            return self.action_space.sample()
        else:
            return None

    def enviroment_change(self):
        # randomly change done/undone for a part or switch two parts curr_pose/goal_pose with a probability

        return True

    def get_data_for_offline_planning(self):

        ### get robot data
        robots_ee = []
        robots_goal = []
        robots_mask = []
        for i, robot in enumerate(self.robots):
            robot_ee = robot.getObservation_EE()
            robot_goal = robot.goal_pose
            robot_done = robot.is_done

            # normalize
            norm_robot_ee = self.normalize_cartesian_pose(robot_ee)
            norm_robot_goal = self.normalize_cartesian_pose(robot_goal)

            robots_ee.append(norm_robot_ee)
            robots_goal.append(norm_robot_goal)
            robots_mask.append(1 - robot_done)

        ### get part data
        parts_init = []
        parts_goal = []
        parts_mask = []
        for j, part in enumerate(self.parts):
            part_init = part.getInitPose()
            part_goal = part.getGoalPose()
            part_done = part.is_success

            # part_init[2] += (np.random.random() - 0.5) / 50
            # part_goal[2] += (np.random.random() - 0.5) / 50

            # normalize
            norm_part_init = self.normalize_cartesian_pose(part_init)
            norm_part_goal = self.normalize_cartesian_pose(part_goal)

            parts_init.append(norm_part_init)
            parts_goal.append(norm_part_goal)
            parts_mask.append(1 - part_done)

        cost_data = {}
        mask_data = {}

        cost_node = np.ones((self.parts_num, self.parts_num))
        mask_node = np.zeros((self.parts_num, self.parts_num))
        cost_node2node = np.ones(
            (self.parts_num + 1, self.parts_num + 1, self.parts_num + 1, self.parts_num + 1))
        mask_node2node = np.zeros((self.parts_num + 1, self.parts_num + 1, self.parts_num + 1, self.parts_num + 1))

        fearure_dim = 20
        fearure_n2n = np.zeros(
            (self.parts_num + 1, self.parts_num + 1, self.parts_num + 1, self.parts_num + 1, fearure_dim))
        fearure_n = np.zeros(
            (self.parts_num, self.parts_num, fearure_dim))

        mask_n = np.zeros((self.parts_num, self.parts_num))
        mask_n2n = np.zeros((self.parts_num + 1, self.parts_num + 1, self.parts_num + 1, self.parts_num + 1))

        # for start 2 node
        for j1 in range(self.parts_num):
            for j2 in range(self.parts_num):
                init_ee1 = robots_ee[0]
                goal_ee1 = parts_init[j1]
                dist_xyz1 = np.linalg.norm(init_ee1[:3] - goal_ee1[:3])
                dist_rz1 = np.linalg.norm(init_ee1[3:] - goal_ee1[3:])
                obs1 = np.concatenate([init_ee1, goal_ee1, [dist_xyz1], [dist_rz1]])

                init_ee2 = robots_ee[1]
                goal_ee2 = parts_init[j2]
                dist_xyz2 = np.linalg.norm(init_ee2[:3] - goal_ee2[:3])
                dist_rz2 = np.linalg.norm(init_ee2[3:] - goal_ee2[3:])
                obs2 = np.concatenate([init_ee2, goal_ee2, [dist_xyz2], [dist_rz2]])

                fearure_n2n[self.parts_num, j1, self.parts_num, j2, :] = np.concatenate([obs1, obs2])[:]
                #### cost design###
                if j1 == j2:
                    mask_node2node[self.parts_num, j1, self.parts_num, j2] = 0
                    cost_node2node[self.parts_num, j1, self.parts_num, j2] = 1
                #### add mask constraint
                elif self.use_mask_constraint and self.add_constraint_mask(j1, j2, -1, -1, robots_ee, robots_goal,
                                                                           parts_init, parts_goal):
                    mask_node2node[self.parts_num, j1, self.parts_num, j2] = 0
                    cost_node2node[self.parts_num, j1, self.parts_num, j2] = 1
                else:
                    c = max(dist_xyz1, dist_xyz2) / (self.default_cost_const)
                    mask_node2node[self.parts_num, j1, self.parts_num, j2] = 1
                    cost_node2node[self.parts_num, j1, self.parts_num, j2] = c
        # for node transfer
        for j1 in range(self.parts_num):
            for j2 in range(self.parts_num):
                init_ee1 = parts_init[j1]
                goal_ee1 = parts_goal[j1]
                dist_xyz1 = np.linalg.norm(init_ee1[:3] - goal_ee1[:3])
                dist_rz1 = np.linalg.norm(init_ee1[3:] - goal_ee1[3:])
                obs1 = np.concatenate([init_ee1, goal_ee1, [dist_xyz1], [dist_rz1]])

                init_ee2 = parts_init[j2]
                goal_ee2 = parts_goal[j2]
                dist_xyz2 = np.linalg.norm(init_ee2[:3] - goal_ee2[:3])
                dist_rz2 = np.linalg.norm(init_ee2[3:] - goal_ee2[3:])
                obs2 = np.concatenate([init_ee2, goal_ee2, [dist_xyz2], [dist_rz2]])

                fearure_n[j1, j2, :] = np.concatenate([obs1, obs2])[:]
                #### cost design###
                if j1 == j2:
                    mask_node[j1, j2] = 0
                    cost_node[j1, j2] = 1
                #### add mask constraint
                elif self.use_mask_constraint and self.add_constraint_mask(j1, j2, -1, -1, robots_ee, robots_goal,
                                                                           parts_init, parts_goal):
                    mask_node[j1, j2] = 0
                    cost_node[j1, j2] = 1
                else:
                    c = max(dist_xyz1, dist_xyz2) / (self.default_cost_const)
                    mask_node[j1, j2] = 1
                    cost_node[j1, j2] = c
        # for node 2 node transit
        for j1_init in range(self.parts_num):
            for j1_goal in range(self.parts_num):
                for j2_init in range(self.parts_num):
                    for j2_goal in range(self.parts_num):
                        init_ee1 = parts_goal[j1_init]
                        goal_ee1 = parts_init[j1_goal]
                        dist_xyz1 = np.linalg.norm(init_ee1[:3] - goal_ee1[:3])
                        dist_rz1 = np.linalg.norm(init_ee1[3:] - goal_ee1[3:])
                        obs1 = np.concatenate([init_ee1, goal_ee1, [dist_xyz1], [dist_rz1]])

                        init_ee2 = parts_goal[j2_init]
                        goal_ee2 = parts_init[j2_goal]
                        dist_xyz2 = np.linalg.norm(init_ee2[:3] - goal_ee2[:3])
                        dist_rz2 = np.linalg.norm(init_ee2[3:] - goal_ee2[3:])
                        obs2 = np.concatenate([init_ee2, goal_ee2, [dist_xyz2], [dist_rz2]])

                        fearure_n2n[j1_init, j1_goal, j2_init, j2_goal, :] = np.concatenate([obs1, obs2])[:]
                        #### cost design###
                        if j1_init == j1_goal:
                            mask_node2node[j1_init, j1_goal, j2_init, j2_goal] = 0
                            cost_node2node[j1_init, j1_goal, j2_init, j2_goal] = 1
                        elif j2_init == j2_goal:
                            mask_node2node[j1_init, j1_goal, j2_init, j2_goal] = 0
                            cost_node2node[j1_init, j1_goal, j2_init, j2_goal] = 1
                        elif j1_init == j2_init:
                            mask_node2node[j1_init, j1_goal, j2_init, j2_goal] = 0
                            cost_node2node[j1_init, j1_goal, j2_init, j2_goal] = 1
                        elif j1_goal == j2_goal:
                            mask_node2node[j1_init, j1_goal, j2_init, j2_goal] = 0
                            cost_node2node[j1_init, j1_goal, j2_init, j2_goal] = 1
                        #### add mask constraint
                        elif self.use_mask_constraint and self.add_constraint_mask(j1_goal, j2_goal, j1_init, j2_init,
                                                                                   robots_ee, robots_goal, parts_init,
                                                                                   parts_goal):
                            mask_node2node[j1_init, j1_goal, j2_init, j2_goal] = 0
                            cost_node2node[j1_init, j1_goal, j2_init, j2_goal] = 1
                        else:
                            c = max(dist_xyz1, dist_xyz2) / (self.default_cost_const)
                            mask_node2node[j1_init, j1_goal, j2_init, j2_goal] = 1
                            cost_node2node[j1_init, j1_goal, j2_init, j2_goal] = c
        # for node 2 end
        for j1 in range(self.parts_num):
            for j2 in range(self.parts_num):
                init_ee1 = parts_goal[j1]
                goal_ee1 = robots_goal[0]
                dist_xyz1 = np.linalg.norm(init_ee1[:3] - goal_ee1[:3])
                dist_rz1 = np.linalg.norm(init_ee1[3:] - goal_ee1[3:])
                obs1 = np.concatenate([init_ee1, goal_ee1, [dist_xyz1], [dist_rz1]])

                init_ee2 = parts_goal[j2]
                goal_ee2 = robots_goal[1]
                dist_xyz2 = np.linalg.norm(init_ee2[:3] - goal_ee2[:3])
                dist_rz2 = np.linalg.norm(init_ee2[3:] - goal_ee2[3:])
                obs2 = np.concatenate([init_ee2, goal_ee2, [dist_xyz2], [dist_rz2]])

                fearure_n2n[j1, self.parts_num, j2, self.parts_num, :] = np.concatenate([obs1, obs2])[:]
                #### cost design###
                if j1 == j2:
                    mask_node2node[j1, self.parts_num, j2, self.parts_num] = 0
                    cost_node2node[j1, self.parts_num, j2, self.parts_num] = 1
                else:
                    c = max(dist_xyz1, dist_xyz2) / (self.default_cost_const)
                    mask_node2node[j1, self.parts_num, j2, self.parts_num] = 1
                    cost_node2node[j1, self.parts_num, j2, self.parts_num] = c

        init_ee1 = robots_ee[0]
        goal_ee1 = robots_goal[0]
        dist_xyz1 = np.linalg.norm(init_ee1[:3] - goal_ee1[:3])
        dist_rz1 = np.linalg.norm(init_ee1[3:] - goal_ee1[3:])
        obs1 = np.concatenate([init_ee1, goal_ee1, [dist_xyz1], [dist_rz1]])

        init_ee2 = robots_ee[1]
        goal_ee2 = robots_goal[1]
        dist_xyz2 = np.linalg.norm(init_ee2[:3] - goal_ee2[:3])
        dist_rz2 = np.linalg.norm(init_ee2[3:] - goal_ee2[3:])
        obs2 = np.concatenate([init_ee2, goal_ee2, [dist_xyz2], [dist_rz2]])

        fearure_n2n[self.parts_num, self.parts_num, self.parts_num, self.parts_num, :] = np.concatenate(
            [obs1, obs2])[:]
        mask_n2n[self.parts_num, self.parts_num, self.parts_num, self.parts_num] = 0
        cost_node2node[self.parts_num, self.parts_num, self.parts_num, self.parts_num] = 1

        feature = {}
        feature['n'] = fearure_n.astype(np.float32)
        feature['n2n'] = fearure_n2n.astype(np.float32)
        feature['dim'] = fearure_dim
        feature['max_cost'] = self.max_cost_const
        feature['part_num'] = self.parts_num

        # cost normalize
        ##devide cost by 2 because we define (c1+c2)/2 as cost of a step of pick&place
        cost_node2node = cost_node2node / 2.
        cost_node = cost_node / 2.
        # cost normalize

        cost_data['n'] = cost_node
        cost_data['n2n'] = cost_node2node

        mask_data['n'] = mask_node
        mask_data['n2n'] = mask_node2node

        return feature, cost_data, mask_data, parts_mask

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

    def add_constraint_mask(self, m, n, m0, n0, robots_ee, robots_goal, parts_init, parts_goal):
        ##### add additional mask constraint
        add_constraint = False
        if self.mask_constraint == "neighbour_index":
            if abs(m - n) == 1:
                add_constraint = True
        elif self.mask_constraint == "pose_overlap" and (m < self.parts_num and n < self.parts_num):
            x1 = parts_init[m][:2]
            y1 = parts_goal[m][:2]
            x2 = parts_init[n][:2]
            y2 = parts_goal[n][:2]

            e1 = robots_ee[0][:2] if m0 == -1 else parts_goal[m0][:2]
            b1 = robots_goal[0][:2]
            e2 = robots_ee[1][:2] if n0 == -1 else parts_goal[n0][:2]
            b2 = robots_goal[1][:2]

            intersect1, _ = tools.get_line_intersection(x1, b1, x2, b2)
            intersect2, _ = tools.get_line_intersection(y1, b1, y2, b2)

            if intersect1 or intersect2 :
                add_constraint = True
        elif self.mask_constraint == "triangle_overlap" and (m < self.parts_num and n < self.parts_num):
            x1 = parts_init[m][:2]
            y1 = parts_goal[m][:2]
            x2 = parts_init[n][:2]
            y2 = parts_goal[n][:2]

            e1 = robots_ee[0][:2] if m0 == -1 else parts_goal[m0][:2]
            b1 = robots_goal[0][:2]
            e2 = robots_ee[1][:2] if n0 == -1 else parts_goal[n0][:2]
            b2 = robots_goal[1][:2]

            tri_exb_1 = [e1, x1, b1]
            tri_exb_2 = [e2, x2, b2]
            tri_xyb_1 = [x1, y1, b1]
            tri_xyb_2 = [x2, y2, b2]

            polygons_exb = tools.get_overlap_convex_polygon(tri1_points=tri_exb_1, tri2_points=tri_exb_2)
            polygons_xyb = tools.get_overlap_convex_polygon(tri1_points=tri_xyb_1, tri2_points=tri_xyb_2)
            if len(polygons_exb) > 0 or len(polygons_xyb) > 0:
                add_constraint = True

        elif self.mask_constraint == "random_number" and (m < self.parts_num and n < self.parts_num):
            x1 = parts_init[m][:2]
            y1 = parts_goal[m][:2]
            x2 = parts_init[n][:2]
            y2 = parts_goal[n][:2]

            e1 = robots_ee[0][:2] if m0 == -1 else parts_goal[m0][:2]
            b1 = robots_goal[0][:2]
            e2 = robots_ee[1][:2] if n0 == -1 else parts_goal[n0][:2]
            b2 = robots_goal[1][:2]

            task_value = (x1 + y1 + e1 + x2 + y2 + e2)
            task_value = task_value.sum()
            task_value = int(task_value * 100000) % 10
            if task_value < 2:
                add_constraint = True
        else:
            pass
        return add_constraint
