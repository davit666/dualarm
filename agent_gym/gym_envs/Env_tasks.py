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
from bullet_env.robot import Robot
from bullet_env.part import Part
from bullet_env.p_utils import (
    create_cube_object,
    draw_sphere_body,
    draw_cube_body,
    remove_markers,
    remove_obstacles
)


# largeValObservation = np.inf

class Mode():
    def __init__(self):
        self.robot_mode_list = ["transit", "transfer"]
        self.moving_mode_list = ["up", "moving", "down"]
        self.counter_threshold = 2
        self.switch_counter = 0
        self.initialize_mode()

    def initialize_mode(self):
        self.moving_mode_idx = 1
        self.robot_mode_idx = 0
        self.switch_counter = 0
        self.update_mode()

    def check_counter(self):
        if self.switch_counter >= self.counter_threshold:
            return True
        else:
            self.switch_counter += 1
            return False

    def switch_mode(self):
        self.switch_counter = 0
        self.moving_mode_idx += 1
        if self.moving_mode_idx >= len(self.moving_mode_list):
            self.robot_mode_idx += 1
            self.moving_mode_idx = 0
            if self.robot_mode_idx >= len(self.robot_mode_list):
                self.robot_mode_idx = 0
        self.update_mode()

    def update_mode(self):
        self.robot_mode = self.robot_mode_list[self.robot_mode_idx]
        self.moving_mode = self.moving_mode_list[self.moving_mode_idx]

    def get_mode(self):
        return [self.robot_mode, self.moving_mode]

    def get_mode_id(self):
        if self.robot_mode == "transit" and self.moving_mode == "moving":
            return 0
        elif self.robot_mode == "transit" and self.moving_mode == "down":
            return 1
        elif self.robot_mode == "transfer" and self.moving_mode == "up":
            return 2
        elif self.robot_mode == "transfer" and self.moving_mode == "moving":
            return 3
        elif self.robot_mode == "transfer" and self.moving_mode == "down":
            return 4
        elif self.robot_mode == "transit" and self.moving_mode == "up":
            return 5
        else:
            print("emergengy occured !!!!! mode error")
            print([self.robot_mode, self.moving_mode])
            return -1


maximum_cost = 10000
largeValObservation = np.inf  ###############


class Env_tasks(gym.GoalEnv):
    metadata = {}

    def __init__(self,
                 env_config,
                 urdfRoot=pybullet_data.getDataPath(),
                 useInverseKinematics=None,
                 renders=False,
                 showBallMarkers=False,
                 freezeAction=False,
                 task_allocator_action_type="MultiDiscrete",
                 task_allocator_reward_type=None,
                 task_allocator_obs_type=None,
                 motion_planner_reward_type="delta_dist_with_sparse_reward",
                 motion_planner_obs_type="common_obs",
                 motion_planner_action_type="ee",
                 parts_num=6,
                 fragment_length=20,
                 maxSteps=300):

        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._freezeAction = freezeAction
        self._showBallMarkers = showBallMarkers

        self._useInverseKinematics = useInverseKinematics

        self._renders = renders
        self._maxSteps = maxSteps
        self._fragment_length = fragment_length
        self.parts_num = parts_num

        self.action_type = task_allocator_action_type
        self.task_allocator_reward_type = task_allocator_reward_type
        self.task_allocator_obs_type = task_allocator_obs_type
        self.motion_planner_reward_type = motion_planner_reward_type
        self.motion_planner_obs_type = motion_planner_obs_type
        self.motion_planner_action_type = motion_planner_action_type

        self.success_dist_threshold = env_config['success_dist_threshold'] + 0.1
        self.lift_height = 0.55  # 0.15
        self.dead_lock_count_threshold = 400 / fragment_length
        self.use_reset = False

        self.sync_execution = True

        self.selected_planner = self.go_straight_planner

        self.observation_history_length = env_config['observation_history_length']

        ######## load robot env and robot policy
        self.robots_env = Env_gr(env_config, renders=self._renders, useInverseKinematics=self._useInverseKinematics,
                                 freezeAction=self._freezeAction,
                                 showBallMarkers=self._showBallMarkers, maxSteps=500,
                                 reward_type=self.motion_planner_reward_type, obs_type=self.motion_planner_obs_type,
                                 action_type=self.motion_planner_action_type, in_task=True)
        self.robots_num = self.robots_env.robots_num
        self.base_height = self.robots_env._partsBaseSize[2] * 2

        self.robot_policy = None
        self.action_dim = self.parts_num + 1
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

    def assign_policy(self, policy=None, type="sb3"):
        if policy is None:
            print("no policy input")
            return None
        if type == "sb3":
            self.robot_policy = policy
            self.selected_planner = self.sb3_ppo_planner
            return True
        else:
            return None

    def setup(self):
        ######## define paras
        self.env_step_counter = 0
        self.need_task_allocation = True
        self.markers = []

        ######## get robots
        self.robots = []
        for robot in self.robots_env.robots:
            self.robots.append(robot)
        ######## assign mode to robot
        for robot in self.robots:
            robot.mode = Mode()
        ######## load parts and get parts
        self.parts = []
        # self.parts_colors = ['b', 'b2', 'b3', 'b4', 'b5', 'b6']
        for j in range(self.parts_num):
            # part_color = self.parts_colors[j]
            # print(part_color)
            part = Part(useInverseKinematics=self._useInverseKinematics, type='b')
            self.parts.append(part)

        p.stepSimulation()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.terminated = 0
        self.success = 0
        self.fail = 0
        self.env_step_counter = 0
        self.need_task_allocation = True
        self.state_info = {}
        self.prev_state_info = {}
        self.clear_markers()
        self.accumulated_reward = 0
        self.dead_lock_count = 0
        self.stop_with_task_undone = False
        self.accumulated_reward_dict = {}
        self.accumulated_reward_dict['robots'] = [0] * self.robots_num
        self.accumulated_reward_dict['parts'] = [0] * self.parts_num
        self.accumulated_reward_dict['task'] = 0
        self.accumulated_reward_dict['triangle'] = [0] * self.robots_num

        ######## robot reset
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

        # global_poses_base = goal_poses_base + init_poses_base
        # random.shuffle(global_poses_base)
        # init_poses_base = global_poses_base[:self.parts_num]
        # goal_poses_base = global_poses_base[self.parts_num:]

        random.shuffle(goal_poses_base)
        random.shuffle(init_poses_base)
        ######################################################
        if np.random.random() < 0:
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

    def get_mode(self, robot_list=[], use_mode_id=False):
        if len(robot_list) == 0:
            robot_list = self.robots
        if use_mode_id:
            robots_mode_ids = []
            for robot in robot_list:
                robot_mode_id = robot.mode.get_mode_id()
                robots_mode_ids.append(robot_mode_id)
            return robots_mode_ids
        else:
            robots_modes = []
            for robot in robot_list:
                robot_mode, moving_mode = robot.mode.get_mode()
                robots_modes.append([robot_mode, moving_mode])
            return robots_modes

    def get_states(self):
        self.state_info = {}
        ######## get robots states
        robots_js = []
        robots_ee = []
        robots_goal = []

        for robot in self.robots:
            js = robot.getObservation_JS()
            js = robot.normalize_JS(js)
            ee = robot.getObservation_EE(useInverseKinematics=True)
            ee = self.robots_env.normalize_cartesian_pose(ee)
            goal = robot.goal
            goal = self.robots_env.normalize_cartesian_pose(goal)

            robots_js.append(js)
            robots_ee.append(ee)
            robots_goal.append(goal)

        ######## get parts pose and goal
        parts_pose = []
        parts_goal = []
        for part in self.parts:
            grasp_pose = part.getGraspPose(useInverseKinematics=True)
            grasp_pose = self.robots_env.normalize_cartesian_pose(grasp_pose)
            goal_pose = part.getGoalPose(useInverseKinematics=True)
            goal_pose = self.robots_env.normalize_cartesian_pose(goal_pose)

            parts_pose.append(grasp_pose)
            parts_goal.append(goal_pose)

        ######## get task mask, mode
        robots_mask_allocated = np.zeros((self.robots_num, self.action_dim))
        robots_mask_available = np.zeros((self.robots_num, self.action_dim))

        for i, robot in enumerate(self.robots):
            task_idx_allocated = robot.task_idx_allocated
            robots_mask_allocated[i, task_idx_allocated + 1] = 1

            if robot.is_in_process:
                robots_mask_available[i, task_idx_allocated + 1] = 1
            else:
                robots_mask_available[i, 0] = 1
                for j, part in enumerate(self.parts):
                    if not (part.is_in_process or part.is_success):
                        robots_mask_available[i, j + 1] = 1

        robots_mode_ids = np.array(self.get_mode(use_mode_id=True))
        parts_mode_ids = np.array([0] * self.parts_num)
        parts_success = np.array([0] * self.parts_num)
        for j, part in enumerate(self.parts):
            if part.is_success:
                parts_success[j] = 1
            elif part.is_in_process:
                assert (1 in robots_mask_allocated[:, j + 1])
                for i, robot in enumerate(self.robots):
                    if robot.is_in_process and (robot.task_idx_allocated == j):
                        parts_mode_ids[j] = robots_mode_ids[i]
        ######## calculate cost of edges due to mode
        #### dist between robot and part
        dist_robots2parts = np.zeros((self.robots_num, self.parts_num))
        for i, robot in enumerate(self.robots):
            for j, part in enumerate(self.parts):
                dist_robots2parts[i, j] = np.linalg.norm(robots_ee[i] - parts_pose[j])
                ##### should rotation distance allowed to be counted as a part of distance cost ??????

        #### dist between part and part
        dist_parts2parts = np.zeros((self.parts_num, self.parts_num))
        for j in range(self.parts_num):
            for k in range(self.parts_num):
                if k != j:
                    part_j, part_k = self.parts[j], self.parts[k]
                    if any([part_j.is_success, part_k.is_success]):
                        dist_parts2parts[j, k] = maximum_cost
                    else:
                        dist_parts2parts[j, k] = np.linalg.norm(parts_goal[j] - parts_pose[k])
                else:
                    dist_parts2parts[j, j] = maximum_cost

        ######## get robot, part, mask state observation
        #### robot obs: js,ee,goal,mode,dist2parts
        robots_obs = []
        for i, robot, in enumerate(self.robots):
            robot_obs = np.concatenate(
                [robots_js[i], robots_ee[i], robots_goal[i], [robots_mode_ids[i]], robots_mask_allocated[i, :],
                 dist_robots2parts[i, :]])
            robots_obs.append(robot_obs)
        #### part obs: pose,goal,in_process,success,mode,dist2parts
        parts_obs = []
        for j, part in enumerate(self.parts):
            part_obs = np.concatenate(
                [parts_pose[j], parts_goal[j], [part.is_in_process], [part.is_success], [parts_mode_ids[j]],
                 dist_parts2parts[j, :]])
            parts_obs.append(part_obs)
        #### mask obs: choosable
        masks_obs = []
        for i, robot in enumerate(self.robots):
            mask_obs = robots_mask_available[i, :]
            masks_obs.append(mask_obs)

        #### get observation list
        observation = [np.concatenate(robots_obs), np.concatenate(parts_obs), np.concatenate(masks_obs)]
        ######## record state infrmation
        self.state_info['robots_obs'] = robots_obs
        self.state_info['parts_obs'] = parts_obs
        self.state_info['masks_obs'] = masks_obs

        self.state_info['robots_mode_id'] = robots_mode_ids
        self.state_info['robot_in progress'] = np.array([robot.is_in_process for robot in self.robots])

        self.state_info['parts_success'] = parts_success
        self.state_info['parts_in_progress'] = np.array([part.is_in_process for part in self.parts])
        self.state_info['parts_mode_id'] = parts_mode_ids

        self.state_info['robots_mask_available'] = robots_mask_available
        self.state_info['robots_mask_allocated'] = robots_mask_allocated

        self.state_info['mode_switch_count'] = np.array([robot.mode_switched for robot in self.robots])

        #### for deadlock checking
        dead_lock_dist_log = {}
        for i, robot in enumerate(self.robots):
            dead_lock_dist_log['robot_{}_goal'.format(i + 1)] = np.linalg.norm(robots_ee[i] - robots_goal[i])
            for k, other_robot in enumerate(self.robots):
                if i < k:
                    dead_lock_dist_log['robot_{}_robot_{}'.format(i + 1, k + 1)] = np.linalg.norm(
                        robots_ee[i] - robots_ee[k])
        self.state_info['dead_lock_dist_log'] = dead_lock_dist_log
        self.state_info['execute_planner_full_execution'] = False
        ###################
        dist_cost = {}
        for i, robot in enumerate(self.robots):
            for j, part in enumerate(self.parts):
                dist_cost["dist_robot_{}_to_part_{}".format(i + 1, j + 1)] = dist_robots2parts[i, j]
        for j in range(self.parts_num):
            for k in range(self.parts_num):
                dist_cost["dist_part_{}_to_part_{}".format(j + 1, k + 1)] = dist_parts2parts[j, k]
        self.state_info['dist_cost'] = dist_cost
        ###################
        return observation

    def step(self, allocator_action):
        self.env_step_counter += 1

        if self.action_type == "Box":
            allocator_action = self.sample_allocator_action(allocator_action)
        self.allocate_task(allocator_action)

        full_execution, no_execution = self.execute_planner()
        observation = self._observation()
        self.state_info['allocator_action'] = allocator_action
        self.state_info['execute_planner_full_execution'] = full_execution
        self.state_info['execute_planner_no_execution'] = no_execution
        #### check dead lock
        # no mode change
        if all([self.state_info['mode_switch_count'][i] == self.prev_state_info['mode_switch_count'][i] for i in
                range(self.robots_num)]):
            # no task change
            task_unchanged = True
            for i in range(self.robots_num):
                task_unchanged = task_unchanged and np.dot(self.state_info['robots_mask_allocated'][i, :],
                                                           self.prev_state_info['robots_mask_allocated'][i, :])
            if task_unchanged:
                self.dead_lock_count += 1
            else:
                self.dead_lock_count = 0
        else:
            self.dead_lock_count = 0

        reward = self._reward()
        done = self._termination()

        if done:
            episode_info = {}
            episode_info['1_success/is_success'] = self.success
            episode_info['2_task_done/all'] = sum(self.state_info['parts_success'])
            episode_info['3_num_steps/num_steps_in_a_episode_when_success'] = self._maxSteps if any(
                robot.is_failed for robot in self.robots) else self.env_step_counter
            episode_info['3_num_steps/num_steps_in_a_episode'] = self.env_step_counter
            if self.success:
                episode_info['3_num_steps/num_steps_in_a_episode_only_success'] = self.env_step_counter
            else:
                episode_info['3_num_steps/num_steps_in_a_episode_without_success'] = self.env_step_counter
            episode_info['4_fail/all'] = any([robot.is_failed for robot in self.robots]) or self.fail
            episode_info['4_fail/dead_lock'] = self.dead_lock_count > self.dead_lock_count_threshold
            episode_info['4_fail/finish_with_task_undone'] = 1 if self.env_step_counter >= self._maxSteps else 0
            episode_info['4_fail/stop_with_task_undone'] = self.stop_with_task_undone
            episode_info['5_accumulated_reward/all_reward'] = self.accumulated_reward
            episode_info['5_average_reward/all_reward'] = self.accumulated_reward / self.env_step_counter
            episode_info['5_accumulated_reward/task'] = self.accumulated_reward_dict['task']
            episode_info['5_average_reward/task'] = self.accumulated_reward_dict['task'] / self.env_step_counter
            for i, robot in enumerate(self.robots):
                episode_info['2_task_done/robot_{}'.format(i + 1)] = robot.success_count
                episode_info['4_fail/robot_{}'.format(i + 1)] = robot.is_failed
                episode_info['5_accumulated_reward/robot_{}'.format(i + 1)] = self.accumulated_reward_dict['robots'][i]
                episode_info['5_accumulated_reward/triangle_{}'.format(i + 1)] = \
                    self.accumulated_reward_dict['triangle'][i]
                episode_info['5_average_reward/robot_{}'.format(i + 1)] = self.accumulated_reward_dict['robots'][
                                                                              i] / self.env_step_counter
                episode_info['5_average_reward/triangle_{}'.format(i + 1)] = self.accumulated_reward_dict['triangle'][
                                                                                 i] / self.env_step_counter
                episode_info['6_working_time/robot_{}'.format(i + 1)] = 1 - robot.slack_step / self.env_step_counter
                episode_info['7_task_switched/robot_{}'.format(i + 1)] = robot.task_switched
                episode_info['8_mode_switched/robot_{}'.format(i + 1)] = robot.mode_switched
                episode_info['9_accumulated_dist/robot_{}'.format(i + 1)] = robot.accumulated_dist

                if self.robots[i].is_collision:
                    coll_with_obj = True
                    for k in range(self.robots_num):
                        if (self.robots[i].is_collision and self.robots[k].is_collision):
                            if i != k:
                                coll_with_obj = False
                            if i < k:
                                episode_info['4_fail/hit_mutual_{}_{}'.format(i + 1, k + 1)] = 1
                                if self.state_info['robots_mode_id'][i] in [1, 2, 4, 5] or \
                                        self.state_info['robots_mode_id'][k] in [1, 2, 4, 5]:
                                    episode_info['4_fail/hit_mutual_{}_{}_in_mode_switch'.format(i + 1, k + 1)] = 1
                                else:
                                    episode_info['4_fail/hit_mutual_{}_{}_in_mode_switch'.format(i + 1, k + 1)] = 0
                        else:
                            if i < k:
                                episode_info['4_fail/hit_mutual_{}_{}'.format(i + 1, k + 1)] = 0
                                episode_info['4_fail/hit_mutual_{}_{}_in_mode_switch'.format(i + 1, k + 1)] = 0
                    episode_info['4_fail/hit_robot_{}'.format(i + 1)] = coll_with_obj
                else:
                    episode_info['4_fail/hit_robot_{}'.format(i + 1)] = False
                    for k in range(self.robots_num):
                        if i < k:
                            episode_info['4_fail/hit_mutual_{}_{}'.format(i + 1, k + 1)] = 0
                            episode_info['4_fail/hit_mutual_{}_{}_in_mode_switch'.format(i + 1, k + 1)] = 0
            for j, part in enumerate(self.parts):
                episode_info['5_accumulated_reward/spart_{}'.format(j + 1)] = self.accumulated_reward_dict['parts'][j]
                episode_info['5_average_reward/spart_{}'.format(j + 1)] = self.accumulated_reward_dict['parts'][
                                                                              j] / self.env_step_counter


        else:
            episode_info = {'2_task_done/all': sum(self.state_info['parts_success'])}
        self.prev_state_info = self.state_info

        return observation, reward, done, episode_info

    def _observation(self):
        obs_dict = self.get_states()
        observation = np.concatenate(obs_dict)

        if len(self.observation_history) == self.observation_history_length:
            self.observation_history.pop(0)
        self.observation_history.append(observation)
        # print("obs_history length:  ", len(self.observation_history))
        # print(self.observation_history[-1])

        return observation

    def _reward(self):

        #### dead lock check
        if self.dead_lock_count > self.dead_lock_count_threshold:
            if self._renders:
                print("dead lock occured")
            for robot in self.robots:
                robot.is_failed = True
        #### all robot slack check
        if all([robot.is_success and robot.task_idx_allocated == -1 for robot in self.robots]):
            if not all([task_part.is_success for task_part in self.parts]):
                for robot in self.robots:
                    robot.is_failed = True

            # if self.state_info['execute_planner_full_execution']:
            #     dead_lock_dist_log = self.state_info['dead_lock_dist_log']
            #     prev_dead_lock_dist_log = self.prev_state_info['dead_lock_dist_log']
            #     for i, robot in enumerate(self.robots):
            #         for k, other_robot in enumerate(self.robots):
            #             if i < k:
            #                 dead_lock_dist = dead_lock_dist_log['robot_{}_goal'.format(i + 1)] + dead_lock_dist_log[
            #                     'robot_{}_goal'.format(k + 1)] + dead_lock_dist_log[
            #                                      'robot_{}_robot_{}'.format(i + 1, k + 1)]
            #                 prev_dead_lock_dist = prev_dead_lock_dist_log['robot_{}_goal'.format(i + 1)] + \
            #                                       prev_dead_lock_dist_log[
            #                                           'robot_{}_goal'.format(k + 1)] + prev_dead_lock_dist_log[
            #                                           'robot_{}_robot_{}'.format(i + 1, k + 1)]
            #                 delta_dead_lock_dist = abs(dead_lock_dist - prev_dead_lock_dist)
            #                 # print('delta_dead_lock_dist:\t', delta_dead_lock_dist)
            #                 if delta_dead_lock_dist <= 0.01 and dead_lock_dist_log[
            #                     'robot_{}_robot_{}'.format(i + 1, k + 1)] < 0.1:
            #                     if self._renders:
            #                         print("dead lock")
            #                     robot.is_failed = True
            #                     other_robot.is_failed = True

        #### for robot
        robots_rewards = []
        for i, robot in enumerate(self.robots):
            robot_reward = 0

            task_allocated = self.state_info["robots_mask_allocated"][i, :]
            task_allocated_idx = np.where(task_allocated == 1)[0] - 1
            prev_task_allocated = self.prev_state_info["robots_mask_allocated"][i, :]
            prev_task_allocated_idx = np.where(prev_task_allocated == 1)[0] - 1
            assert len(task_allocated_idx) <= 1 and len(prev_task_allocated_idx) <= 1

            # task change (bad)
            if len(task_allocated_idx) == 1:
                if len(prev_task_allocated_idx) != 0:
                    curr_task_idx = task_allocated_idx[0]
                    prev_task_idx = prev_task_allocated_idx[0]
                    if curr_task_idx != prev_task_idx and prev_task_idx != -1:
                        if not self.parts[prev_task_idx].is_success:
                            robot.task_switched += 1
                            robot_reward -= 0.1

            # not working(bad)
            if (self.state_info["robots_mask_allocated"][i, 0] == 1) and (
                    1 in self.state_info["robots_mask_available"][i, 1:]):
                robot.slack_step += 1
                robot_reward -= 0.1

            # collision or deadlock(bad)
            if robot.is_failed:
                robot_reward -= 10

            robots_rewards.append(robot_reward)

        triangle_penaltys = []
        for i, robot in enumerate(self.robots):
            # base_xy = robot.BasePos[:2]
            # ee_xy = robot.getObservation_EE()[:2]
            # goal_xy = robot.goal[:2]
            # robot_triangle_area = tools.calcul_triangle_area(base_xy, ee_xy, goal_xy,
            #                                                  minimum_triangle_area=self.robots_env.minimum_triangle_area)
            # total_overlap_area = 0
            # maximum_cutting_ratio = 0
            # if robot.mode.get_mode_id() in [3, 6]:
            #     for other_robot in self.robots:
            #         if robot == other_robot:
            #             continue
            #         overlap_area, cutting_ratio = self.robots_env.calcul_robot_overlap_working_area_and_cutting_ratio(
            #             robot, other_robot)
            #
            #         # total_overlap_area += overlap_area
            #         maximum_cutting_ratio = max(maximum_cutting_ratio, cutting_ratio)
            # robot_cutting_ratio_penalty = - maximum_cutting_ratio * 1
            # # robot_overlap_area_penalty = - total_overlap_area * 0.1

            robot_cutting_ratio_penalty = 0
            triangle_penaltys.append(robot_cutting_ratio_penalty)

        #### for parts
        parts_rewards = []
        for j, part in enumerate(self.parts):
            part_reward = 0
            # part in process(good)
            is_in_process = self.state_info['parts_in_progress'][j]
            pre_is_in_process = self.prev_state_info['parts_in_progress'][j]
            if is_in_process and not pre_is_in_process:
                part_reward += 0.1

            # part mode switch(good)
            part_mode_id = self.state_info['parts_mode_id'][j]
            prev_part_mode_id = self.prev_state_info['parts_mode_id'][j]
            if part_mode_id != prev_part_mode_id:
                part_reward += 0.1

            # part succeed(good)
            is_success = self.state_info['parts_success'][j]
            prev_is_success = self.prev_state_info['parts_success'][j]
            if is_success and not prev_is_success:
                part_reward += 1
            parts_rewards.append(part_reward)

        #### for task
        task_actions = np.zeros((self.robots_num, self.action_dim))
        allocator_action = self.state_info['allocator_action']
        for i, robot, in enumerate(self.robots):
            idx = allocator_action[i] % self.action_dim
            task_actions[i, idx] = 1

        task_reward = 0
        for i, robot in enumerate(self.robots):
            # action not available(bad)
            mask_available = self.prev_state_info['robots_mask_available'][i, :]
            if np.dot(mask_available, task_actions[i, :]) == 0:
                task_reward -= 1
            # # action same as others(bad)
            # for k, other_robot in enumerate(self.robots):
            #     if i != k and np.dot(task_actions[i, 1:], task_actions[k, 1:]) > 0:
            #         task_reward -= 0.5

        #### all subtasks complete(good)
        if all(self.state_info['parts_success']) and all(
                [robot.is_success and robot.task_idx_allocated == -1 for robot in self.robots]):
            task_reward += 10

        #### sum all rewards
        reward = sum(robots_rewards) + sum(parts_rewards) + task_reward + sum(triangle_penaltys)

        for i in range(self.robots_num):
            self.accumulated_reward_dict['robots'][i] += robots_rewards[i]
            self.accumulated_reward_dict['triangle'][i] += triangle_penaltys[i]
        for j in range(self.parts_num):
            self.accumulated_reward_dict['parts'][j] += parts_rewards[j]
        self.accumulated_reward_dict['task'] += task_reward
        self.accumulated_reward += reward

        return reward

    def _termination(self):
        if self.env_step_counter >= self._maxSteps:
            self.terminated = True
            # self.fail = 1
        if any([robot.is_failed for robot in self.robots]):
            self.terminated = True
            self.fail = 1
            if self._renders:
                for i, robot in enumerate(self.robots):
                    if robot.is_failed:
                        print("robot_{} failed".format(i + 1))

        if all([robot.is_success and robot.task_idx_allocated == -1 for robot in self.robots]):
            self.terminated = True
            if all([task_part.is_success for task_part in self.parts]):
                self.success = 1
                if self._renders:
                    print("goallllllll!!!!!!!!!!!!!!!!!!!!")
            else:
                self.fail = 1
                self.stop_with_task_undone = True
                if self._renders:
                    print("robots stopped with tasks undone !!!!!!!!!!!!!!!!!!!!")
        return self.terminated

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
        ######## extract allocator_action
        assert len(allocator_action) == self.robots_num, "allocator action:{} is not equal to robot num:{}".format(
            len(allocator_action, self.robots_num))
        tasks_selected = []
        available_mask = self.state_info['robots_mask_available']
        for i, robot in enumerate(self.robots):
            idx = allocator_action[i] % self.action_dim - 1
            if available_mask[i, idx + 1] == 0:
                print("@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                if robot.is_in_process:
                    idx = robot.task_idx_allocated
                else:
                    idx = -1
            tasks_selected.append(idx)

        if self._renders:
            print("action: \t", allocator_action)
            print("mask  :\t", available_mask)

        ######## get mode
        # robots_modes = self.get_mode()
        ######## assign task and goal due to mode and action
        robots_goals = []
        for i, robot in enumerate(self.robots):
            task_idx = tasks_selected[i]
            #### assign task
            robot.task_idx_allocated = task_idx
            #### assign goal
            if task_idx == -1:
                robot_goal = self.assign_robot_goal(robot)
            else:
                task_part = self.parts[task_idx]
                robot_goal = self.assign_robot_goal(robot, task_part)
            robots_goals.append(robot_goal)

        # for i, robot in enumerate(self.robots):
        #     print("robot_{}_have task_{}".format(i + 1, tasks_selected[i] + 1), "\tgoal is:\t", robot.goal)
        #     print("task start:\t", self.parts[tasks_selected[i]].getGraspPose())
        #     print("task end:\t", self.parts[tasks_selected[i]].getGoalPose())
        self.need_task_allocation = False
        return True

    def assign_robot_goal(self, robot, task_part=None):
        if task_part is None or robot.task_idx_allocated == -1:
            ######## force robot goal back to its initial pose if no task allocated
            robot.resetGoalPose(default_pose=True)
            return robot.goal
        # [robot_mode, moving_mode] = self.get_mode([robot])[0]
        # if robot_mode == "transit" and moving_mode == "moving":
        #     robot_goal = task_part.getGraspPose()
        #     robot_goal[2] += self.lift_height
        # elif robot_mode == "transit" and moving_mode == "down":
        #     robot_goal = task_part.getGraspPose()
        # elif robot_mode == "transfer" and moving_mode == "up":
        #     robot_goal = task_part.getGraspPose()
        #     robot_goal[2] += self.lift_height
        # elif robot_mode == "transfer" and moving_mode == "moving":
        #     robot_goal = task_part.getGoalPose()
        #     robot_goal[2] += self.lift_height
        # elif robot_mode == "transfer" and moving_mode == "down":
        #     robot_goal = task_part.getGoalPose()
        # elif robot_mode == "transit" and moving_mode == "up":
        #     robot_goal = task_part.getGoalPose()
        #     robot_goal[2] += self.lift_height
        # else:
        #     print("robot mode gets wrong: ", robot_mode[0])
        #     robot_goal = None
        robot_mode_id = self.get_mode([robot], use_mode_id=True)[0]
        if robot_mode_id == 0:
            robot_goal = task_part.getGraspPose()
            robot_goal[2] = self.lift_height
        elif robot_mode_id == 1:
            robot_goal = task_part.getGraspPose() + 0.01
        elif robot_mode_id == 2:
            robot_goal = task_part.getGraspPose()
            robot_goal[2] = self.lift_height
        elif robot_mode_id == 3:
            robot_goal = task_part.getGoalPose()
            robot_goal[2] = self.lift_height
        elif robot_mode_id == 4:
            robot_goal = task_part.getGoalPose() + 0.01
        elif robot_mode_id == 5:
            robot_goal = task_part.getGoalPose()
            robot_goal[2] = self.lift_height
        else:
            print("robot mode gets wrong, id : ", robot_mode_id)
            robot_goal = None
        robot.resetGoalPose(robot_goal)
        return robot_goal

    def execute_planner(self, customize_fragment_length=None):
        ######## define fragment length to execute motion planner
        fragment_length = customize_fragment_length if customize_fragment_length is not None else self._fragment_length

        ######## assign planners due to current states
        robots_modes, robots_planners = self.assign_planners()
        # print("start new fragment @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # for i, robot in enumerate(self.robots):
        #     print("robot_{}_mode: \t".format(i + 1), robots_modes[i])
        #     print("planner is:\t", robots_planners[i])
        # time.sleep(5)
        ########loop start
        for fragment_count in range(fragment_length):
            #### get planner action due to goal and policy
            if self.selected_planner in robots_planners:
                robots_planner_actions = self.selected_planner()
                for i, robot in enumerate(self.robots):
                    if robots_planners[i] != self.selected_planner:
                        robots_planner_actions[i] = robots_planners[i]([robot])[0]
            else:
                robots_planner_actions = [np.array([0] * self.robots_env.action_dim)] * self.robots_num
                for i, robot in enumerate(self.robots):
                    robots_planner_actions[i] = robots_planners[i]([robot])[i]
            # print(robots_modes)
            #### apply planner action to robots and get planner termination, mode change
            #### if planner terminated, implement mode change and apply extra action(pick&place) due to current mode
            planner_action = np.concatenate(robots_planner_actions)

            _, _, _, _ = self.robots_env.step(planner_action, scale_action=False, use_reset=self.use_reset)

            for i, robot in enumerate(self.robots):
                ee = robot.getObservation_EE()
                g = robot.goal_pose
                print("robot:\t", i, "\tgoal:\t", g, "\tee:\t", ee, "\tsuccess:\t", robot.is_success, "\nrobot:\t", i,
                      "\tmode:\t", robots_modes[i][0], robots_modes[i][1],"\tplanner:\t", robots_planners[i] )

            fragment_terminated = False
            if all([robot.is_success and robot.task_idx_allocated == -1 for robot in self.robots]):
                # env finished
                fragment_terminated = True
            # else:

            elif (self.sync_execution and all([robot.is_success for robot in self.robots])) or not self.sync_execution:
                ##############################end
                # time.sleep(3)
                for robot in self.robots:

                    if robot.is_success and robot.mode.check_counter():
                        # a robot is success, switch its mode
                        self.assign_mode_switch(robot)
                        # re-assign planners due to mode switch
                        robots_modes, robots_planners = self.assign_planners()
                        if self.need_task_allocation:
                            fragment_terminated = True
                    elif robot.is_failed:
                        fragment_terminated = True

            #### check loop number and planner termination, break the loop if condition satisfies
            if fragment_terminated:
                break
        if self._renders:
            print("fragment end, the length is:\t", fragment_count + 1)
        return fragment_count + 1 >= fragment_length, fragment_count <= 1

    def assign_planners(self):
        ######## get mode
        robots_modes = self.get_mode()
        ######## choose police due to mode
        robots_planners = []
        for i, robot in enumerate(self.robots):
            robot_mode = robots_modes[i][0]
            moving_mode = robots_modes[i][1]
            if moving_mode in ["up", "down"]:
                robot_planner = self.go_straight_planner
            elif moving_mode == "moving":
                # if robot.is_success:
                if (self.sync_execution and all([robot_tmp.is_success for robot_tmp in self.robots])) or (
                        not self.sync_execution and robot.is_success):
                    robot_planner = self.go_straight_planner
                else:
                    robot_planner = self.selected_planner
            else:
                robot_planner = None
            robots_planners.append(robot_planner)
        return robots_modes, robots_planners

    def assign_mode_switch(self, robot=None):
        if robot not in self.robots:
            return None
        assert robot.is_success, "robot is not success to his current goal yet, no need to switch mode"
        robot.is_success = False
        task_part_idx = robot.task_idx_allocated
        ######## force robot goal back to its initial pose if no task allocated
        if task_part_idx == -1:
            self.assign_robot_goal(robot)
            return None
        ######## assign mode switch
        robot_mode_id = self.get_mode([robot], use_mode_id=True)[0]
        task_part = self.parts[task_part_idx]
        mode_msg = ""
        if robot_mode_id == 0:
            #### reached above an object, a task starts to be executed, pair robot and task
            robot.is_in_process = True
            task_part.is_in_process = True
            mode_msg = "robot_{} reached above object_{}, a task starts to be executed, pair robot and task".format(
                self.robots.index(robot) + 1, task_part_idx + 1)
            ## need re-allocated task for other robot
            self.need_task_allocation = True
        elif robot_mode_id == 1:
            #### touched the object, start picking action
            part_idx_picked = self.apply_picking_item(robot, calibrate_gripper=False)
            assert part_idx_picked == task_part_idx + 1, "picked item is item_{}, while current task is item_{} for robot_{}".format(
                part_idx_picked, task_part_idx + 1, self.robots.index(robot))
            mode_msg = "robot_{} touched the object_{}, start picking action".format(
                self.robots.index(robot) + 1, task_part_idx + 1)
        elif robot_mode_id == 2:
            #### item successfully picked, robot lifted, start moving
            mode_msg = "object_{} successfully picked, robot_{} lifted, start moving".format(task_part_idx + 1,
                                                                                             self.robots.index(
                                                                                                 robot) + 1)
        elif robot_mode_id == 3:
            #### reached above goal of object, put down robot arm
            mode_msg = "robot_{} reached above goal of object_{}, start put down robot arm".format(
                self.robots.index(robot) + 1, task_part_idx + 1)
        elif robot_mode_id == 4:
            #### reached goal of object, start placing action
            res = self.apply_placing_item(robot)
            assert res, "no item to place"
            mode_msg = "robot_{} reached goal of object_{}, start placing action".format(
                self.robots.index(robot) + 1, task_part_idx + 1)
        elif robot_mode_id == 5:
            #### item successfully placed, robot lifted, the task finished, unpair robot and task
            task_part.checkIsPlaced()
            assert task_part.is_success, "item is not placed in the right place"
            robot.is_in_process = False
            robot.task_idx_allocated = -1
            task_part.is_in_process = False
            task_part.robot_idx_to_allocate = -1
            mode_msg = "object_{} successfully placed, robot_{} lifted, the task finished, unpair robot and task".format(
                task_part_idx + 1,
                self.robots.index(robot) + 1)
            ## need re-allocate task for current robot
            self.need_task_allocation = True
            robot.success_count += 1
        else:
            assert 0, "un founded robot mode id: {}".format(robot_mode_id)

        # [robot_mode, moving_mode] = self.get_mode([robot])[0]
        # task_part = self.parts[task_part_idx]
        # mode_msg = ""
        # if robot_mode == "transit" and moving_mode == "moving":
        #     #### reached above an object, a task starts to be executed, pair robot and task
        #     robot.is_in_process = True
        #     task_part.is_in_process = True
        #     mode_msg = "robot_{} reached above object_{}, a task starts to be executed, pair robot and task".format(
        #         self.robots.index(robot) + 1, task_part_idx + 1)
        #     ## need re-allocated task for other robot
        #     self.need_task_allocation = True
        # elif robot_mode == "transit" and moving_mode == "down":
        #     #### touched the object, start picking action
        #     part_idx_picked = self.apply_picking_item(robot, calibrate_gripper=False)
        #     assert part_idx_picked == task_part_idx + 1, "picked item is item_{}, while current task is item_{} for robot_{}".format(
        #         part_idx_picked, task_part_idx + 1, self.robots.index(robot))
        #     mode_msg = "robot_{} touched the object_{}, start picking action".format(
        #         self.robots.index(robot) + 1, task_part_idx + 1)
        # elif robot_mode == "transfer" and moving_mode == "up":
        #     #### item successfully picked, robot lifted, start moving
        #     mode_msg = "object_{} successfully picked, robot_{} lifted, start moving".format(task_part_idx + 1,
        #                                                                                      self.robots.index(
        #                                                                                          robot) + 1)
        # elif robot_mode == "transfer" and moving_mode == "moving":
        #     #### reached above goal of object, put down robot arm
        #     mode_msg = "robot_{} reached above goal of object_{}, start put down robot arm".format(
        #         self.robots.index(robot) + 1, task_part_idx + 1)
        # elif robot_mode == "transfer" and moving_mode == "down":
        #     #### reached goal of object, start placing action
        #     res = self.apply_placing_item(robot)
        #     assert res, "no item to place"
        #     mode_msg = "robot_{} reached goal of object_{}, start placing action".format(
        #         self.robots.index(robot) + 1, task_part_idx + 1)
        # elif robot_mode == "transit" and moving_mode == "up":
        #     #### item successfully placed, robot lifted, the task finished, unpair robot and task
        #     task_part.checkIsPlaced()
        #     assert task_part.is_success, "item is not placed in the right place"
        #     robot.is_in_process = False
        #     robot.task_idx_allocated = -1
        #     task_part.is_in_process = False
        #     task_part.robot_idx_to_allocate = -1
        #     mode_msg = "object_{} successfully placed, robot_{} lifted, the task finished, unpair robot and task".format(
        #         task_part_idx + 1,
        #         self.robots.index(robot) + 1)
        #     ## need re-allocate task for current robot
        #     self.need_task_allocation = True
        # else:
        #     assert 0, "un founded robot mode: {} {}".format(robot_mode, moving_mode)

        ######## update mode and re-assign goal
        robot.mode.switch_mode()
        robot.mode_switched += 1

        self.assign_robot_goal(robot, task_part)
        if self._renders:
            print(mode_msg)

        return True

    def clear_markers(self):
        remove_obstacles(self.markers)
        self.markers = []
        return True

    ####################################### extra actions: pick, place ################################
    def apply_picking_item(self, robot, item_selected=None, calibrate_gripper=False):
        ######## check if robot is currently picking an item
        if robot.item_picking is not None:
            if self._renders:
                print("robot is currently picking item No.{} , can not pick anymore".format(
                    self.parts.index(robot.item_picking) + 1))
            return False

        item_to_pick = None
        ee = robot.getObservation_EE()
        ######## check if selected item can be picked
        if item_selected is not None:
            grasp_pose = item_selected.getGraspPose() + 0.01
            if np.linalg.norm(ee - grasp_pose) < self.success_dist_threshold:
                min_dist = np.linalg.norm(ee - grasp_pose)
                item_to_pick = item_selected

        ######## check if there is any other item that can be picked
        else:
            min_dist = self.success_dist_threshold
            for j, part in enumerate(self.parts):
                grasp_pose = part.getGraspPose()
                if np.linalg.norm(ee - grasp_pose) < min_dist:  #########$$$$$$$$$$$$$$$$$$$$$
                    min_dist = np.linalg.norm(ee - grasp_pose)
                    item_to_pick = part

        ######## no item can be picked
        if item_to_pick is None:
            print("No item can be picked by current robot {}, min_dist:{}".format(self.robots.index(robot), min_dist))
            for j, part in enumerate(self.parts):
                grasp_pose = part.getGraspPose()
                print(np.linalg.norm(ee - grasp_pose))
            if self._renders:
                print("No item can be picked by current robot {}".format(self.robots.index(robot)))
            return False

        ######## item founded and calibrate robot gripper
        if calibrate_gripper:
            pick_action = robot.calculStraightAction2Goal(item_to_pick.getGraspPose())
            other_action = [0 for i in range(len(pick_action))]
            for i in range(self.robots_num):
                if i == self.robots.index(robot):
                    self.robots[i].applyAction(pick_action)
                else:
                    self.robots[i].applyAction(other_action)
        ######## apply picking action
        cid = robot.pickItem(item_to_pick)
        idx_to_pick = self.parts.index(item_to_pick)
        self.parts[idx_to_pick].picked(robot, cid)
        p.stepSimulation()
        return idx_to_pick + 1

    def apply_placing_item(self, robot):
        ######## check if robot have picking item
        if robot.item_picking is None:
            if self._renders:
                print("there is currently no item picked by robot {}".format(self.robots.index(robot)))
            return False
        else:

            item_to_place = robot.item_picking
            print("!!!!!!!!!!!!!!!!!!!!")
            print(self._useInverseKinematics)
            print("robot.item_picking", robot.item_picking)
            print("item_to_place.getGoalPose()       ", item_to_place.getGoalPose())
            ######## calibrate robot gripper and clear robot velocity to prepare placing
            place_action = robot.calculStraightAction2Goal(item_to_place.getGoalPose())
            other_action = [0 for i in range(len(place_action))]
            for i in range(self.robots_num):
                if i == self.robots.index(robot):
                    self.robots[i].applyAction(place_action)

                else:
                    self.robots[i].applyAction(other_action)
            p.stepSimulation()
            for i in range(self.robots_num):
                self.robots[i].applyAction(other_action)
            p.stepSimulation()
            ######## apply placing action
            robot.placeItem(item_to_place)

            idx_tp_place = self.parts.index(item_to_place)
            self.parts[idx_tp_place].placed(robot)

            return True

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

        # allocator_action = []
        # action_list = []
        # for i, robot in enumerate(self.robots):
        #     act = [0] * self.parts_num
        #     if robot.is_in_process:
        #         # robot have target task and is in process, keep target
        #         idx = robot.task_idx_allocated
        #         act[idx] = 1
        #     elif len(selectable_part_idx) > 0:
        #         idx_with_min_dist = selectable_part_idx[0]
        #         min_dist = 10000
        #         for idx in selectable_part_idx:
        #             dist = dist_cost["dist_robot_{}_to_part_{}".format(i + 1, idx + 1)]
        #             if dist <= min_dist:
        #                 min_dist = dist
        #                 idx_with_min_dist = idx
        #         selectable_part_idx.remove(idx_with_min_dist)
        #         act[idx_with_min_dist] += 1
        #     action_list.append(act)
        # print("allocator action:\t", allocator_action)
        # return allocator_action

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

    ####################################### motion planners ################################
    def sb3_ppo_planner(self, robot_list=[]):
        if len(robot_list) == 0:
            robot_list = self.robots
        ######## get predicted action
        obs = self.robots_env._observation()
        predicted_action = self.robot_policy.predict(obs, deterministic=True)[0]
        # policy = self.robot_policy.policy
        # observation, vectorized_env = policy.obs_to_tensor(obs)
        # features = policy.extract_features(observation)
        # latent_pi = policy.mlp_extractor.forward(features)[0]
        # distribution = policy._get_action_dist_from_latent(latent_pi)
        # predicted_action = distribution.get_actions(deterministic=True).cpu().detach().numpy()[0]

        ######## extract action from prediction
        act_dim = self.robots_env.action_dim
        robots_planner_action_list = []
        for robot in robot_list:
            robot_idx = self.robots.index(robot)
            robot_action = np.array(predicted_action[robot_idx * act_dim:(robot_idx + 1) * act_dim])
            robot_action *= self.robots_env.action_scale
            robots_planner_action_list.append(robot_action)
        print("robots_planner_action_list", robots_planner_action_list)
        return robots_planner_action_list

        return

    def go_straight_planner(self, robot_list=[]):
        if len(robot_list) == 0:
            robot_list = self.robots
        robots_planner_action_list = []
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        for i, robot in enumerate(self.robots):
            # print(i, robot.getObservation_EE(), robot.goal, robot.goal - robot.getObservation_EE())
            mode_id = robot.mode.get_mode_id()
            if self.use_reset and not robot.is_success:
                if mode_id in [1, 2, 4, 5]:
                    commands_scale = 0.05
                else:
                    commands_scale = 0.05
            else:
                commands_scale = 0.05
            robot_planner_action = robot.calculStraightAction2Goal(robot.goal_pose, commands_scale=commands_scale)
            robots_planner_action_list.append(robot_planner_action)
        # print(np.linalg.norm(robot_planner_action[0]), np.linalg.norm(robot_planner_action[1]))
        # time.sleep(3)
        # print("go straight")
        # print(robots_planner_action_list)
        # time.sleep(0.2)
        # print("out")

        return robots_planner_action_list

    def __del__(self):
        p.disconnect()
