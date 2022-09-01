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
from Env_gr import Env_gr

largeValObservation = np.inf  ###############


class Env_asynchronous_gr(Env_gr):

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
                 maxSteps=800,
                 reward_type="delta_dist_with_sparse_reward",
                 obs_type="common_obs",
                 in_task=False,
                 use_plot=False
                 ):
        # print("Now enter Env_a_k __init__")
        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._freezeAction = freezeAction
        self._isEnableSelfCollision = isEnableSelfCollision
        self._showBallMarkers = showBallMarkers
        self._reward_type = reward_type
        self._obs_type = obs_type
        self._in_task = in_task
        self._use_plot = use_plot
        # self._observation = []
        # self._envStepCounter = 0
        self._renders = renders
        self._evaluate = evaluate
        self._maxSteps = maxSteps
        self.terminated = 0

        ############### load config  #################
        self._useInverseKinematics = env_config[
            'useInverseKinematics'] if useInverseKinematics is None else useInverseKinematics
        self._success_freeze = env_config['success_freeze']
        self._fill_triangle = env_config['fill_triangle']
        self._normalize_pose = env_config['normalize_pose']
        self.move_with_obj = False
        self._sequence_task = env_config['sequence_task'] or self._evaluate
        # if self._sequence_task:
        #     self._maxSteps = min(1500,self._maxSteps)

        self._partsBaseSize = env_config['partsBaseSize']
        self._partsBasePos = env_config['partsBasePos']
        self._partsBaseOrn = env_config['partsBaseOrn']
        self._partsBaseColor = env_config['partsBaseColor']

        self._beltBaseSize = env_config['beltBaseSize']
        self._beltBasePos = env_config['beltBasePos']
        self._beltBaseOrn = env_config['beltBaseOrn']
        self._beltBaseColor = env_config['beltBaseColor']

        self.delta_pos_weight = env_config['delta_pos_weight']
        self.delta_orn_weight = env_config['delta_orn_weight']
        self.coll_penalty_obj = env_config['coll_penalty_obj']
        self.coll_penalty_robot = env_config['coll_penalty_robot']
        self.reach_bonus_pos = env_config['reach_bonus_pos']
        self.reach_bonus_orn = env_config['reach_bonus_orn']
        self.joint_success_bonus = env_config['joint_success_bonus']
        self.delta_area_ratio_weight = env_config['delta_area_ratio_weight']

        self.action_scale = env_config['action_scale']
        self.reward_scale = env_config['reward_scale']

        # self.success_dist_threshold_pos = env_config['success_dist_threshold_pos']
        # self.success_dist_threshold_orn = env_config['success_dist_threshold_orn']
        self.success_dist_threshold = env_config['success_dist_threshold']
        self.safety_dist_threshold = 0.01 if self._in_task else env_config['safety_dist_threshold']

        self.minimum_triangle_area = env_config['minimum_triangle_area']

        self.show_ball_freq = env_config['show_ball_freq']

        self.observation_history_length = env_config['observation_history_length']
        ################################

        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        # timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")

        self.seed()
        self.setup()

        self._action_bound = 1
        if self._isDiscrete:
            self.action_dim = 7
        else:
            if self._useInverseKinematics:
                self.action_dim = 4
            else:
                self.action_dim = 6

        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

    def generate_inits_and_goals(self):
        pattern = "chain"
        # pattern = "cycle"
        # pattern = "cycle2"
        goal_sets = []
        init_sets = []
        init_y = [-0.4, 0.4]
        goal_y = [0.25, -0.25]
        for i, robot in enumerate(self.robots):
            # init
            init = robot.sample_EE_pose(robot.init_state)[:4]
            init[0] = np.random.random() - 0.5
            init[1] = init_y[i] + (np.random.random() / 5 - 0.1)
            init[2] = 0.55 + (np.random.random() / 5 - 0.1)
            init[3] = 0
            init_sets.append(init)

            # goals
            robot_goals = []
            if pattern == "chain":
                goal_num = np.random.randint(2, 5)
                for k in range(goal_num):
                    goal = robot.sample_EE_pose(robot.goal_state)[:4]
                    goal[3] = 0
                    robot_goals.append(goal)
            elif pattern == "cycle":
                goal_num = 2
                for k in range(goal_num):
                    goal = robot.sample_EE_pose(robot.goal_state)[:4]
                    goal[1] = goal_y[i] + (np.random.random() / 2 - 0.25)
                    goal[3] = 0
                    robot_goals.append(goal)
                robot_goals.append(init)
            elif pattern == "cycle2":
                goal_num = np.random.randint(1, 4)
                for k in range(goal_num):
                    goal = robot.sample_EE_pose(robot.goal_state)[:4]
                    goal[3] = 0
                    robot_goals.append(goal)
                robot_goals.append(init)
            goal_sets.append(robot_goals)

        return init_sets, goal_sets

    def reset(self):
        self.terminated = 0
        self.env_step_counter = 0
        self.obs_info = {}
        self.prev_obs_info = {}
        self.who_reaced_first = -1

        self.clear_ball_markers()

        while True:
            init_sets, goal_sets = self.generate_inits_and_goals()
            # print(init_sets, goal_sets)
            for i, robot in enumerate(self.robots):
                init_pose = init_sets[i]
                goal_pose = goal_sets[i][0]
                robot.reset(init_pos=init_pose, goal_pos=goal_pose, reset_with_obj=self.move_with_obj)
                robot.goal_set = goal_sets[i]
                robot.reach_count = 0
                robot.reached_goal = 0
                robot.goal_pose = goal_pose
                robot.init_pose = init_pose

            # check collision, regenerate init/goal poses if collides
            check = True
            if check:
                break
        colors = [np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])]
        for i, robot in enumerate(self.robots):
            robot.norm_dist = np.linalg.norm(robot.getObservation_EE() - robot.goal)
            for k in range(len(robot.goal_set) - 1):
                robot.norm_dist += np.linalg.norm(robot.goal_set[k + 1] - robot.goal_set[k])
            robot.accumulated_dist = 0

            for k in range(len(robot.goal_set)):
                if self._renders and not self._in_task:
                    color = colors[i]
                    target = draw_cube_body(robot.goal_set[k][:3], [0, 0, 0, 1], [0.03, 0.02, 0.005], color)
                    # target = draw_sphere_body(robot.goal_EE_pos[:3], 0.05, [0, 0, 1, 1])
                    self.ball_markers.append(target)

        self.observation_history = []
        obs = self._observation()
        self.prev_obs_info = self.obs_info

        # print("observation info",obs)

        if self._showBallMarkers:
            for i, robot in enumerate(self.robots):
                ball = draw_sphere_body(self.obs_info['achieved_goal_{}'.format(i + 1)][:3], 0.02, [0, 1, 0, 1])
                self.ball_markers.append(ball)

        # print(obs_info)
        return obs

    def get_states(self):

        self.obs_info = {}
        obs_list = []
        for i, robot in enumerate(self.robots):
            # print(i)

            js = robot.getObservation_JS()
            normalized_js = robot.normalize_JS(js)

            ee = robot.getObservation_EE()
            normalized_ee = self.normalize_cartesian_pose(ee)
            # goal = robot.goal_pose
            # normalized_goal = self.normalize_cartesian_pose(goal)
            init = robot.init_pose
            normalized_init = self.normalize_cartesian_pose(init)

            link_pos_list = []
            link_pos_list.append(np.array(robot.BasePos[:3]))
            for j in range(robot.robotEndEffectorIndex):
                state = p.getLinkState(robot.robotUid, j)
                link_pos = state[0][:3]
                link_pos_list.append(np.array(link_pos))



            # dist2goal = np.linalg.norm(ee - goal)
            dist2init = np.linalg.norm(ee - init)
            dist2goals = []
            for k in range(len(robot.goal_set)):
                kth_goal = robot.goal_set[k]
                dist2goal = np.linalg.norm(ee - kth_goal)
                dist2goals.append(dist2goal)

            if robot.is_reached == 1:

                if not robot.is_success:
                    robot.reach_count += 1
                if robot.reach_count >= len(robot.goal_set):
                    robot.is_success = True
                else:
                    # print("innnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
                    current_goal = robot.goal_set[robot.reach_count]
                    robot.resetGoalPose(goal_pos=current_goal)
                    robot.goal_pose = current_goal
                    # print(i, robot.goal)
                # print(i, robot.reach_count, robot.is_success, len(robot.goal_set))

            goal = robot.goal_pose
            normalized_goal = self.normalize_cartesian_pose(goal)
            target_pose = normalized_goal
            ##### dist to obstacles
            min_dist2obj = self.get_closest_dists(robot)

            base_obs = np.concatenate([normalized_js, normalized_ee, target_pose])

            robot_obs = np.concatenate([base_obs])

            obs_list.append(robot_obs)

            self.obs_info["observation_{}".format(i + 1)] = robot_obs
            self.obs_info["joint_states_{}".format(i + 1)] = js
            self.obs_info["achieved_goal_{}".format(i + 1)] = ee
            self.obs_info["desired_goal_{}".format(i + 1)] = goal
            self.obs_info["dist_to_goal_{}".format(i + 1)] = dist2goals[min(robot.reach_count, len(robot.goal_set) - 1)]
            self.obs_info["dist_to_goals_{}".format(i + 1)] = dist2goals
            self.obs_info["reach_count_{}".format(i + 1)] = robot.reach_count
            self.obs_info["dist_to_init_{}".format(i + 1)] = dist2init
            self.obs_info["min_dist_to_obstacle_{}".format(i + 1)] = min_dist2obj
            self.obs_info['link_pos_list_{}'.format(i+1)] = link_pos_list
            # self.obs_info["reached_goal_{}".format(i + 1)] = robot.reached_goal
            # self.obs_info["come_back_{}".format(i + 1)] = robot.come_back
            #### triangle obs info
            self.obs_info["total_overlap_working_area_{}".format(i + 1)] = 0
            self.obs_info["maximum_cutting_ratio_{}".format(i + 1)] = 0
            self.obs_info["robot_working_area_{}".format(i + 1)] = 0
            self.obs_info["triangle_features_{}".format(i + 1)] = 0

        # print(obs_info)
        for i, robot in enumerate(self.robots):
            #### link pos and distance between links from different robots
            link_pos_list = self.obs_info['link_pos_list_{}'.format(i+1)]
            robot_link_info = []
            for j, other_robot in enumerate(self.robots):
                if i == j:
                    continue
                link_pos_list_from_other_robot = self.obs_info['link_pos_list_{}'.format(j+1)]
                dist_link2link = np.zeros((len(link_pos_list),len(link_pos_list_from_other_robot)))
                for m in range(len(link_pos_list)):
                    for n in range(len(link_pos_list_from_other_robot)):
                        dist_link2link[m,n] = np.linalg.norm(link_pos_list[m] - link_pos_list_from_other_robot[n])

                self.obs_info['dist_link2_link_{}_{}'.format(i + 1,j + 1)] = dist_link2link

            links_obs = []
            for m in range(len(link_pos_list)):
                link_pos = link_pos_list[m]
                link_dist = []
                for j, other_robot in enumerate(self.robots):
                    if i == j:
                        continue
                    link_dist.append(self.obs_info['dist_link2_link_{}_{}'.format(i + 1,j + 1)][m,:])
                link_dist = np.concatenate(link_dist)
                links_obs += [link_pos, link_dist]
            links_obs = np.concatenate(links_obs)

            # obs_list[i] = np.concatenate([obs_list[i], links_obs])


        for i, robot in enumerate(self.robots):
            ##### triangle states
            base_xy = robot.BasePos[:2]
            ee_xy = robot.getObservation_EE()[:2]
            goal_xy = robot.goal_pose[:2]
            triangle_edges = tools.get_edge_length_of_triangle(base_xy, ee_xy, goal_xy)
            triangle_angles = tools.get_base_angle_of_triangle(base_xy, ee_xy, goal_xy)
            robot_triangle_area = tools.calcul_triangle_area(base_xy, ee_xy, goal_xy,
                                                             minimum_triangle_area=self.minimum_triangle_area)
            ## calcul total overlap area , max cutting ratio
            total_overlap_area = 0
            maximum_cutting_ratio = 0
            for other_robot in self.robots:
                if robot == other_robot:
                    continue
                overlap_area, cutting_ratio = self.calcul_robot_overlap_working_area_and_cutting_ratio(robot,
                                                                                                       other_robot)
                total_overlap_area += overlap_area
                maximum_cutting_ratio = max(maximum_cutting_ratio, cutting_ratio)
            #### triangle obs
            triangle_obs = np.concatenate(
                [triangle_edges, triangle_angles, [robot_triangle_area, total_overlap_area, maximum_cutting_ratio]])

            obs_list[i] = np.concatenate([obs_list[i], triangle_obs])

            self.obs_info["total_overlap_working_area_{}".format(i + 1)] = total_overlap_area
            self.obs_info["maximum_cutting_ratio_{}".format(i + 1)] = maximum_cutting_ratio
            self.obs_info["robot_working_area_{}".format(i + 1)] = robot_triangle_area
            self.obs_info["triangle_features_{}".format(i + 1)] = triangle_obs

        return obs_list

    def step(self, action, scale_action=True, use_reset=False):

        obs, reward, done, info = super(Env_asynchronous_gr, self).step(action, scale_action=scale_action,
                                                                        use_reset=use_reset)

        if done:
            episode_info = {}
            episode_info['num_steps/num_steps_in_a_episode_when_success'] = self._maxSteps if any(
                robot.is_failed for robot in self.robots) else self.env_step_counter
            for i, robot in enumerate(self.robots):
                episode_info['is_success/robot_{}'.format(i + 1)] = self.robots[
                    i].is_success if not self._sequence_task else self.robots[i].task_success_count
                episode_info['reward_accu/robot_{}'.format(i + 1)] = self.robots[
                    i].accumulated_reward
                episode_info['reward_ave/robot_{}'.format(i + 1)] = self.robots[
                                                                        i].accumulated_reward / self.env_step_counter
                episode_info['dist_to_goal_dist/robot_{}'.format(i + 1)] = self.obs_info[
                    'dist_to_goal_{}'.format(i + 1)]
                if robot.is_success:
                    episode_info['motion_efficiency/robot_{}'.format(i + 1)] = robot.accumulated_dist / robot.norm_dist

                episode_info['reach_num/robot_{}'.format(i + 1)] = robot.reach_count
                episode_info['reach_rate/robot_{}'.format(i + 1)] = robot.reach_count / len(robot.goal_set)

                if self.robots[i].is_collision:
                    coll_with_obj = True
                    for k in range(self.robots_num):
                        if (self.robots[i].is_collision and self.robots[k].is_collision):
                            if i != k:
                                coll_with_obj = False
                            if i < k:
                                episode_info['hit/mutual_{}_{}'.format(i + 1, k + 1)] = 1
                        else:
                            if i < k:
                                episode_info['hit/mutual_{}_{}'.format(i + 1, k + 1)] = 0
                    episode_info['hit/robot_{}'.format(i + 1)] = coll_with_obj
                else:
                    episode_info['hit/robot_{}'.format(i + 1)] = False
                    for k in range(self.robots_num):
                        if i < k:
                            episode_info['hit/mutual_{}_{}'.format(i + 1, k + 1)] = 0

            episode_info['is_success/all'] = all(
                robot.is_success for robot in self.robots) if not self._sequence_task else sum(
                robot.task_success_count for robot in self.robots) / self.robots_num
            episode_info['reward_ave/all'] = sum(
                robot.accumulated_reward for robot in self.robots) / self.env_step_counter
            episode_info['reward_accu/all'] = sum(
                robot.accumulated_reward for robot in self.robots)
            episode_info['dist_to_goal_dist/ave'] = sum(
                self.obs_info['dist_to_goal_{}'.format(i + 1)] for i in range(self.robots_num)) / self.robots_num
            episode_info['hit/all'] = any(robot.is_failed for robot in self.robots)
            episode_info['reach_num/ave'] = sum(
                episode_info['reach_num/robot_{}'.format(i + 1)] for i in
                range(self.robots_num)) / self.robots_num
            episode_info['reach_rate/ave'] = sum(
                episode_info['reach_rate/robot_{}'.format(i + 1)] for i in
                range(self.robots_num)) / self.robots_num

            if all(robot.is_success for robot in self.robots):
                episode_info['motion_efficiency/ave'] = sum(
                    episode_info['motion_efficiency/robot_{}'.format(i + 1)] for i in
                    range(self.robots_num)) / self.robots_num

            # episode_info = {'episode': info}
        else:
            episode_info = {'goal_reached': all(robot.is_success for robot in self.robots)}

        return obs, reward, done, episode_info

    def _reward(self):

        reward_list = []
        for (i, robot) in enumerate(self.robots):
            ##### calcul delta dist2goal
            # dist2goal = self.obs_info["dist_to_goal_{}".format(i + 1)]
            # prev_dist2goal = self.prev_obs_info["dist_to_goal_{}".format(i + 1)]
            # delta_dist2_goal = dist2goal - prev_dist2goal
            # ##### calcul delta dist2goal
            # dist2init = self.obs_info["dist_to_init_{}".format(i + 1)]
            # prev_dist2init = self.prev_obs_info["dist_to_init_{}".format(i + 1)]
            # delta_dist2_init = dist2init - prev_dist2init
            # ##### reach goal
            # reached_goal = self.obs_info["reached_goal_{}".format(i + 1)]
            # come_back = self.obs_info["come_back_{}".format(i + 1)]
            # prev_reached_goal = self.prev_obs_info["reached_goal_{}".format(i + 1)]
            # prev_come_back = self.prev_obs_info["come_back_{}".format(i + 1)]

            dist2goals = self.obs_info["dist_to_goals_{}".format(i + 1)]
            prev_dist2goals = self.prev_obs_info["dist_to_goals_{}".format(i + 1)]
            prev_target = min(self.prev_obs_info["reach_count_{}".format(i + 1)], len(robot.goal_set) - 1)
            curr_target = min(self.obs_info["reach_count_{}".format(i + 1)], len(robot.goal_set) - 1)
            delta_dist2goal = dist2goals[prev_target] - prev_dist2goals[prev_target]

            reach_bonus = 0

            is_reached = self.check_reach(robot, dist2goals[curr_target])

            if is_reached == 1:

                reach_bonus += self.reach_bonus_pos

            prev_cutting_ratio = self.prev_obs_info["maximum_cutting_ratio_{}".format(i + 1)]
            curr_cutting_ratio = self.obs_info["maximum_cutting_ratio_{}".format(i + 1)]
            delta_cutting_ratio = curr_cutting_ratio - prev_cutting_ratio if not dist2goals[
                                                                                     prev_target] <= self.success_dist_threshold else 0

            prev_overlapping_area = self.prev_obs_info["total_overlap_working_area_{}".format(i + 1)]
            prev_working_area = self.prev_obs_info["robot_working_area_{}".format(i + 1)]
            prev_overlapping_ratio = prev_overlapping_area / prev_working_area

            curr_overlapping_area = self.obs_info["total_overlap_working_area_{}".format(i + 1)]
            curr_working_area = self.obs_info["robot_working_area_{}".format(i + 1)]
            curr_overlapping_ratio = curr_overlapping_area / curr_working_area

            delta_overlapping_ratio = curr_overlapping_ratio - prev_overlapping_ratio if not dist2goals[
                                                                                                 prev_target] <= self.success_dist_threshold else 0

            ##### is collision
            collision_id = self.check_collision(robot)

            ##### reward calculation
            if collision_id > 0:
                if collision_id in [r.robotUid for r in self.robots] and collision_id != robot.robotUid:
                    robot_reward = - self.coll_penalty_robot
                else:
                    robot_reward = - self.coll_penalty_obj
            else:
                #### robot reached , them next time robot will not received any reward bonus
                robot_reward = - delta_dist2goal * self.delta_pos_weight + reach_bonus

                robot_reward -= delta_overlapping_ratio * self.delta_area_ratio_weight
                robot_reward -= delta_cutting_ratio * self.delta_area_ratio_weight

            ##### append robot reward to robots reward list
            robot.accumulated_reward += robot_reward
            reward_list.append(robot_reward)

        return reward_list

    def _termination(self):
        if self.env_step_counter >= self._maxSteps:
            # for robot in self.robots:
            # if not robot.is_success:
            #     robot.is_failed = True
            # if self._renders:
            # print("step counter:{} exceeds max step:{}".format(self.env_step_counter, self._maxSteps))
            return True
        if all(robot.is_success or robot.is_failed for robot in self.robots):
            # if all(robot.is_success for robot in self.robots) and self._renders:
            # print("GOAAAALLLLLL!!!!!!!!!!!")
            return True
        # stop early
        if any(robot.is_failed for robot in self.robots):
            if self._renders:
                for i, robot in enumerate(self.robots):
                    if robot.is_failed:
                        print("robot {} failed".format(i + 1))
            return True

        return False

    def go_straight_planning(self, robot_list=[]):
        if len(robot_list) == 0:
            robot_list = self.robots
        robots_planner_action_list = []
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        for i, robot in enumerate(self.robots):
            commands_scale = 0.1
            if not robot.reached_goal:
                target = robot.goal_pose
            else:
                target = robot.init_pose
            robot_planner_action = robot.calculStraightAction2Goal(target, commands_scale=commands_scale)
            robot_planner_action /= self.action_scale
            robots_planner_action_list.append(robot_planner_action)

        return np.concatenate(robots_planner_action_list)


if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
