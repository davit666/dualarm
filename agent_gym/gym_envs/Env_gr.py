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
    create_cylinder_object,
    draw_sphere_body,
    draw_cube_body,
    remove_markers,
    remove_obstacles
)
from bullet_env import tools
from bullet_env.plot import Robot_Triangle_Plot

largeValObservation = np.inf  ###############


class Env_gr(gym.GoalEnv):
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
        self._action_type = action_type
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
        if self._action_type in ["js", "js_control_ee_reward"]:
            self._useInverseKinematics = False
        self._stepback_if_collide = env_config['stepback_if_collide']
        self._keep_bonus_when_success = env_config['keep_bonus_when_success']
        self._fill_triangle = env_config['fill_triangle']
        self._normalize_pose = env_config['normalize_pose']
        self.move_with_obj = False if self._in_task else env_config['move_with_obj']
        self.fixed_obj_shape = env_config['fixed_obj_shape']
        self.obj_shape_type = env_config['obj_shape_type']
        self._sequence_task = env_config['sequence_task'] or self._evaluate
        if self._sequence_task:
            self._maxSteps = max(2000, self._maxSteps)

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
            self.action_dim = 6
        else:
            if self._useInverseKinematics:
                self.action_dim = 4
            else:
                self.action_dim = 6

        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

    def get_observation_space(self):
        observation_dim = len(self._observation())
        observation_high = np.array([largeValObservation] * observation_dim)
        observation_space = spaces.Box(-observation_high, observation_high)
        return observation_space

    def get_action_space(self):
        if self._isDiscrete:
            action_space = spaces.Discrete((self.action_dim * self.robots_num))
        else:
            action_high = np.array([self._action_bound] * (self.action_dim * self.robots_num))
            action_space = spaces.Box(-action_high, action_high)
        return action_space

    def setup(self):
        ######## bullet setup
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.8)

        ######## load environment
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"))
        # self._partsBaseSize[0] *= 1.5
        self._partsBaseUid = create_cube_object(self._partsBasePos, orientation=self._partsBaseOrn,
                                                halfExtents=self._partsBaseSize, color=self._partsBaseColor)
        # self._partsBaseUid = create_cylinder_object(self._partsBasePos,
        #                                             radius=self._partsBaseSize[1], height=self._partsBaseSize[2]*2,
        #                                             color=self._partsBaseColor)
        # self._beltBaseUid = create_cube_object(self._beltBasePos, orientation=self._beltBaseOrn, halfExtents=self._beltBaseSize, color=self._beltBaseColor)

        ######## load robots
        self.robots = []

        self._robot1 = Robot(robotName="abb", useInverseKinematics=self._useInverseKinematics)
        self.robots.append(self._robot1)
        self._robot2 = Robot(robotName="kawasaki", useInverseKinematics=self._useInverseKinematics)
        self.robots.append(self._robot2)

        ######## update , visualize workspace
        self.ball_markers = []
        self.global_workspace = np.array([[0., 0.], [0., 0.], [0.5, 0.5]])
        for i, robot in enumerate(self.robots):
            self.update_workspace(robot)

        xyz = (self.global_workspace[:, 0] + self.global_workspace[:, 1]) / 2.
        dxdydz = (self.global_workspace[:, 0] - self.global_workspace[:, 1]) / 2.
        self.global_workspace_cube = create_cube_object(xyz[:], halfExtents=dxdydz[:], color=[1, 0, 0, 0.1])
        self.ball_markers.append(self.global_workspace_cube)

        ######## load items
        self.parts = []
        if self.move_with_obj:
            for i in range(len(self.robots)):
                self._part1 = Part(useInverseKinematics=self._useInverseKinematics, type='b')
                self.parts.append(self._part1)
                if not self.fixed_obj_shape:
                    self._part2 = Part(useInverseKinematics=self._useInverseKinematics, type='b2')
                    self.parts.append(self._part2)
                    self._part3 = Part(useInverseKinematics=self._useInverseKinematics, type='b3')
                    self.parts.append(self._part3)
                    self._part4 = Part(useInverseKinematics=self._useInverseKinematics, type='b4')
                    self.parts.append(self._part4)
                    self._part5 = Part(useInverseKinematics=self._useInverseKinematics, type='b5')
                    self.parts.append(self._part5)
                    self._part6 = Part(useInverseKinematics=self._useInverseKinematics, type='b6')
                    self.parts.append(self._part6)
                    self._part7 = Part(useInverseKinematics=self._useInverseKinematics, type='c1')
                    self.parts.append(self._part7)
                    self._part8 = Part(useInverseKinematics=self._useInverseKinematics, type='s1')
                    self.parts.append(self._part8)
                    self._part9 = Part(useInverseKinematics=self._useInverseKinematics, type='t')
                    self.parts.append(self._part9)

            assert len(self.parts) <= 18, "no more space to place parts"

        ######## parameters
        self.robots_num = len(self.robots)
        self.parts_num = len(self.parts)
        self.env_step_counter = 0

        # # reset parts position
        # if self.move_with_obj:
        #     self.release_all_parts()

        p.stepSimulation()
        if self._use_plot:
            self.plot = self.create_plot()

        self.reset()
        return True

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.terminated = 0
        self.env_step_counter = 0
        self.obs_info = {}
        self.prev_obs_info = {}

        self.clear_ball_markers()

        while True:
            if self.move_with_obj:
                self.release_all_parts()
                self.mount_parts()
            for i, robot in enumerate(self.robots):
                if robot.is_success and np.random.random() < 0.5:
                    reset_init_pos = False
                else:
                    reset_init_pos = True
                robot.reset(reset_init_pos=reset_init_pos, reset_with_obj=self.move_with_obj)
                robot.task_success_count = 0
                # if self._evaluate:
                #     robot.goal_EE_pos[2] = 0.5
                #     robot.goal[2] = 0.5

            # check collision, regenerate init/goal poses if collides
            check = self.check_init_goal_poses()
            if check:
                for i, robot in enumerate(self.robots):
                    robot.reach_count = 0
                    robot.reached_goal = 0
                    robot.goal_set = [robot.goal]
                break

        for i, robot in enumerate(self.robots):
            robot.norm_dist = np.linalg.norm(robot.getObservation_EE() - robot.goal)
            robot.accumulated_dist = 0
            robot.accumulated_normalized_dist = 0
            robot.accumulated_step = 0

            if self._renders and not self._in_task:
                color = robot.item_picking.color if robot.item_picking is not None else [0, 0, 1, 1]
                target = draw_cube_body(robot.goal_EE_pos[:3], robot.goal_EE_pos[3:], [0.03, 0.02, 0.005], color)
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

    def step(self, action, scale_action=True, use_reset=False):
        self.env_step_counter += 1
        a = []
        if not self._freezeAction:
            ######## scale action if needed
            if scale_action:
                action *= self.action_scale

            ######## execute action robot by robot
            j = 0
            for i, robot in enumerate(self.robots):
                act = action[j:j + self.action_dim]
                a.append(act)
                # #### use heuristics action
                # if self._evaluate:
                #     if i == 2:
                #         act = robot.calculStraightAction2Goal(robot.goal)
                #         act = np.array(act) * 0.5
                if robot.is_success:
                    robot.applyAction(act, use_reset=False)
                else:
                    robot.applyAction(act, use_reset=use_reset)
                j += self.action_dim
        ######## update bullet env
        self._p.stepSimulation()
        if self._renders:
            time.sleep(self._timeStep * 2)
        ######## get updated obs, reward, done, info
        obs = self._observation()
        if self._showBallMarkers and (self.env_step_counter + 1) % self.show_ball_freq == 0:
            for i in range(self.robots_num):
                ball = draw_sphere_body(self.obs_info['achieved_goal_{}'.format(i + 1)][:3], 0.02, [0, 1, 0, 1])
                self.ball_markers.append(ball)

        reward_list = self._reward()
        reward = sum(reward_list) * self.reward_scale
        reward += self.joint_success_bonus * self.reward_scale * all(robot.is_success for robot in self.robots)

        last_step_js = self.prev_obs_info["joint_states_{}".format(2)]

        ##### step back if collision or out of bounding box
        if self._stepback_if_collide:
            if any([robot.is_collision for robot in self.robots]):
                # time.sleep(0.2)
                # print("!!!!!!!a")
                # print("prev ", self.prev_obs_info["joint_states_{}".format(2)])
                # print("curr ", self.obs_info["joint_states_{}".format(2)])

                for i, robot in enumerate(self.robots):
                    last_step_js = self.prev_obs_info["joint_states_{}".format(i + 1)]
                    last_step_ee = self.prev_obs_info["achieved_goal_{}".format(i + 1)]
                    robot.reset_by_EE_pose(last_step_ee)
                self._p.stepSimulation()
                obs = self._observation()
                # print("curr2 ", self.obs_info["joint_states_{}".format(2)])
                # print("!!!!!!!a")
                # time.sleep(1)
        else:
            for i, robot in enumerate(self.robots):
                if robot.is_collision or robot.is_out_of_bounding_box:
                    robot.is_failed = True

        ##### count accumulated value
        for i, robot in enumerate(self.robots):
            robot.accumulated_dist += np.linalg.norm(
                self.obs_info["achieved_goal_{}".format(i + 1)][:3] - self.prev_obs_info[
                                                                          "achieved_goal_{}".format(i + 1)][:3])
            robot.accumulated_normalized_dist += np.linalg.norm(
                self.obs_info["achieved_normalized_goal_{}".format(i + 1)][:3] - self.prev_obs_info[
                                                                                     "achieved_normalized_goal_{}".format(
                                                                                         i + 1)][:3])

            robot.accumulated_step += not (robot.is_success or robot.is_failed)
            robot.accumulated_collision += robot.is_collision
            robot.accumulated_out_of_bounding_box += robot.is_out_of_bounding_box

        self.terminated = self._termination()

        self.prev_obs_info = self.obs_info

        if self._sequence_task:
            obs = self.automately_generate_task()

        if self.terminated:
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
                episode_info['dist_to_goal_ee/robot_{}'.format(i + 1)] = self.obs_info[
                    'dist_to_goal_ee_{}'.format(i + 1)]
                episode_info['dist_to_goal_js/robot_{}'.format(i + 1)] = self.obs_info[
                    'dist_to_goal_js_{}'.format(i + 1)]
                if robot.is_success:
                    episode_info['motion_efficiency/robot_{}'.format(i + 1)] = robot.accumulated_dist / robot.norm_dist

                episode_info['reach_num/robot_{}'.format(i + 1)] = robot.reach_count
                episode_info['reach_rate/robot_{}'.format(i + 1)] = robot.reach_count / len(robot.goal_set)
                episode_info['is_collision_num/robot_{}'.format(i + 1)] = robot.is_collision
                episode_info['is_collision_rate/robot_{}'.format(i + 1)] = robot.is_collision / self._maxSteps
                episode_info['is_out_of_bounding_box_num/robot_{}'.format(i + 1)] = robot.is_out_of_bounding_box
                episode_info['is_out_of_bounding_box_rate/robot_{}'.format(
                    i + 1)] = robot.is_out_of_bounding_box / self._maxSteps

                if self.robots[i].is_failed:
                    coll_with_obj = True
                    for k in range(self.robots_num):
                        if (self.robots[i].is_failed and self.robots[k].is_failed):
                            if i != k:
                                coll_with_obj = False
                            if i < k:
                                episode_info['is_failed/mutual_{}_{}'.format(i + 1, k + 1)] = 1
                        else:
                            if i < k:
                                episode_info['failed/mutual_{}_{}'.format(i + 1, k + 1)] = 0
                    episode_info['is_failed/robot_{}'.format(i + 1)] = coll_with_obj
                else:
                    episode_info['is_failed/robot_{}'.format(i + 1)] = False
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
            episode_info['dist_to_goal_ee/ave'] = sum(
                self.obs_info['dist_to_goal_ee_{}'.format(i + 1)] for i in range(self.robots_num)) / self.robots_num
            episode_info['dist_to_goal_js/ave'] = sum(
                self.obs_info['dist_to_goal_js_{}'.format(i + 1)] for i in range(self.robots_num)) / self.robots_num

            episode_info['is_failed/all'] = any(robot.is_failed for robot in self.robots)
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
            episode_info['prediction_model_state_data'] = self.get_prediction_model_state_data()
        else:
            episode_info = {'goal_reached': all(robot.is_success for robot in self.robots)}
        # print("reward:\t", reward, reward_list)
        # print("obs:\t", obs)
        # time.sleep(30)
        return obs, reward, self.terminated, episode_info

    def _observation(self):
        obs_list = self.get_states()
        observation = np.concatenate([robot_obs for robot_obs in obs_list])

        if len(self.observation_history) == self.observation_history_length:
            self.observation_history.pop(0)
        self.observation_history.append(observation)
        # print("obs_history length:  ", len(self.observation_history))
        # print(self.observation_history[-1])

        if self._use_plot:
            tri_img = self.plot.output_img_matrix()
            if self._renders:
                self.plot.show_img_matrix()

        return observation

    def get_states(self):
        self.obs_info = {}
        obs_list = []
        for i, robot in enumerate(self.robots):
            # print(i)

            js = robot.getObservation_JS()
            # print(i,' ',js)
            normalized_js = robot.normalize_JS(js)

            ee = robot.getObservation_EE()
            normalized_ee = self.normalize_cartesian_pose(ee)
            goal_ee = robot.goal_pose
            normalized_goal_ee = self.normalize_cartesian_pose(goal_ee)
            goal_js = robot.goal_JS_pos.copy()
            normalized_goal_js = robot.normalize_JS(goal_js)

            init_ee = robot.init_pose
            normalized_init_ee = self.normalize_cartesian_pose(init_ee)
            init_js = robot.init_JS_pos
            normalized_init_js = robot.normalize_JS(init_js)

            link_pos_list = []
            link_pose_obs = []
            link_pos_list.append(np.array(robot.BasePos[:3]))
            for j in range(robot.robotEndEffectorIndex):
                state = p.getLinkState(robot.robotUid, j)
                link_pos = np.array(state[0][:3])
                link_orn = np.array(state[1])
                link_pos_list.append(np.array(link_pos))
                normalized_link_pos = self.normalize_cartesian_pose(link_pos)
                link_pose_obs.append(np.concatenate([normalized_link_pos, link_orn]))

            # dist2goal = np.linalg.norm(ee - goal)
            dist2init_ee = np.linalg.norm(ee - init_ee)
            dist2init_js = np.linalg.norm(js - init_js)

            dist2goal_ee = np.linalg.norm(ee - goal_ee)
            dist2goal_js = np.linalg.norm(js - goal_js)

            dist2goals_ee = [dist2goal_ee]
            dist2goals_js = [dist2goal_js]

            if self._sequence_task and robot.is_reached == 1:
                if not robot.is_success:
                    robot.reach_count += 1
                    robot.is_success = True

            is_success = robot.is_success

            ##### dist to obstacles
            min_dist2obj = self.get_closest_dists(robot)

            base_obs = np.concatenate([normalized_js, normalized_ee, normalized_goal_ee])

            if self._action_type == "js":
                base_obs = np.concatenate([base_obs, normalized_goal_js])

            base_obs = np.concatenate([base_obs, [is_success]])

            self.obs_info["base_obs_{}".format(i + 1)] = base_obs
            # self.obs_info["is_success_{}".format(i + 1)] = [is_success]

            robot_obs = np.concatenate([base_obs])

            obs_list.append(robot_obs)

            self.obs_info["observation_{}".format(i + 1)] = robot_obs.copy()
            self.obs_info["joint_states_{}".format(i + 1)] = js.copy()
            self.obs_info["achieved_goal_{}".format(i + 1)] = ee.copy()
            self.obs_info["achieved_normalized_goal_{}".format(i + 1)] = normalized_ee.copy()
            self.obs_info["desired_goal_ee{}".format(i + 1)] = goal_ee.copy()
            self.obs_info["desired_goal_js{}".format(i + 1)] = goal_js.copy()
            self.obs_info["dist_to_goal_ee_{}".format(i + 1)] = dist2goals_ee[0]
            self.obs_info["dist_to_goal_js_{}".format(i + 1)] = dist2goals_js[0]
            self.obs_info["dist_to_goals_ee_{}".format(i + 1)] = dist2goals_ee
            self.obs_info["dist_to_goals_js_{}".format(i + 1)] = dist2goals_js
            self.obs_info["reach_count_{}".format(i + 1)] = robot.reach_count
            self.obs_info["dist_to_init_ee_{}".format(i + 1)] = dist2init_ee
            self.obs_info["dist_to_init_js_{}".format(i + 1)] = dist2init_js
            self.obs_info["min_dist_to_obstacle_{}".format(i + 1)] = min_dist2obj
            self.obs_info['link_pos_list_{}'.format(i + 1)] = link_pos_list
            self.obs_info['link_pose_obs_{}'.format(i + 1)] = np.concatenate(link_pose_obs)
            # print(len(self.obs_info['link_pose_obs_{}'.format(i + 1)]))
            #### triangle obs info
            self.obs_info["total_overlap_working_area_{}".format(i + 1)] = 0
            self.obs_info["maximum_cutting_ratio_{}".format(i + 1)] = 0
            self.obs_info["robot_working_area_{}".format(i + 1)] = 0
            self.obs_info["triangle_features_{}".format(i + 1)] = 0

        # print(obs_info)

        ######## link distance observation

        for i, robot in enumerate(self.robots):
            #### link pos and distance between links from different robots
            link_pos_list = self.obs_info['link_pos_list_{}'.format(i + 1)]
            robot_link_info = []
            for j, other_robot in enumerate(self.robots):
                if i == j:
                    continue
                link_pos_list_from_other_robot = self.obs_info['link_pos_list_{}'.format(j + 1)]
                dist_link2link = np.zeros((len(link_pos_list), len(link_pos_list_from_other_robot)))
                for m in range(len(link_pos_list)):
                    for n in range(len(link_pos_list_from_other_robot)):
                        dist_link2link[m, n] = np.linalg.norm(link_pos_list[m] - link_pos_list_from_other_robot[n])

                self.obs_info['dist_link2_link_{}_{}'.format(i + 1, j + 1)] = dist_link2link

            links_obs = []
            for m in range(len(link_pos_list)):
                link_pos = link_pos_list[m]
                link_dist = []
                for j, other_robot in enumerate(self.robots):
                    if i == j:
                        continue
                    link_dist.append(self.obs_info['dist_link2_link_{}_{}'.format(i + 1, j + 1)][m, :])
                link_dist = np.concatenate(link_dist)
                links_obs += [link_pos, link_dist]
            links_obs = np.concatenate(links_obs)

            self.obs_info["link_dist_obs_{}".format(i + 1)] = links_obs
            if self._obs_type == "common_obs_with_links_dist":
                obs_list[i] = np.concatenate([obs_list[i], links_obs])

        ######## object bounding box observation

        for i, robot in enumerate(self.robots):
            if robot.item_picking is None:
                obj_bounding_box = np.array([0, 0, 0])
                obj_center_pose = robot.getObservation_EE()
                obj_center_pose[2] -= 0.05
                obj_grasp_pose = robot.getObservation_EE()
                obj_grasp_pose[2] -= 0.05
                obj_bb_observation = np.concatenate([obj_bounding_box, obj_center_pose])

                obj_bounding_box = np.array([0.5, 0.1, 0.05])
                obj_center_pose = robot.getObservation_EE()
                obj_center_pose[2] -= 0.075
                obj_bb_observation = np.concatenate([obj_bounding_box, obj_center_pose])


            else:
                obj_bounding_box = robot.item_picking.part_size
                obj_center_pose = robot.item_picking.getCenterPose()
                obj_grasp_pose = robot.item_picking.getGraspPose()

                obj_bb_observation = np.concatenate([obj_bounding_box, obj_center_pose])

            self.obs_info["obj_bb_obs_{}".format(i + 1)] = obj_bb_observation
            self.obs_info["obj_bounding_box_{}".format(i + 1)] = obj_bounding_box
            self.obs_info["obj_center_pose_{}".format(i + 1)] = obj_center_pose
            self.obs_info["obj_grasp_pose_{}".format(i + 1)] = obj_grasp_pose
            if self._obs_type == "common_obs_with_obj_bb":
                obs_list[i] = np.concatenate([obs_list[i], obj_bb_observation])

        # ######## triangle observation
        # for i, robot in enumerate(self.robots):
        #     ##### triangle states
        #     base_xy = robot.BasePos[:2]
        #     ee_xy = robot.getObservation_EE()[:2]
        #     goal_xy = robot.goal_pose[:2]
        #     triangle_edges = tools.get_edge_length_of_triangle(base_xy, ee_xy, goal_xy)
        #     triangle_angles = tools.get_base_angle_of_triangle(base_xy, ee_xy, goal_xy)
        #     robot_triangle_area = tools.calcul_triangle_area(base_xy, ee_xy, goal_xy,
        #                                                      minimum_triangle_area=self.minimum_triangle_area)
        #     ## calcul total overlap area , max cutting ratio
        #     total_overlap_area = 0
        #     maximum_cutting_ratio = 0
        #     for other_robot in self.robots:
        #         if robot == other_robot:
        #             continue
        #         overlap_area, cutting_ratio = self.calcul_robot_overlap_working_area_and_cutting_ratio(robot,
        #                                                                                                other_robot)
        #         total_overlap_area += overlap_area
        #         maximum_cutting_ratio = max(maximum_cutting_ratio, cutting_ratio)
        #     #### triangle obs
        #     triangle_obs = np.concatenate(
        #         [triangle_edges, triangle_angles, [robot_triangle_area, total_overlap_area, maximum_cutting_ratio]])
        #
        #     obs_list[i] = np.concatenate([obs_list[i], triangle_obs])
        #
        #     self.obs_info["total_overlap_working_area_{}".format(i + 1)] = total_overlap_area
        #     self.obs_info["maximum_cutting_ratio_{}".format(i + 1)] = maximum_cutting_ratio
        #     self.obs_info["robot_working_area_{}".format(i + 1)] = robot_triangle_area
        #     self.obs_info["triangle_features_{}".format(i + 1)] = triangle_obs

        # print(obs_info)
        #### get triangle img
        if self._use_plot:
            self.plot.update_plot()
        return obs_list

    def _reward(self):

        reward_list = []
        for (i, robot) in enumerate(self.robots):
            ##### calcul delta dist2goal
            dist2goal_ee = self.obs_info["dist_to_goal_ee_{}".format(i + 1)]
            prev_dist2goal_ee = self.prev_obs_info["dist_to_goal_ee_{}".format(i + 1)]
            delta_dist2goal_ee = dist2goal_ee - prev_dist2goal_ee

            dist2goal_js = self.obs_info["dist_to_goal_js_{}".format(i + 1)]
            prev_dist2goal_js = self.prev_obs_info["dist_to_goal_js_{}".format(i + 1)]
            delta_dist2goal_js = dist2goal_js - prev_dist2goal_js

            ##### calcul delta obstacle field energy
            min_dist2obj = self.obs_info["min_dist_to_obstacle_{}".format(i + 1)]
            prev_min_dist2obj = self.prev_obs_info["min_dist_to_obstacle_{}".format(i + 1)]

            safety_threshold = robot.max_distance_from_others
            shape_factor = 8.
            ## field energy calculation
            obstacle_field = -2. / (1. + math.exp((float(min_dist2obj)) / safety_threshold) * shape_factor)
            prev_obstacle_field = -2. / (
                    1. + math.exp((float(prev_min_dist2obj)) / safety_threshold) * shape_factor)
            delta_obstacle_field = obstacle_field - prev_obstacle_field
            # print("dist\t",i+1,"\t",delta_dist2_goal)
            # print("obs\t",i+1,"\t",delta_obstacle_field)

            #### define dist2goal due to action type
            if self._action_type == "js":
                dist2goal = dist2goal_js
                delta_dist2goal = delta_dist2goal_js
            elif self._action_type in ["ee", "js_control_ee_reward"]:
                dist2goal = dist2goal_ee
                delta_dist2goal = delta_dist2goal_ee

            ##### reach goal
            is_reached = self.check_reach(robot, dist2goal)
            # print("robot{}".format(i + 1), "\tdist2goal:",dist2goal,"\tis_reached:",is_reached,"\tis_success:",robot.is_success)
            ##### is collision
            collision_id = self.check_collision(robot)

            ##### reward calculation
            if collision_id > 0:
                if collision_id in [r.robotUid for r in self.robots] and collision_id != robot.robotUid:
                    robot_reward = - self.coll_penalty_robot
                else:
                    robot_reward = - self.coll_penalty_obj

            elif robot.is_success:

                robot_reward = 0
                # keep bonus when reaching goal
                if self._keep_bonus_when_success:
                    if self._reward_type == "delta_dist_with_sparse_reward":
                        robot_reward = - delta_dist2goal * self.delta_pos_weight
                    elif self._reward_type == "delta_dist_field_with_sparse_reward":
                        robot_reward = - delta_dist2goal * self.delta_pos_weight + delta_obstacle_field * self.delta_pos_weight
                    elif self._reward_type == "negative_dist_with_sparse_reward":
                        robot_reward = - dist2goal_ee * self.delta_pos_weight
                    elif self._reward_type == "negative_dist_with_sparse_reward":
                        robot_reward = - dist2goal_ee * self.delta_pos_weight + obstacle_field * self.delta_pos_weight

                # remove bonus when reaching goal
                else:
                    if self._reward_type == "delta_dist_field_with_sparse_reward":
                        robot_reward += + delta_obstacle_field * self.delta_pos_weight
                #### make robot move as less as possible after robot is success using delta movement penalty
                # pass

            else:
                #### robot reached , them next time robot will not received any reward bonus
                if robot.is_reached:
                    robot.is_success = True
                #### different types of reward bonus:
                if self._reward_type == "delta_dist_with_sparse_reward":

                    robot_reward = - delta_dist2goal * self.delta_pos_weight + is_reached * self.reach_bonus_pos
                elif self._reward_type == "delta_dist_field_with_sparse_reward":

                    robot_reward = -delta_dist2goal * self.delta_pos_weight + delta_obstacle_field * self.delta_pos_weight + is_reached * self.reach_bonus_pos

                elif self._reward_type == "negative_dist_with_sparse_reward":

                    robot_reward = - dist2goal_ee * self.delta_pos_weight + is_reached * self.reach_bonus_pos

                elif self._reward_type == "negative_dist_field_with_sparse_reward":

                    robot_reward = - dist2goal_ee * self.delta_pos_weight + obstacle_field * self.delta_pos_weight + is_reached * self.reach_bonus_pos

                ########################### add delta overlapping area ration as inspiration
                elif self._reward_type == "delta_dist_&_overlap_area_ratio_with_sparse_reward":

                    prev_overlapping_area = self.prev_obs_info["total_overlap_working_area_{}".format(i + 1)]
                    prev_working_area = self.prev_obs_info["robot_working_area_{}".format(i + 1)]
                    prev_overlapping_ratio = prev_overlapping_area / prev_working_area

                    curr_overlapping_area = self.obs_info["total_overlap_working_area_{}".format(i + 1)]
                    curr_working_area = self.obs_info["robot_working_area_{}".format(i + 1)]
                    curr_overlapping_ratio = curr_overlapping_area / curr_working_area

                    delta_overlapping_area = curr_overlapping_area - prev_overlapping_area
                    delta_overlapping_area_ratio = delta_overlapping_area / curr_working_area
                    delta_overlapping_ratio = curr_overlapping_ratio - prev_overlapping_ratio

                    # print("dist\t", - delta_dist2_goal)
                    # print("area\t", - delta_overlapping_area_ratio)
                    robot_reward = - delta_dist2goal * self.delta_pos_weight - delta_overlapping_ratio * self.delta_area_ratio_weight + is_reached * self.reach_bonus_pos

                elif self._reward_type == "delta_dist_&_cutting_area_ratio_with_sparse_reward":

                    prev_cutting_ratio = self.prev_obs_info["maximum_cutting_ratio_{}".format(i + 1)]
                    curr_cutting_ratio = self.obs_info["maximum_cutting_ratio_{}".format(i + 1)]

                    delta_cutting_ratio = curr_cutting_ratio - prev_cutting_ratio

                    # print("dist\t", - delta_dist2_goal)
                    # print("cutting ratio\t", - delta_overlapping_area_ratio)
                    robot_reward = - delta_dist2goal * self.delta_pos_weight - delta_cutting_ratio * self.delta_area_ratio_weight + is_reached * self.reach_bonus_pos

                else:
                    print("unknown reward type is : ", self._reward_type, " ,reward is set to zero")
                    robot_reward = 0

            ######## set movement energy cost (delta movement)
            # curr_js = self.obs_info["joint_states_{}".format(i + 1)]
            # prev_js = self.prev_obs_info["joint_states_{}".format(i + 1)]
            # delta_js_movement = np.linalg.norm(curr_js - prev_js)
            # robot_reward -= delta_js_movement * self.delta_pos_weight * 0

            ##### append robot reward to robots reward list
            robot.accumulated_reward += robot_reward
            reward_list.append(robot_reward)

        return reward_list

    def _termination(self):
        if self.env_step_counter >= self._maxSteps:
            if self._renders:
                print("finisted at step counter:{} \texceeds max step:{}".format(self.env_step_counter, self._maxSteps))
            return True
        if all(robot.is_success for robot in self.robots):
            if self._renders:
                print("finisted at step counter:{} \tGOAAAALLLLLL!!!!!!!!!!!".format(self.env_step_counter))
            return True
        # stop early
        if any(robot.is_failed for robot in self.robots):
            if self._renders:
                print("finisted at step counter:{} \ttask failed".format(self.env_step_counter))
                for i, robot in enumerate(self.robots):
                    if robot.is_failed:
                        print("robot {} failed".format(i + 1))
                    if robot.is_collision:
                        print("robot {} collided".format(i + 1))
                    if robot.is_out_of_bounding_box:
                        print("robot {} is out of bounding box".format(i + 1))
            return True

        return False

    def update_workspace(self, robot=None):
        if robot is not None:
            for k in range(len(self.global_workspace)):
                self.global_workspace[k, 0] = min(self.global_workspace[k, 0], robot.workspace[k, 0])
                self.global_workspace[k, 1] = max(self.global_workspace[k, 1], robot.workspace[k, 1])
        return self.global_workspace

    def check_init_goal_poses(self):
        ######## check reachability for goal poses
        for robot in self.robots:
            robot.initialize_by_EE_pose(robot.goal_EE_pos)
            robot.goal_JS_pos = robot.getObservation_JS()
        goal_failed = False
        for robot in self.robots:
            coll, _ = robot.check_collision(collision_distance=0.1)
            goal_failed = goal_failed or coll
        if goal_failed:
            if self._renders:
                print("initialization failed in goal_pose reset, redo robot reset")
            return False
        ######## check collision for init poses
        for robot in self.robots:
            robot.initialize_by_EE_pose(robot.init_EE_pos)
            robot.init_JS_pos = robot.getObservation_JS()
        init_failed = False
        for robot in self.robots:
            coll, _ = robot.check_collision(collision_distance=0.1)
            init_failed = init_failed or coll
        if init_failed:
            if self._renders:
                print("initialization failed in init_pose reset, redo robot reset")
            return False
        return True

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

    def automately_generate_task(self):
        for i, robot in enumerate(self.robots):
            if robot.is_success:
                robot.task_success_count += 1
                robot.is_reached = 0
                robot.is_success = False
                robot.resetGoalPose()
                robot.goal_set.append(robot.goal)
                # if self._evaluate:
                #     robot.goal_EE_pos[2] = 0.5
                #     robot.goal[2] = 0.5
                if self._renders:
                    print("new goal for robot {} generated".format(i + 1))
                    color = robot.item_picking.color if robot.item_picking is not None else [0, 0, 1, 1]
                    target = draw_cube_body(robot.goal_EE_pos[:3], robot.goal_EE_pos[3:], [0.03, 0.02, 0.005],
                                            color)
                    self.ball_markers.append(target)

        return self._observation()

    def check_reach(self, robot, dist2goal):
        # print("reach:\t",dist2goal)
        success_dist_threshold = self.success_dist_threshold
        if self._action_type == "js":
            success_dist_threshold /= 2
        if self._in_task:
            moving_mode = robot.mode.get_mode()[1]

            if moving_mode in ["down", "up"]:
                success_dist_threshold = 0.01
        if dist2goal <= success_dist_threshold and not robot.is_reached:
            robot.is_reached = True
            if self._renders:
                print("robot {} reached !!!".format(1 + self.robots.index(robot)))
                color = robot.item_picking.color if robot.item_picking is not None else [0, 0, 1, 1]
                color = [1, 0.5, 0, 1]
                target1 = draw_cube_body(robot.goal_EE_pos[:3], robot.goal_EE_pos[3:], [0.031, 0.021, 0.0051], color)
                # target1 = draw_sphere_body(robot.goal_EE_pos[:3], 0.051, [0, 1, 0, 1])
                self.ball_markers.append(target1)
            return 1
        elif dist2goal >= success_dist_threshold and robot.is_reached:
            robot.is_reached = False
            if self._showBallMarkers:
                print("robot {} reached !!!".format(1 + self.robots.index(robot)))
                color = robot.item_picking.color if robot.item_picking is not None else [0, 0, 1, 1]
                target1 = draw_cube_body(robot.goal_EE_pos[:3], robot.goal_EE_pos[3:], [0.031, 0.021, 0.0051], color)
                self.ball_markers.append(target1)
            return -1
        else:
            return 0

    def check_collision(self, robot):
        ## update mode if in task execution, skip collision check if in picking/placing action
        if self._in_task:
            moving_mode = robot.mode.get_mode()[1]
            if moving_mode in ["down", "up"]:
                return -1

        ## check collision
        is_collision, coll_id = robot.check_collision(collision_distance=self.safety_dist_threshold)
        # if (is_collision and not robot.is_success) and coll_id not in self.ball_markers:
        if is_collision and coll_id not in self.ball_markers:
            # robot.is_failed = True
            robot.is_collision = True
            if self._renders:
                print("robot {} hit !!".format(1 + self.robots.index(robot)))
                ball = draw_sphere_body(self.obs_info['achieved_goal_{}'.format(1 + self.robots.index(robot))][:3],
                                        0.05,
                                        [1, 0, 0, 1])
                self.ball_markers.append(ball)
            return coll_id
        else:
            robot.is_collision = False

        ## check out of bounding box
        ee_pos = self.obs_info["achieved_goal_{}".format(self.robots.index(robot) + 1)]
        js_pos = self.obs_info["joint_states_{}".format(self.robots.index(robot) + 1)]
        out_of_bounding_box = False
        for i in range(len(js_pos)):
            if js_pos[i] > robot.ul[i] or js_pos[i] < robot.ll[i]:
                out_of_bounding_box = True
                break
        for j in range(3):
            if ee_pos[j] > self.global_workspace[j][1] or ee_pos[j] < self.global_workspace[j][0]:
                out_of_bounding_box = True
                break
        if out_of_bounding_box:
            # robot.is_failed = True
            robot.is_out_of_bounding_box = True
            if self._renders:
                print("robot {} get out of bounding box !!".format(1 + self.robots.index(robot)))
                ball = draw_sphere_body(self.obs_info['achieved_goal_{}'.format(1 + self.robots.index(robot))][:3],
                                        0.05,
                                        [0, 0, 0, 1])
                self.ball_markers.append(ball)
            return 99
        else:
            robot.is_out_of_bounding_box = False

        return -1

    def get_closest_dists(self, robot):
        min_dist = robot.max_distance_from_others
        robot.update_closest_points()
        for i, closest_points_to_other in enumerate(
                robot.closest_points_to_others):
            if len(closest_points_to_other) > 0:
                for point in closest_points_to_other:
                    if point[1] == robot.robotUid and (point[2] in robot.contactUids or point[2] in self.ball_markers):
                        pass
                    else:
                        min_dist = min(min_dist, point[8])
        return min_dist

    def calcul_robot_overlap_working_area_and_cutting_ratio(self, robot1, robot2):
        if self._in_task:
            return 0, 0
        if robot1 == robot2:
            return 0, 0
        base1_xy = robot1.BasePos[:2]
        ee1_xy = robot1.getObservation_EE()[:2]
        goal1_xy = robot1.goal[:2]
        triangle1 = np.array([base1_xy, ee1_xy, goal1_xy])

        base2_xy = robot2.BasePos[:2]
        ee2_xy = robot2.getObservation_EE()[:2]
        goal2_xy = robot2.goal[:2]
        triangle2 = np.array([base2_xy, ee2_xy, goal2_xy])

        overlap_convex_polygon_points = tools.get_overlap_points(tri1_points=triangle1, tri2_points=triangle2,
                                                                 plot=False)

        overlap_area = tools.get_overlap_area(tri1_points=triangle1, tri2_points=triangle2, type="D", plot=False,
                                              minimun_overlap_area=self.minimum_triangle_area,
                                              overlap_convex_polygon_points=overlap_convex_polygon_points)
        cutting_ratio = tools.get_cutting_ratio(tri1_points=triangle1, tri2_points=triangle2, plot=False,
                                                use_area=True,
                                                overlap_convex_polygon_points=overlap_convex_polygon_points)
        return overlap_area, cutting_ratio

    def create_plot(self):
        plot = Robot_Triangle_Plot(fill_triangle=self._fill_triangle)
        for i, robot in enumerate(self.robots):
            plot.create_robot_triangle(i)
        plot.create_plot()
        return plot

    # def apply_picking_item(self, robot, item_selected=None, calibrate_gripper=False):
    #     ######## check if robot is currently picking an item
    #     if robot.item_picking is not None:
    #         if self._renders:
    #             print("robot is currently picking item No.{} , can not pick anymore".format(
    #                 self.parts.index(robot.item_picking) + 1))
    #         return False
    #
    #     item_to_pick = None
    #     ee = robot.getObservation_EE()
    #     ######## check if selected item can be picked
    #     if item_selected is not None:
    #         grasp_pose = item_selected.getGraspPose()
    #         if np.linalg.norm(ee - grasp_pose) < self.success_dist_threshold:
    #             item_to_pick = item_selected
    #
    #     ######## check if there is any other item that can be picked
    #     else:
    #         for j, part in enumerate(self.parts):
    #             grasp_pose = part.getGraspPose()
    #             print(j, np.linalg.norm(ee - grasp_pose))
    #             if np.linalg.norm(ee - grasp_pose) < self.success_dist_threshold:
    #                 item_to_pick = part
    #                 break
    #     ######## no item can be picked
    #     if item_to_pick is None:
    #         if self._renders:
    #             print("No item can be picked by current robot {}".format(self.robots.index(robot)))
    #         return False
    #
    #     ######## item founded and calibrate robot gripper
    #     if calibrate_gripper:
    #         pick_action = robot.calculStraightAction2Goal(item_to_pick.getGraspPose())
    #         other_action = [0 for i in range(len(pick_action))]
    #         for i in range(self.robots_num):
    #             if i == self.robots.index(robot):
    #                 self.robots[i].applyAction(pick_action)
    #             else:
    #                 self.robots[i].applyAction(other_action)
    #     ######## apply picking action
    #     cid = robot.pickItem(item_to_pick)
    #     idx_to_pick = self.parts.index(item_to_pick)
    #     self.parts[idx_to_pick].picked(robot, cid)
    #     p.stepSimulation()
    #     return idx_to_pick + 1
    #
    # def apply_placing_item(self, robot):
    #     ######## check if robot have picking item
    #     if robot.item_picking is None:
    #         if self._renders:
    #             print("there is currently no item picked by robot {}".format(self.robots.index(robot)))
    #         return False
    #     else:
    #
    #         item_to_place = robot.item_picking
    #         ######## calibrate robot gripper and clear robot velocity to prepare placing
    #         place_action = robot.calculStraightAction2Goal(item_to_place.getGoalPose())
    #         other_action = [0 for i in range(len(place_action))]
    #         for i in range(self.robots_num):
    #             if i == self.robots.index(robot):
    #                 self.robots[i].applyAction(place_action)
    #
    #             else:
    #                 self.robots[i].applyAction(other_action)
    #         p.stepSimulation()
    #         for i in range(self.robots_num):
    #             self.robots[i].applyAction(other_action)
    #         p.stepSimulation()
    #         ######## apply placing action
    #         robot.placeItem(item_to_place)
    #
    #         idx_tp_place = self.parts.index(item_to_place)
    #         self.parts[idx_tp_place].placed(robot)
    #
    #         return True
    def go_straight_planning(self, robot_list=[]):

        if len(robot_list) == 0:
            robot_list = self.robots

        commands_scale = 0.1
        robots_planner_action_list = []

        for i, robot in enumerate(self.robots):

            if self._action_type in ["js", "js_control_ee_reward"]:
                js_target = robot.goal_JS_pos.copy()
                robot_planner_action = robot.calculStraightAction2Goal(js_target, type="js",
                                                                       commands_scale=commands_scale)
            elif self._action_type == "ee":
                target = robot.goal_pose
                robot_planner_action = robot.calculStraightAction2Goal(target, type="ee", commands_scale=commands_scale)
            else:
                robot_planner_action = np.array([0] * self.action_dim)
            robot_planner_action /= self.action_scale
            robots_planner_action_list.append(robot_planner_action)

        return np.concatenate(robots_planner_action_list)

    def release_all_parts(self):
        # demount parts being picked by robot
        for i, robot in enumerate(self.robots):
            part = robot.item_picking
            if part is not None:
                robot.placeItem(part)
                part.placed(robot)

        # set all parts to their initial position
        for i in range(self.robots_num):
            init_pose_x = np.array([0.2, 0.6, 1])
            init_pose_y = np.array([-0.6, 0, 0.6])

            for j in range(self.parts_num // self.robots_num):
                part = self.parts[i * self.parts_num // self.robots_num + j]
                x = (init_pose_x[j // 3] + 1.2) * (-1) ** i
                y = init_pose_y[j % 3]
                z = self._partsBaseSize[2] * 2
                part.resetInitPose([x, y, z], [0, 0, 0, 1])
        return 1

    def mount_parts(self):
        if self.move_with_obj:
            for i in range(self.robots_num):
                robot = self.robots[i]

                if self.obj_shape_type == "task":
                    if np.random.random() < 0.5:
                        part = self.parts[i * self.parts_num // self.robots_num + 0]
                        cid = robot.pickItem(part)
                        part.picked(robot, cid)

                else:
                    if self.fixed_obj_shape:
                        j = 0
                    elif self.obj_shape_type == "random":
                        j = random.randint(0, self.parts_num // 2 - 1)
                    elif self.obj_shape_type == "bar":
                        j = 1
                    elif self.obj_shape_type == "disk":
                        j = 6
                    elif self.obj_shape_type == "cube":
                        j = 4
                    elif self.obj_shape_type == "small":
                        j = 5
                    elif self.obj_shape_type == "sphere":
                        j = 7
                    else:
                        j = 0
                    part = self.parts[i * self.parts_num // self.robots_num + j]

                    ######## apply picking action
                    cid = robot.pickItem(part)
                    part.picked(robot, cid)

    def clear_ball_markers(self):
        remove_obstacles(self.ball_markers[1:])
        self.ball_markers = [self.global_workspace_cube]

    def get_prediction_model_state_data(self):

        if not self.terminated:
            return None

        robots_input = []
        robots_output = []

        for i, robot in enumerate(self.robots):
            ee = robot.getObservation_EE()
            normalized_ee = self.normalize_cartesian_pose(ee)

            init_js = robot.init_JS_pos
            normalized_init_js = robot.normalize_JS(init_js)

            init_ee = robot.init_pose
            normalized_init_ee = self.normalize_cartesian_pose(init_ee)
            goal_ee = robot.goal_pose
            normalized_goal_ee = self.normalize_cartesian_pose(goal_ee)

            dist_xyz = np.linalg.norm(init_ee[:3] - goal_ee[:3])
            normalized_dist_xyz = np.linalg.norm(normalized_init_ee[:3] - normalized_goal_ee[:3])
            dist_rz = np.linalg.norm(init_ee[3:] - goal_ee[3:])
            normalized_dist_rz = np.linalg.norm(normalized_init_ee[3:] - normalized_goal_ee[3:])

            is_picking = 0 if robot.item_picking is None else 1

            robot_input = np.concatenate(
                [init_js, init_ee, goal_ee, [dist_xyz], [dist_rz], [is_picking], normalized_init_js, normalized_init_ee,
                 normalized_goal_ee, [normalized_dist_xyz], [normalized_dist_rz]])

            robots_input.append(robot_input)

            suss = robot.is_success
            fail = robot.is_failed
            acc_traj_length = robot.accumulated_dist + np.linalg.norm(ee[:3] - goal_ee[:3])
            acc_traj_normalized_length = robot.accumulated_normalized_dist + np.linalg.norm(
                normalized_ee[:3] - normalized_goal_ee[:3])
            acc_step = robot.accumulated_step + 1 if not fail else self._maxSteps
            acc_normalized_step = acc_step / self._maxSteps

            robot_output = np.array(
                [suss, fail, acc_traj_length, acc_step, acc_traj_normalized_length, acc_normalized_step])
            robots_output.append(robot_output)

        global_suss = all([output[0] for output in robots_output])
        global_fail = any([output[1] for output in robots_output])
        global_acc_traj_length = max([output[2] for output in robots_output])
        global_acc_step = max([output[3] for output in robots_output])
        global_acc_traj_normalized_length = max([output[4] for output in robots_output])
        global_acc_normalized_step = max([output[5] for output in robots_output])
        global_output = np.array(
            [global_suss, global_fail, global_acc_traj_length, global_acc_step, global_acc_traj_normalized_length,
             global_acc_normalized_step])

        robots_input = np.concatenate([input for input in robots_input])
        robots_output = np.concatenate([output for output in robots_output])
        # print(len(robots_input))
        # print(len(robots_output))
        prediction_model_state_data = np.concatenate([robots_input,robots_output,global_output])
        # print(len(prediction_model_state_data))
        return prediction_model_state_data


if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
