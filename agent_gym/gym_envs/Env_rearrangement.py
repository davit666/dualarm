import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0,'../../')

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

largeValObservation = np.inf  ###############


class Env_rearrangement(gym.GoalEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 env_config,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 useInverseKinematics=None,
                 renders=False,
                 showBallMarkers=False,
                 isDiscrete=False,
                 freezeAction=False,
                 maxSteps=500,
                 reward_type="delta_dist_with_sparse_reward",
                 obs_type="common_obs",
                 action_type="js"):
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

        # self._observation = []
        # self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0

        ############### load config  #################
        self._useInverseKinematics = env_config[
            'useInverseKinematics'] if useInverseKinematics is None else useInverseKinematics

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

        self.action_scale = env_config['action_scale']
        self.reward_scale = env_config['reward_scale']

        self.sucess_dist_threshold_pos = env_config['sucess_dist_threshold_pos']
        self.sucess_dist_threshold_orn = env_config['sucess_dist_threshold_orn']
        self.sucess_dist_threshold = env_config['sucess_dist_threshold']
        self.safety_dist_threshold = env_config['safety_dist_threshold']

        self.show_ball_freq = env_config['show_ball_freq']
        ######################################################################################################3
        self.use_gripper = False
        self.observe_velocity = False
        self.use_robot_dense_reward = False
        self.picking_bonus = 0.1
        self.wrong_picking_penalty = 0.1
        self.placement_bonus = 0.1
        self.wrong_placement_penalty = 0.1
        self.wrong_gripper_penalty = 0
        ################################

        ######## bullet initialization
        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        ######## define action dimension ########
        if (self._isDiscrete):
            action_dim = 7
        else:
            if self._useInverseKinematics:
                # ee control DOF 4
                action_dim = 4
            else:
                # js control DOF 6
                action_dim = 6
            if self.use_gripper:
                action_dim += 1
        self.action_dim = action_dim

        ######## setup environment
        self.seed()
        self.setup()

        ######## act space definition
        if (self._isDiscrete):
            self.action_space = spaces.Discrete((action_dim * self.robots_num))
        else:
            self._action_bound = 1
            action_high = np.array([self._action_bound] * (action_dim * self.robots_num))
            self.action_space = spaces.Box(-action_high, action_high)
        ######## obs space definition
        observation_dim = len(self._get_obs())
        observation_high = np.array([largeValObservation] * observation_dim)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    def setup(self):
        ######## bullet setup
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.8)

        ######## load environment
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"))
        self._partsBaseUid = create_cube_object(self._partsBasePos, orientation=self._partsBaseOrn,
                                                halfExtents=self._partsBaseSize, color=self._partsBaseColor)
        # self._beltBaseUid = create_cube_object(self._beltBasePos, orientation=self._beltBaseOrn, halfExtents=self._beltBaseSize, color=self._beltBaseColor)

        ######## load robots
        self.robots = []

        self._robot1 = Robot(robotName="abb", useInverseKinematics=self._useInverseKinematics)
        self.robots.append(self._robot1)
        self._robot2 = Robot(robotName="kawasaki", useInverseKinematics=self._useInverseKinematics)
        self.robots.append(self._robot2)

        ######## load items
        self.parts = []
        self.obj_path = os.path.join(currentdir, "urdf_model/obj")

        self._part1 = Part(useInverseKinematics=self._useInverseKinematics,color = 'b')
        self.parts.append(self._part1)
        self._part2 = Part(useInverseKinematics=self._useInverseKinematics,color = 'b2')
        self.parts.append(self._part2)
        ######## parameters
        self.robots_num = len(self.robots)
        self.parts_num = len(self.parts)
        self.env_step_counter = 0
        self.ball_markers = []
        p.stepSimulation()

        self.reset()
        return True

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        ######## reset paramenters
        self.terminated = 0
        self.env_step_counter = 0
        self.obs_info = {}
        self.prev_obs_info = {}
        self.clear_ball_markers()

        ######## reset robot poses
        count = 0
        while True:
            count += 1
            collision = False
            for i, robot in enumerate(self.robots):
                self.apply_placing_item(robot)
                robot.reset()
                # check generated init pos
                coll, _ = robot.check_collision(collision_distance=self.safety_dist_threshold)
                collision = collision or coll
            # regenerate if something goes wrong
            if not collision:
                break
            else:
                if self._renders:
                    print("collision found in initialization, regenerate init pos for robot {}".format(i + 1))
            if count > 10:
                break
        ######## reset parts poses and goal poses
        # poses = [[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        poses = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
        # poses = [[1.2, 1.2], [-1.2, -1.2], [1.2, -1.2], [-1.2, 1.2]]

        random.shuffle(poses)
        k = 0
        for part in self.parts:
            x = poses[k][0] * 0.375 + np.random.random() * 0.2 - 0.375
            y = poses[k][1] * 0.25 + np.random.random() * 0.2 - 0.25
            # x = poses[k][0] + np.random.random() * 1 - 0.5
            # y = poses[k][1] * 0.6 + np.random.random() * 0.6 - 0.3
            # x /= 2
            # y /= 2
            z = self._partsBaseSize[2] * 2
            rz = math.pi * np.random.random() / 2 - math.pi / 4
            pos = [x, y, z]
            orn = p.getQuaternionFromEuler([0, 0, rz])
            part.resetInitPose(pos, orn)
            k += 1

            x = poses[k][0] * 0.375 + np.random.random() * 0.75 - 0.375
            y = poses[k][1] * 0.25 + np.random.random() * 0.5 - 0.25
            # x = poses[k][0] + np.random.random() * 1 - 0.5
            # y = poses[k][1] * 0.6 + np.random.random() * 0.6 - 0.3
            # x /= 2
            # y /= 2
            z = self._partsBaseSize[2] * 2
            rz = math.pi * np.random.random() / 2 - math.pi / 4
            pos = [x, y, z]
            orn = p.getQuaternionFromEuler([0, 0, rz])
            part.resetGoalPose(pos, orn)
            k += 1

            part.checkIsPlaced()
        ######## visualize goal
        for i, part in enumerate(self.parts):
            if self._renders:
                goal_pos_orn = part.getGoalPose(useInverseKinematics=False)
                target = draw_cube_body(goal_pos_orn[:3], goal_pos_orn[3:], [0.03, 0.02, 0.0025], part.color)
                self.ball_markers.append(target)
        ######## get obs and update state history
        self.obs_info = {}
        obs = self._get_obs()
        self.prev_obs_info = self.obs_info
        # print("observasion info",obs)
        ######## visualize trajectory
        if self._showBallMarkers:
            for i, robot in enumerate(self.robots):
                ball = draw_sphere_body(self.obs_info['achieved_goal_{}'.format(i + 1)][:3], 0.05, [0, 1, 0, 1])
                self.ball_markers.append(ball)

        # print(obs_info)
        return obs

    def _get_obs(self):
        # self.obs_info = {}
        robots_obs_list = []
        parts_obs_list = []
        relations_obs_list = []

        ######## calculate observasion robot by robot
        for i, robot in enumerate(self.robots):
            # joint states
            js = robot.getObservation_JS()
            normalized_js = robot.normalize_JS(js)
            # end effector poses
            ee = robot.getObservation_EE()
            # dist to obstacles
            min_dist2obj = self.get_closest_dists(robot)
            # sucker
            sucking = robot.sucking
            grasping = 1 if robot.item_picking is not None else 0

            robot_state = np.concatenate([normalized_js, ee, [sucking, grasping]])

            # velocity
            if self.observe_velocity:
                if 'robot_action_{}'.format(i + 1) in self.obs_info:
                    vel = self.obs_info['robot_action_{}'.format(i + 1)]
                else:
                    vel = [0] * self.action_dim
                robot_state = np.concatenate([robot_state, vel])
            ######## select observasion type
            if self._obs_type == "common_obs":
                robot_obs = robot_state
            elif self._obs_type == "common_obs_with_prev_state":
                robot_state_curr = robot_state
                # calculate robot state in last step
                robot_state_prev = self.prev_obs_info["robot_state_{}".format(i + 1)] if "robot_state_{}".format(
                    i + 1) in self.prev_obs_info else robot_state_curr[:]
                # calculate current state and previous state
                robot_obs = np.concatenate([robot_state_curr, robot_state_prev])
                # update previous state
                self.obs_info["robot_state_{}".format(i + 1)] = robot_state_curr
                assert len(robot_state_prev) == len(robot_state_curr)
            else:
                print("obs type is : ", self._obs_type)
                robot_obs = []
            robots_obs_list.append(robot_obs)
            ######## update current state information related with robot observasion
            self.obs_info["robot_observasion_{}".format(i + 1)] = robot_obs
            self.obs_info["robot_joint_states_{}".format(i + 1)] = js
            self.obs_info["robot_achieved_goal_{}".format(i + 1)] = ee
            # self.obs_info["robot_sucking_{}".format(i + 1)] = sucking
            # self.obs_info["robot_grasping_{}".format(i + 1)] = grasping
            self.obs_info["robot_min_dist_to_obstacle_{}".format(i + 1)] = min_dist2obj
            self.obs_info[
                'robot_picking_item_{}'.format(i + 1)] = None if robot.item_picking is None else self.parts.index(
                robot.item_picking) + 1
        ######## calculate observasion part by part
        for j, part in enumerate(self.parts):
            # grasp pose
            grasp_pose = part.getGraspPose()
            # goal pose
            goal_pose = part.getGoalPose()
            # dist 2 goal
            dist2goal = np.linalg.norm(grasp_pose - goal_pose)

            is_success = part.checkIsPlaced() if self.use_gripper else part.is_success
            part_obs = np.concatenate([grasp_pose, goal_pose, [is_success]])
            parts_obs_list.append(part_obs)
            ######## update current state information related with part observasion
            self.obs_info["part_current_pose_{}".format(j + 1)] = grasp_pose
            self.obs_info["part_goal_pose_{}".format(j + 1)] = goal_pose
            self.obs_info["part_dist_to_goal_{}".format(j + 1)] = dist2goal
            self.obs_info["part_is_success_{}".format(j + 1)] = is_success
            self.obs_info[
                "part_picked_agent_{}".format(j + 1)] = None if part.picked_robot is None else self.robots.index(
                part.picked_robot) + 1
        ######## calculate observasion of robot-part relationships
        for j, part in enumerate(self.parts):
            for i, robot in enumerate(self.robots):
                ee_pos = self.obs_info["robot_achieved_goal_{}".format(i + 1)]
                grasp_pose = self.obs_info["part_current_pose_{}".format(j + 1)]
                goal_pose = self.obs_info["part_goal_pose_{}".format(j + 1)]

                delta_grasp_ee = grasp_pose - ee_pos
                delta_goal_ee = goal_pose - ee_pos

                relation_obs = np.concatenate([delta_goal_ee, delta_grasp_ee])
                relations_obs_list.append(relation_obs)
                ######## update current state information related with part observasion
                self.obs_info["delta_grasp_dist_robot_{}_part_{}".format(i + 1, j + 1)] = delta_grasp_ee
                self.obs_info["delta_goal_dist_robot_{}_part_{}".format(i + 1, j + 1)] = delta_goal_ee
        ######## concate observasion from different robots
        robots_obs = np.concatenate([robot_obs for robot_obs in robots_obs_list])
        parts_obs = np.concatenate([part_obs for part_obs in parts_obs_list])
        relations_obs = np.concatenate([relation_obs for relation_obs in relations_obs_list])
        obs = np.concatenate([robots_obs, parts_obs, relations_obs])
        # print(obs_info)
        return obs

    def actions_extraction(self, action, robot_num=None):
        ######## extract robots action from network output
        if robot_num is None:
            robot_num = self.robots_num
        assert len(action) == self.action_dim * robot_num
        k = 0

        robots_action_list = []
        for j in range(robot_num):
            robot_action = action[k:k + self.action_dim]
            for i in range(len(robot_action)):
                robot_action[i] *= self.action_scale
            if self.use_gripper:
                robot_action[-1] /= self.action_scale
                robot_action[-1] += 1
                robot_action[-1] /= 2

            robots_action_list.append(robot_action)
            k += self.action_dim
        return robots_action_list

    def apply_robot_action(self, robot, robot_action=None):
        ######### freeze robot when no action input
        if robot_action is None:
            robot_action = [0] * self.action_dim
        ######## extract robot action and gripper action
        if self.use_gripper:
            robot_command = robot_action[:-1]
            gripper_command = robot_action[-1]
        else:
            robot_command = robot_action
            gripper_command = None
        ######## apply robot command
        robot.applyAction(robot_command)
        ######## apply gripper action
        if gripper_command is None:
            return None
        else:
            gripper_response = None
            if gripper_command > np.random.random():
                # pick item when sucker turns on
                robot.sucking = True
                if robot.item_picking is None:
                    gripper_response = self.apply_picking_item(robot)
            else:
                # place item when sucker turns off
                robot.sucking = False
                if robot.item_picking is not None:
                    gripper_response = self.apply_placing_item(robot)
            # print("gripper:\t",gripper_response)
            return gripper_response

    def apply_picking_item(self, robot, item_selected=None):
        ######## check if robot is currently picking an item
        if robot.item_picking is not None:
            if self._renders:
                print("robot is currently picking item No.{} , can not pick anymore".format(
                    self.parts.index(robot.item_picking) + 1))
            return False
        else:
            item_to_pick = None
            ee = robot.getObservation_EE()

            ######## check if selected item can be picked
            if item_selected is not None:
                grasp_pose = item_selected.getGraspPose()
                if np.linalg.norm(ee - grasp_pose) < self.sucess_dist_threshold:
                    item_to_pick = item_selected

            ######## check if there is any other item that can be picked
            else:
                for i, part in enumerate(self.parts):
                    grasp_pose = part.getGraspPose()
                    if np.linalg.norm(ee - grasp_pose) < self.sucess_dist_threshold:
                        item_to_pick = part
                        break
            ######## no item can be picked
            if item_to_pick is None:
                if self._renders:
                    print("No item can be picked by current robot {}".format(self.robots.index(robot)))
                return False

            else:
                ######## item founded and calibrate robot gripper
                pick_action = robot.calculStraightAction2Goal(item_to_pick.getGraspPose())
                other_action = [0 for i in range(len(pick_action))]
                for j in range(self.robots_num):
                    if j == self.robots.index(robot):
                        self.robots[j].applyAction(pick_action)
                    else:
                        self.robots[j].applyAction(other_action)
                ######## apply picking action
                cid = robot.pickItem(item_to_pick)
                idx_to_pick = self.parts.index(item_to_pick)
                self.parts[idx_to_pick].picked(robot, cid)
                p.stepSimulation()
                return True

    def apply_placing_item(self, robot):
        ######## check if robot have picking item
        if robot.item_picking is None:
            if self._renders:
                print("there is currently no item picked by robot {}".format(self.robots.index(robot)))
            return False
        else:

            item_to_place = robot.item_picking
            ######## calibrate robot gripper and clear robot velocity to prepare placing
            place_action = robot.calculStraightAction2Goal(item_to_place.getGoalPose())
            other_action = [0 for i in range(len(place_action))]
            for j in range(self.robots_num):
                if j == self.robots.index(robot):
                    self.robots[j].applyAction(place_action)

                else:
                    self.robots[j].applyAction(other_action)
            p.stepSimulation()
            for j in range(self.robots_num):
                self.robots[j].applyAction(other_action)
            p.stepSimulation()
            ######## apply placing action
            robot.placeItem(item_to_place)

            idx_tp_place = self.parts.index(item_to_place)
            self.parts[idx_tp_place].placed(robot)

            return True
    def auto_apply_pick_and_place(self):
        for i, robot in enumerate(self.robots):
            for j, part in enumerate(self.parts):
                if robot.item_picking is None and part.picked_robot is None and not part.is_success:
                    if np.linalg.norm(self.obs_info["delta_grasp_dist_robot_{}_part_{}".format(i + 1,
                                                                                               j + 1)]) < self.sucess_dist_threshold:
                        self.apply_picking_item(robot, part)
                        break
            if robot.item_picking is not None:
                part = robot.item_picking
                j = self.parts.index(part)
                if np.linalg.norm(self.obs_info["delta_goal_dist_robot_{}_part_{}".format(i + 1,
                                                                                          j + 1)]) < self.sucess_dist_threshold:
                    self.apply_placing_item(robot)
                    part.is_success = True
        obs = self._get_obs()
        return obs
    def step(self, action):
        self.env_step_counter += 1
        self.obs_info = {}
        ######## extract network output into actions, apply actions and get gripper responses
        robots_action_list = self.actions_extraction(action)
        if not self._freezeAction:
            grippers_response_list = []
            for i, robot in enumerate(self.robots):
                robot_action = robots_action_list[i]
                gripper_response = self.apply_robot_action(robot, robot_action)
                grippers_response_list.append(gripper_response)

        self._p.stepSimulation()
        if self._renders:
            time.sleep(self._timeStep)
        ######## get observasion
        for i, robot in enumerate(self.robots):
            self.obs_info['robot_action_{}'.format(i + 1)] = robots_action_list[i]
            self.obs_info['robot_sucker_{}'.format(i + 1)] = robot.sucking
            self.obs_info['robot_gripper_response_{}'.format(i + 1)] = grippers_response_list[i]
        obs = self._get_obs()
        ######## automaticly execute pick and place when condition satisfies
        if not self.use_gripper:
            obs = self.auto_apply_pick_and_place()

        ######## visualize trajectory
        if self._showBallMarkers and (self.env_step_counter + 1) % self.show_ball_freq == 0:
            for i in range(self.robots_num):
                ball = draw_sphere_body(self.obs_info['achieved_goal_{}'.format(i + 1)][:3], 0.05, [0, 1, 0, 1])
                self.ball_markers.append(ball)
        ######## get reward
        reward = self._reward()
        # reward += self.joint_success_bonus * all(part.is_success for part in self.parts)
        reward += self.joint_success_bonus * all(part.is_success for part in self.parts)
        reward *= self.reward_scale
        ######## check termination
        self.terminated = self._termination()
        ######## update episode imformation data
        self.prev_obs_info = self.obs_info
        if self.terminated:
            episode_info = {}

            for i in range(self.robots_num):
                episode_info['reward_accumulated/robot_{}'.format(i + 1)] = self.obs_info[
                    "robot_accumulated_reward_{}".format(i + 1)]
                episode_info['picking_count/robot_{}'.format(i + 1)] = self.robots[i].picking_count
                episode_info['placing_count/robot_{}'.format(i + 1)] = self.robots[i].placing_count
                episode_info['is_failed/robot_{}'.format(i + 1)] = self.robots[i].is_failed
                episode_info['dist_to_part/robot_{}'.format(i + 1)] = min(
                    [np.linalg.norm(self.obs_info["delta_grasp_dist_robot_{}_part_{}".format(i + 1, j + 1)]) for j in
                     range(self.parts_num)])
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
            for j in range(self.parts_num):
                episode_info['reward_accumulated/part_{}'.format(j + 1)] = self.obs_info[
                    "part_accumulated_reward_{}".format(j + 1)]
                episode_info['dist_to_goal_dist/part_{}'.format(j + 1)] = self.obs_info[
                    "part_dist_to_goal_{}".format(j + 1)]
                episode_info['dist_to_goal_%/part_{}'.format(j + 1)] = self.obs_info[
                                                                           "part_dist_to_goal_{}".format(j + 1)] / (
                                                                           np.linalg.norm(
                                                                               self.parts[j].getInitPose() - self.parts[
                                                                                   j].getGoalPose()))
                episode_info['is_success/part_{}'.format(j + 1)] = self.obs_info["part_is_success_{}".format(j + 1)]
            episode_info['num_steps/num_steps_in_a_episode_when_success'] = self._maxSteps if any(
                robot.is_failed for robot in self.robots) else self.env_step_counter
            episode_info['is_success/all'] = all(part.is_success for part in self.parts)
            episode_info['is_success/count'] = sum(part.is_success for part in self.parts)

            episode_info['reward_accumulated/robot'] = sum(
                episode_info['reward_accumulated/robot_{}'.format(i + 1)] for i in range(self.robots_num))
            episode_info['reward_accumulated/part'] = sum(
                episode_info['reward_accumulated/part_{}'.format(j + 1)] for j in range(self.parts_num))
            episode_info['reward_accumulated/all'] = episode_info['reward_accumulated/part'] + episode_info[
                'reward_accumulated/robot']

            episode_info['dist_to_goal_dist/ave'] = sum(
                episode_info['dist_to_goal_dist/part_{}'.format(j + 1)] for j in range(self.parts_num)) / self.parts_num
            episode_info['dist_to_goal_%/ave'] = sum(
                episode_info['dist_to_goal_%/part_{}'.format(j + 1)] for j in range(self.parts_num)) / self.parts_num
            episode_info['hit/all'] = any(robot.is_collision for robot in self.robots)
            episode_info['is_failed/all'] = any(robot.is_failed for robot in self.robots)
            episode_info['dist_to_part/all'] = sum(
                [episode_info['dist_to_part/robot_{}'.format(i + 1)] for i in range(self.robots_num)]) / self.robots_num
            # episode_info = {'episode': info}
        else:
            episode_info = {'goal_reached': sum(part.is_success for part in self.parts)}
        return obs, reward, self.terminated, episode_info

    def check_reach(self, dist2goal, robot=None):
        ######## check distance to goal
        if dist2goal <= self.sucess_dist_threshold and not robot.is_success:
            if robot is not None:
                robot.is_success = True
                ######## visualization
                if self._renders:
                    print("robot {} reached !!!".format(1 + self.robots.index(robot)))
                    target1 = draw_cube_body(robot.goal_EE_pos[:3], robot.goal_EE_pos[3:], [0.15, 0.075, 0.011],
                                             [1, 0, 0, 1])
                    self.ball_markers.append(target1)
            return 1
        else:
            return 0

    def check_collision(self, robot):
        ######## get robot collision imformation
        is_collision, coll_id = robot.check_collision(collision_distance=self.safety_dist_threshold)
        ######## robot collides with something
        if (is_collision and not robot.is_success) and coll_id not in self.ball_markers:
            robot.is_collision = True
            if self._showBallMarkers:
                print("robot {} hit !!".format(1 + self.robots.index(robot)))
                ball = draw_sphere_body(
                    self.obs_info['robot_achieved_goal_{}'.format(1 + self.robots.index(robot))][:3],
                    0.05, [1, 0, 0, 1])
                self.ball_markers.append(ball)
            return coll_id
        else:
            return -1

    def get_closest_dists(self, robot):
        ######## get closest distance between selected robot and other robots/objs
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

    def _reward(self):
        #### sparse reward
        ## part based
        parts_reward_list = [0] * self.parts_num
        for j, part in enumerate(self.parts):
            # picked reward
            part_reward = 0
            if self.prev_obs_info["part_picked_agent_{}".format(j + 1)] is None and self.obs_info[
                "part_picked_agent_{}".format(j + 1)] is not None:
                if self.prev_obs_info["part_is_success_{}".format(j + 1)] is True:
                    part_reward -= self.wrong_picking_penalty
                else:
                    part_reward += self.picking_bonus
            # place reward
            elif self.prev_obs_info["part_picked_agent_{}".format(j + 1)] is not None and self.obs_info[
                "part_picked_agent_{}".format(j + 1)] is None:
                is_success = self.obs_info["part_is_success_{}".format(j + 1)]
                if is_success:
                    part_reward += self.placement_bonus
                else:
                    part_reward -= self.wrong_placement_penalty
            else:
                pass
            # others ?
            # update
            parts_reward_list[j] = part_reward
        ## robot based
        robots_reward_list = [0] * self.robots_num
        for i, robot in enumerate(self.robots):
            robot_reward = 0
            # collision check
            collision_id = self.check_collision(robot)
            if collision_id > 0:
                robot.is_failed = True
                if collision_id in [r.robotUid for r in self.robots] and collision_id != robot.robotUid:
                    robot_reward = - self.coll_penalty_robot
                else:
                    robot_reward = - self.coll_penalty_obj
            # gripper_response
            if self.use_gripper:
                if self.obs_info['robot_gripper_response_{}'.format(i + 1)] is False:
                    robot_reward -= self.wrong_gripper_penalty
                    # robot.is_failed = True
            # others ?
            # update
            robots_reward_list[i] = robot_reward

        #### dense reward
        ## part based
        for j, part in enumerate(self.parts):
            if self.obs_info["part_is_success_{}".format(j + 1)] is True:
                pass
            elif self.obs_info["part_picked_agent_{}".format(j + 1)] is not None:
                dist2goal = self.obs_info["part_dist_to_goal_{}".format(j + 1)]
                prev_dist2goal = self.prev_obs_info["part_dist_to_goal_{}".format(j + 1)]
                delta_dist2goal = dist2goal - prev_dist2goal
                parts_reward_list[j] += - delta_dist2goal * self.delta_pos_weight
            else:
                part2robots = []
                prev_part2robots = []
                for i, robot in enumerate(self.robots):
                    if self.obs_info['robot_picking_item_{}'.format(i + 1)] is None:
                        d = np.linalg.norm(self.obs_info["delta_grasp_dist_robot_{}_part_{}".format(i + 1, j + 1)])
                    else:
                        d = 10000
                    part2robots.append(d)
                    if self.prev_obs_info['robot_picking_item_{}'.format(i + 1)] is None:
                        prev_d = np.linalg.norm(
                            self.prev_obs_info["delta_grasp_dist_robot_{}_part_{}".format(i + 1, j + 1)])
                    else:
                        prev_d = 10000
                    prev_part2robots.append(prev_d)
                min_part2robots = min(part2robots)
                min_prev_part2robots = min(prev_part2robots)
                delta_dist2goal = min_part2robots - min_prev_part2robots
                parts_reward_list[j] += - delta_dist2goal * self.delta_pos_weight
        ## robot based
        if self.use_robot_dense_reward:
            for i, robot in enumerate(self.robots):
                if self.obs_info["robot_picking_item_{}".format(i + 1)] is not None:
                    j = self.obs_info["robot_picking_item_{}".format(i + 1)] - 1
                    dist2goal = self.obs_info["part_dist_to_goal_{}".format(j + 1)]
                    prev_dist2goal = self.prev_obs_info["part_dist_to_goal_{}".format(j + 1)]
                    delta_dist2goal = dist2goal - prev_dist2goal
                    robots_reward_list[i] += - delta_dist2goal * self.delta_pos_weight
                else:
                    robot2parts = []
                    prev_robot2parts = []
                    for j, part in enumerate(self.parts):
                        if self.obs_info["part_is_success_{}".format(j + 1)] is True:
                            d = 10000
                        elif self.obs_info["part_picked_agent_{}".format(j + 1)] is None:
                            d = np.linalg.norm(self.obs_info["delta_grasp_dist_robot_{}_part_{}".format(i + 1, j + 1)])
                        else:
                            d = 10000
                        robot2parts.append(d)
                        if self.prev_obs_info["part_is_success_{}".format(j + 1)] is True:
                            prev_d = 10000
                        elif self.prev_obs_info["part_picked_agent_{}".format(j + 1)] is None:
                            prev_d = np.linalg.norm(
                                self.prev_obs_info["delta_grasp_dist_robot_{}_part_{}".format(i + 1, j + 1)])
                        else:
                            prev_d = 10000
                        prev_robot2parts.append(prev_d)
                    min_robot2parts = min(robot2parts)
                    min_prev_robot2part = min(prev_robot2parts)
                    delta_dist2goal = min_robot2parts - min_prev_robot2part
                    robots_reward_list[i] += - delta_dist2goal * self.delta_pos_weight

        ######## update reward info
        for i, robot_reward in enumerate(robots_reward_list):
            self.obs_info["robot_reward_{}".format(i + 1)] = robot_reward
            self.obs_info["robot_accumulated_reward_{}".format(i + 1)] = self.prev_obs_info[
                                                                             "robot_accumulated_reward_{}".format(
                                                                                 i + 1)] + robot_reward if "robot_accumulated_reward_{}".format(
                i + 1) in self.prev_obs_info else robot_reward
        for j, part_reward in enumerate(parts_reward_list):
            self.obs_info["part_reward_{}".format(j + 1)] = part_reward
            self.obs_info["part_accumulated_reward_{}".format(j + 1)] = self.prev_obs_info[
                                                                            "part_accumulated_reward_{}".format(
                                                                                j + 1)] + part_reward if "part_accumulated_reward_{}".format(
                j + 1) in self.prev_obs_info else part_reward
        #### sum rewards
        robot_reward_sum = sum(robots_reward_list)
        part_reward_sum = sum(parts_reward_list)
        self.obs_info["robot_reward_sum"] = robot_reward_sum
        self.obs_info["robot_accumulated_reward_sum"] = self.prev_obs_info[
                                                            "robot_accumulated_reward_sum"] + robot_reward_sum if "robot_accumulated_reward_sum" in self.prev_obs_info else robot_reward_sum
        self.obs_info["part_reward_sum"] = part_reward_sum
        self.obs_info["part_accumulated_reward_sum"] = self.prev_obs_info[
                                                           "part_accumulated_reward_sum"] + part_reward_sum if "part_accumulated_reward_sum" in self.prev_obs_info else part_reward_sum
        # print([robot_reward_sum, part_reward_sum])
        reward = sum([robot_reward_sum, part_reward_sum])

        return reward

    def _termination(self):
        # max step reached
        if self.env_step_counter >= self._maxSteps:
            return True
        # task finished
        if all([part.is_success for part in self.parts]):
            print("GOAAAALLLLLL!!!!!!!!!!!")
            return True
        # stop when picked
        # if all(part.is_success or part.is_picked for part in self.parts):
        #     print("OKAY")
        #     return True
        # stop early because of collision
        if any([robot.is_failed for robot in self.robots]):
            return True

        return False

    def clear_ball_markers(self):
        ######## remove visual markers
        remove_obstacles(self.ball_markers)
        self.ball_markers = []

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step
