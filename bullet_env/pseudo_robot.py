import os

currentdir = os.path.dirname(os.path.abspath(__file__))
os.sys.path.insert(0, currentdir)

import pybullet as p
import numpy as np
import math

ROBOT_NAME_LIST = ["abb", "kawasaki", "ur5"]


class PseudoPart():
    def __init__(self):
        self.success_dist_threshold = 0.1
        self.sucking_space = 0.075
        self.lift_height = 0.125
        self.part_size = np.array([0.5, 0.1, 0.05])

        self.init_pos = None
        self.init_orn = None
        self.goal_pos = None
        self.goal_orn = None

        self.link_idx_to_pick = 0
        self.pose = None
        self.reset()

    def reset(self):
        self.picked_robot = None
        self.constrain_cid = None
        self.is_success = False
        self.is_in_process = False
        self.is_picked = False
        self.robot_idx_to_allocate = -1
        # self.checkIsPlaced()

    def resetInitPose(self, pos, orn, reset_by_ee_pose=False):
        self.init_pos = pos
        self.init_orn = orn
        self.init_pos[2] += self.part_size[2] / 2.0 + self.sucking_space + self.lift_height + (np.random.random() - 0.5) / 50.

        self.init_pose = np.concatenate([self.init_pos, p.getEulerFromQuaternion(self.init_orn)[-1:]])

        self.pose = self.init_pose.copy()
        # self.checkIsPlaced()

    def resetGoalPose(self, pos, orn):
        self.goal_pos = pos
        self.goal_orn = orn
        self.goal_pos[2] += self.part_size[2] / 2.0 + self.sucking_space + self.lift_height + (np.random.random() - 0.5) / 50.

        self.goal_pose = np.concatenate([self.goal_pos, p.getEulerFromQuaternion(self.goal_orn)[-1:]])

    def load_part_data(self, p_data):
        init = p_data['init']
        goal = p_data['goal']

        self.init_pose = init.copy()
        self.goal_pose = goal.copy()
        self.pose = self.init_pose.copy()

        return True

    def getPose(self):
        assert self.pose is not None
        pose =np.array(self.pose.copy())
        return pose
    def getGoalPose(self):
        assert self.goal_pose is not None
        pose = np.array(self.goal_pose.copy())
        return pose
    def getInitPose(self):
        assert self.init_pose is not None
        pose = np.array(self.init_pose.copy())
        return pose

    # def picked(self, robot):
    #     assert self.picked_robot is None
    #     self.picked_robot = robot
    #     self.is_picked = True
    #
    # def placed(self, robot):
    #     assert self.picked_robot == robot
    #     self.picked_robot = None
    #     self.is_picked = False

    def pickedAndPlaced(self):
        self.is_success = True
        self.pose = self.getGoalPose()


ROBOT_NAME_LIST = ["abb", "kawasaki", "ur5"]


class PseudoRobot:
    def __init__(self, robotName="abb", urdfRootPath=os.path.join(currentdir, "urdf_model")):
        self.urdfRootPath = urdfRootPath
        # self.timeStep = timeStep
        self.useInverseKinematics = True

        self.robot_name = robotName
        assert self.robot_name in ROBOT_NAME_LIST

        self.robot_config = None
        if self.robot_name == "abb":
            from robot_config.abb_config import ABB
            self.robot_config = ABB()
        elif self.robot_name == "kawasaki":
            from robot_config.kawasaki_config import KAWASAKI
            self.robot_config = KAWASAKI()

        # print(self.robot_config)
        assert self.robot_config is not None

        self.path = self.urdfRootPath + self.robot_config.path
        self.BasePos = self.robot_config.BasePos
        self.BaseOrn = p.getQuaternionFromEuler(self.robot_config.BaseOrn)

        # self.maxVelocity = self.robot_config.maxVelocity
        # self.maxForce = self.robot_config.maxForce

        # self.actionDimension = self.robot_config.actionDimension
        # self.robotEndEffectorIndex = self.robot_config.robotEndEffectorIndex
        # self.robotGripperIndex = self.robot_config.robotGripperIndex

        self.init_jointStates = self.robot_config.init_jointStates
        # self.ul = self.robot_config.upper_limits
        # self.ll = self.robot_config.lowwer_limits
        # self.jr = self.robot_config.joint_ranges
        # self.rp = self.robot_config.rest_poses
        # self.jd = self.robot_config.joint_dumpings
        # posx = sum(self.robot_config.init_pos1_x) / 2
        # posy = sum(self.robot_config.init_pos1_y) / 2
        # posz = sum(self.robot_config.init_pos1_z) / 2
        self.workspace = np.array(
            [self.robot_config.init_pos1_x, self.robot_config.init_pos1_y, self.robot_config.init_pos1_z])

        self.default_goal_pose = np.array(self.robot_config.default_ee_pose)

        self.default_commands_scale = 0.1

        # self.contactUids = []

        self.init_state = 1
        self.goal_state = 2
        self.item_picking = None
        self.constrain_cid = None

        # self.initialize_random_pose()############

        self.setup()

    def setup(self):
        # print(self.urdfRootPath)
        # print(self.path)

        self.resetInitPose(default_pose=True)

        self.sucking = False
        self.is_success = False
        self.is_failed = False
        self.is_reached = False
        self.is_collision = False
        self.accumulated_reward = 0
        self.picking_count = 0
        self.placing_count = 0

        self.current_pose = None
        self.is_done = False

        return True

    def reset(self, reset_init_pos=True, init_pos=None, goal_pos=None, default_goal_pose = True):

        if reset_init_pos:
            self.init_EE_pos = self.sample_EE_pose(self.init_state) if init_pos is None else init_pos
            self.initialize_by_EE_pose(self.init_EE_pos)

        self.resetGoalPose(goal_pos, default_pose=default_goal_pose)
        self.resetInitPose(default_pose=True)
        self.init_pose = self.goal_pose.copy()
        self.current_pose = self.init_pose.copy()
        # self.init_JS_pos = self.getObservation_JS()
        self.sucking = False
        self.is_success = False
        self.is_reached = False
        self.is_failed = False
        self.is_in_process = False
        self.is_collision = False
        self.is_out_of_bounding_box = False
        self.accumulated_reward = 0
        self.accumulated_collision = 0
        self.accumulated_out_of_bounding_box = 0
        self.picking_count = 0
        self.placing_count = 0
        self.task_idx_allocated = -1


        # js = self.getObservation_JS()
        # ee = self.getObservation_EE()

        return True
    def load_robot_data(self, r_data):
        init = r_data['init']
        goal = r_data['goal']

        self.init_EE_pos = np.concatenate((init[:3], p.getQuaternionFromEuler([0,0,init[-1]]))) if self.useInverseKinematics else init.copy()
        self.init_pose = init.copy() if self.useInverseKinematics else self.init_EE_pos.copy()

        self.goal_EE_pos = np.concatenate((goal[:3], p.getQuaternionFromEuler([0,0,goal[-1]]))) if self.useInverseKinematics else goal.copy()
        self.goal_pose = goal.copy() if self.useInverseKinematics else self.goal_EE_pos.copy()

        self.current_pose = self.init_pose.copy()

        return True
    def resetInitPose(self, init_pose=None, default_pose=True):
        if default_pose:
            self.init_EE_pos = self.default_goal_pose.copy()
            self.init_pose = np.concatenate((self.init_EE_pos[:3], p.getEulerFromQuaternion(self.init_EE_pos[3:])[
                                                                   -1:])) if self.useInverseKinematics else self.init_EE_pos
        elif init_pose is not None:
            self.init_pose = init_pose
            init_rpy = [0, 0, init_pose[-1]] if self.useInverseKinematics else init_pose[3:]
            self.init_EE_pos = np.concatenate((init_pose[:3], p.getQuaternionFromEuler(init_rpy)[:]))
        else:
            self.init_pose = self.getObservation_EE()
            init_rpy = [0, 0, self.init_pose[-1]] if self.useInverseKinematics else self.init_pose[3:]
            self.init_EE_pos = np.concatenate((self.init_pose[:3], p.getQuaternionFromEuler(init_rpy)[:]))
        return self.init_pose

    def resetGoalPose(self, goal_pose=None, default_pose=True):
        if default_pose:
            self.goal_EE_pos = self.default_goal_pose.copy()
            self.goal_EE_pos[0] += (np.random.random() - 0.5) / 10.
            self.goal_EE_pos[1] += (np.random.random() - 0.5) / 10.
            self.goal_EE_pos[2] += (np.random.random() - 0.5) / 50.
            self.goal = np.concatenate((self.goal_EE_pos[:3], p.getEulerFromQuaternion(self.goal_EE_pos[3:])[
                                                              -1:])) if self.useInverseKinematics else self.goal_EE_pos
        elif goal_pose is not None:
            self.goal = goal_pose
            goal_rpy = [0, 0, goal_pose[-1]] if self.useInverseKinematics else goal_pose[3:]
            self.goal_EE_pos = np.concatenate((goal_pose[:3], p.getQuaternionFromEuler(goal_rpy)[:]))
        else:
            self.goal_EE_pos = self.sample_EE_pose(self.goal_state)

            #################################################################################################
            # if self.item_picking is not None:########################################
            #     delta_h = max(self.item_picking.part_size[2] - 0.05, 0)
            #     self.goal_EE_pos[2] += delta_h
            self.goal = np.concatenate((self.goal_EE_pos[:3], p.getEulerFromQuaternion(self.goal_EE_pos[3:])[
                                                              -1:])) if self.useInverseKinematics else self.goal_EE_pos

        self.goal_pose = self.goal.copy()
        return self.goal

    def getObservationDimension(self):
        return 4

    # def getObservation_JS(self):
    #     return None

    def getObservation_EE(self, useInverseKinematics=None):
        curr_pose = self.current_pose.copy()
        return curr_pose

    # def normalize_JS(self, js):
    #     return None

    def getBase(self):
        base_pos = self.BasePos
        base_orn = self.BaseOrn

        base_pose = np.array([base_pos[0], base_pos[1], base_pos[2] - 0.01, base_orn[-1]])
        return base_pose

    #### apply action ####
    def getActionDimension(self):
        return 4

    def applyAction(self, target_pose):
        # print(self.robot_name)
        # print("before:", self.getObservation_EE())
        self.current_pose = target_pose.copy()
        # print("after:", self.getObservation_EE())
        return self.getObservation_EE()

    # def calculStraightAction2Goal(self, target_pose, type="ee", commands_scale=None):
    #     commands_scale = commands_scale if commands_scale is not None else self.default_commands_scale
    #     if type == "ee":
    #         curr_ee = self.getObservation_EE()
    #         if not self.useInverseKinematics:
    #             curr_ee = np.concatenate((curr_ee[:3], p.getEulerFromQuaternion(curr_ee[3:])))
    #         delta_ee = target_pose - curr_ee
    #         len_delta_ee_pos = np.linalg.norm(delta_ee[:3])
    #         len_delta_ee_orn = np.linalg.norm(delta_ee[3:])
    #         len_delta_ee_pos = np.linalg.norm(delta_ee)
    #         len_delta_ee_orn = np.linalg.norm(delta_ee)
    #         straight_action_ee = [0] * len(delta_ee)
    #         for i in range(len(delta_ee)):
    #             if abs(delta_ee[i]) < 0.00001:
    #                 delta_ee[i] = 0
    #             if i < 3:
    #                 scale_len = commands_scale * (abs(delta_ee[i]) / len_delta_ee_pos)
    #             else:
    #                 scale_len = commands_scale * (abs(delta_ee[i]) / len_delta_ee_orn)
    #             straight_action_ee[i] = min(abs(delta_ee[i]), scale_len) * 1 if delta_ee[i] >= 0 else -min(
    #                 abs(delta_ee[i]), scale_len)
    #             # straight_action_ee[i] /= commands_scale
    #         return np.array(straight_action_ee)
    #
    #     else:
    #         return None

    def sample_EE_pose(self, state):
        if state == 1 and np.random.random() < 0.5:
            state = 2

        if state == 1:
            posx = np.random.sample() * (self.robot_config.init_pos1_x[1] - self.robot_config.init_pos1_x[0]) + \
                   self.robot_config.init_pos1_x[0]
            posy = np.random.sample() * (self.robot_config.init_pos1_y[1] - self.robot_config.init_pos1_y[0]) + \
                   self.robot_config.init_pos1_y[0]
            posz = np.random.sample() * (self.robot_config.init_pos1_z[1] - self.robot_config.init_pos1_z[0]) + \
                   self.robot_config.init_pos1_z[0]
        elif state == 2:
            posx = np.random.sample() * (self.robot_config.init_pos2_x[1] - self.robot_config.init_pos2_x[0]) + \
                   self.robot_config.init_pos2_x[0]
            posy = np.random.sample() * (self.robot_config.init_pos2_y[1] - self.robot_config.init_pos2_y[0]) + \
                   self.robot_config.init_pos2_y[0]
            posz = np.random.sample() * (self.robot_config.init_pos2_z[1] - self.robot_config.init_pos2_z[0]) + \
                   self.robot_config.init_pos2_z[0]
        r0 = 0  # np.random.sample() * 0.2 - 0.1
        p0 = 0  # np.random.sample() * 0.2 - 0.1
        y0 = np.random.sample() * math.pi / 2  # - math.pi / 4

        pos = [posx, posy, posz]
        rpy = np.array([r0, p0, y0])
        orn = list(p.getQuaternionFromEuler(rpy))
        return np.array(pos + orn)

    def initialize_by_EE_pose(self, target_pos):
        EE_pos = target_pos[:3]
        if self.useInverseKinematics:
            EE_orn = p.getQuaternionFromEuler([0, 0, EE_pos[-1]])
        else:
            EE_orn = target_pos[3:]

        self.current_pose = np.concatenate((EE_pos[:3], p.getEulerFromQuaternion(EE_orn)[
                                                          -1:])) if self.useInverseKinematics else target_pos

        return True

    def pickItem(self, part=None):
        assert part is not None
        self.item_picking = part
        self.picking_count += 1
        return True

    def placeItem(self, part):
        if part is None:
            return False
        else:
            self.item_picking = None
            self.placing_count += 1
            return True

    # def check_collision(self, collision_distance=0.00):
    #     return None
