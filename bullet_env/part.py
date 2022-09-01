import os

currentdir = os.path.dirname(os.path.abspath(__file__))
os.sys.path.insert(0, currentdir)

import pybullet as p
import numpy as np
import math


class Part:
    def __init__(self, urdfRootPath=os.path.join(currentdir, "urdf_model/obj"), useInverseKinematics=False, sucessDistThreshold = 0.1, type = 'b'):
        self.urdfRootPath = urdfRootPath
        self.useInverseKinematics = useInverseKinematics
        self.success_dist_threshold = sucessDistThreshold
        self.sucking_space = 0.075
        self.part_type = type

        self.init_pos = None
        self.init_orn = None
        self.goal_pos = None
        self.goal_orn = None

        if self.part_type == 'b2':
            self.part_size = np.array([0.5, 0.1, 0.05])
            self.color = [0, 1, 1, 1]
            self.partUid = p.loadURDF(os.path.join(self.urdfRootPath, "part_b2.urdf"))

        elif self.part_type == 'b3':
            self.part_size = np.array([0.3, 0.3, 0.1])
            self.color = [0.5, 0, 1, 1]
            self.partUid = p.loadURDF(os.path.join(self.urdfRootPath, "part_b3.urdf"))

        elif self.part_type == 'b4':
            self.part_size = np.array([0.2, 0.4, 0.02])
            self.color = [0, 1, 0, 1]
            self.partUid = p.loadURDF(os.path.join(self.urdfRootPath, "part_b4.urdf"))

        elif self.part_type == 'b5':
            self.part_size = np.array([0.2, 0.2, 0.2])
            self.color = [1, 1, 0, 1]
            self.partUid = p.loadURDF(os.path.join(self.urdfRootPath, "part_b5.urdf"))

        elif self.part_type == 'b6':
            self.part_size = np.array([0.1, 0.1, 0.05])
            self.color = [1, 0.5, 0, 1]
            self.partUid = p.loadURDF(os.path.join(self.urdfRootPath, "part_b6.urdf"))

        elif self.part_type == 'c1':
            self.part_size = np.array([0.4, 0.4, 0.05])
            self.color = [1, 0.7, 0.7, 1]
            self.partUid = p.loadURDF(os.path.join(self.urdfRootPath, "part_c1.urdf"))

        elif self.part_type == 's1':
            self.part_size = np.array([0.25, 0.25, 0.25])
            self.color = [1, 0.7, 0.7, 1]
            self.partUid = p.loadURDF(os.path.join(self.urdfRootPath, "part_s1.urdf"))

        elif self.part_type == 't':
            self.part_size = np.array([0.2, 0.2, 0.1])
            self.color = [0.4, 0.4, 0.4, 1]
            self.partUid = p.loadURDF(os.path.join(self.urdfRootPath, "test.urdf"))

        else:
            self.part_size = np.array([0.3, 0.2, 0.05])
            self.color = [0, 0, 1, 1]
            self.partUid = p.loadURDF(os.path.join(self.urdfRootPath, "part.urdf"))
        self.link_idx_to_pick = 0

        # self.picked_robot = None
        # self.constrain_cid = None
        #
        # self.is_success = False
        # self.is_in_process = False
        # self.is_picked = False
        # self.robot_idx_to_allocate = -1

        self.safety_gap_length = 0.0001 # 0.01 for touching free with ground when grasped
        self.reset()
    def reset(self):
        self.picked_robot = None
        self.constrain_cid = None
        self.is_success = False
        self.is_in_process = False
        self.is_picked = False
        self.robot_idx_to_allocate = -1
        # self.checkIsPlaced()
    def resetInitPose(self, pos, orn,reset_by_ee_pose=False):

        self.init_pos = pos
        self.init_orn = orn
        self.init_pos[2] += self.part_size[2] / 2.0
        if reset_by_ee_pose:
            self.init_pos[2] -= (self.part_size[2]/2.0 + self.sucking_space)

        p.resetBasePositionAndOrientation(self.partUid, self.init_pos, self.init_orn)
        state = p.getLinkState(self.partUid, 0)
        pos, orn = state[0], state[1]
        self.init_pos = np.array(pos)
        self.init_orn = np.array(orn)
        # self.checkIsPlaced()

    def resetGoalPose(self, pos, orn):
        self.goal_pos = pos
        self.goal_orn = orn
        self.goal_pos[2] += self.part_size[2] / 2.0 + self.sucking_space + self.safety_gap_length

    def checkIsPlaced(self):
        grasp_pos = self.getGraspPose()
        goal_pos = self.getGoalPose()
        dist2goal = np.linalg.norm(grasp_pos - goal_pos)
        if dist2goal < self.success_dist_threshold and self.picked_robot is None:
            self.is_success = True
        else:
            self.is_success = False
        return self.is_success
    def getBoundingBoxSize(self):
        boundingbox_size = self.part_size.copy()
        return boundingbox_size
    def getCenterPose(self, useInverseKinematics=None):
        if useInverseKinematics is None:
            useInverseKinematics = self.useInverseKinematics
        state = p.getBasePositionAndOrientation(self.partUid)
        pos, orn = list(state[0]), list(state[1])
        # print('c\t',pos)
        if useInverseKinematics == False:
            return np.array(pos + orn)
        else:
            rpy = p.getEulerFromQuaternion(orn)
            return np.array(pos[:] + [rpy[2]])
    def getBasePose(self, useInverseKinematics=None):
        if useInverseKinematics is None:
            useInverseKinematics = self.useInverseKinematics
        state = p.getLinkState(self.partUid, 0)
        pos, orn = list(state[0]), list(state[1])
        pos[-1] += self.sucking_gap_length
        if useInverseKinematics == False:
            return np.array(pos + orn)
        else:
            rpy = p.getEulerFromQuaternion(orn)
            return np.array(pos[:] + [rpy[2]])
    def getGraspPose(self, useInverseKinematics=None):
        if useInverseKinematics is None:
            useInverseKinematics = self.useInverseKinematics
        state = p.getLinkState(self.partUid, 0)
        pos, orn = state[0], state[1]
        if useInverseKinematics == False:
            grasp_pose = np.array(pos + orn)
        else:
            rpy = p.getEulerFromQuaternion(orn)
            grasp_pose = np.array(pos[:] + rpy[2:])
        grasp_pose[2] += self.safety_gap_length
        return np.array(grasp_pose)
    def getInitPose(self, useInverseKinematics=None):
        if useInverseKinematics is None:
            useInverseKinematics = self.useInverseKinematics
        if useInverseKinematics == False:

            return np.array(list(self.init_pos) + list(self.init_orn))
        else:
            y0 = p.getEulerFromQuaternion(self.init_orn)[-1]
            return np.array(self.init_pos[:] + [y0])

    def getGoalPose(self, useInverseKinematics=None):
        if useInverseKinematics is None:
            useInverseKinematics = self.useInverseKinematics
        if useInverseKinematics == False:

            return np.array(list(self.goal_pos) + list(self.goal_orn))
        else:
            y0 = p.getEulerFromQuaternion(self.goal_orn)[-1]
            return np.array(self.goal_pos[:] + [y0])

    def picked(self, robot, cid):
        assert self.picked_robot is None and self.constrain_cid is None
        self.picked_robot = robot
        self.constrain_cid = cid
        self.is_picked = True

    def placed(self, robot):
        assert self.picked_robot == robot and self.constrain_cid is not None
        self.picked_robot = None
        self.constrain_cid = None
        self.is_picked = False

    def delete(self):
        p.removeBody(self.partUid)
        return True
