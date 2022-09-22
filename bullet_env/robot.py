import os

currentdir = os.path.dirname(os.path.abspath(__file__))
os.sys.path.insert(0, currentdir)

import pybullet as p
import numpy as np
import math
from p_utils import (
    get_self_link_pairs,
    create_cylinder_object,
)

ROBOT_NAME_LIST = ["abb", "kawasaki", "ur5"]


class Robot:
    def __init__(self, robotName="abb", urdfRootPath=os.path.join(currentdir, "urdf_model"), useInverseKinematics=False,
                 timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep

        self.useInverseKinematics = useInverseKinematics
        self.useSimulation = 1

        self.robot_name = robotName
        assert self.robot_name in ROBOT_NAME_LIST

        self.robot_config = None
        if self.robot_name == "abb":
            from robot_config.abb_config import ABB
            self.robot_config = ABB()
            # print(1111111111)
        elif self.robot_name == "kawasaki":
            from robot_config.kawasaki_config import KAWASAKI
            self.robot_config = KAWASAKI()
        # elif self.robot_name == "ur5":
        #     from robot_config.ur5_config import Ur5
        #     self.robot_config = Ur5()

        print(self.robot_config)
        assert self.robot_config is not None

        self.path = self.urdfRootPath + self.robot_config.path
        self.BasePos = self.robot_config.BasePos
        self.BaseOrn = p.getQuaternionFromEuler(self.robot_config.BaseOrn)

        self.maxVelocity = self.robot_config.maxVelocity
        self.maxForce = self.robot_config.maxForce

        self.actionDimension = self.robot_config.actionDimension
        self.robotEndEffectorIndex = self.robot_config.robotEndEffectorIndex
        self.robotGripperIndex = self.robot_config.robotGripperIndex

        self.init_jointStates = self.robot_config.init_jointStates
        self.ul = self.robot_config.upper_limits
        self.ll = self.robot_config.lowwer_limits
        self.jr = self.robot_config.joint_ranges
        self.rp = self.robot_config.rest_poses
        self.jd = self.robot_config.joint_dumpings
        # posx = sum(self.robot_config.init_pos1_x) / 2
        # posy = sum(self.robot_config.init_pos1_y) / 2
        # posz = sum(self.robot_config.init_pos1_z) / 2
        self.workspace = np.array(
            [self.robot_config.init_pos1_x, self.robot_config.init_pos1_y, self.robot_config.init_pos1_z])

        self.default_goal_pose = np.array(self.robot_config.default_ee_pose)

        self.default_commands_scale = 0.1

        self.contactUids = []

        self.initialize_random_pose()

        self.accumulated_reward = 0
        self.is_success = False
        self.is_failed = False

        self.sucking = None
        self.item_picking = None
        self.constrain_cid = None

        self.init_state = 1
        self.goal_state = 2

        self.setup()

    #### steup ####
    def setup(self):
        print(self.urdfRootPath)
        print(self.path)
        self.GROUP_INDEX = {
            'arm': [0, 1, 2, 3, 4, 5],
            'gripper': None
        }
        self.robotUid = p.loadURDF(self.path, basePosition=self.BasePos, baseOrientation=self.BaseOrn, useFixedBase=1,
                                   flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.contactUids.append(self.robotUid)
        self.max_distance_from_others = 0.2
        self.link_pairs = get_self_link_pairs(
            self.robotUid,
            self.GROUP_INDEX['arm'])
        self.numJoints = p.getNumJoints(self.robotUid)
        # print("self.numJoints=",self.numJoints)
        self.jointsInfos = []
        self.jointsStates = []
        for jointIndex in range(self.numJoints):

            jointInfo = p.getJointInfo(self.robotUid, jointIndex)
            jointState = p.getJointState(self.robotUid, jointIndex)
            qIndex = jointInfo[3]
            if qIndex > 0:
                self.jointsInfos.append(jointInfo)
                self.jointsStates.append(jointState[0])
            # print("jointIndex=",jointIndex)
            p.resetJointState(self.robotUid, jointIndex, self.init_jointStates[jointIndex])

        state = p.getLinkState(self.robotUid, self.robotEndEffectorIndex)

        self.endEffectorPos = state[0][:]
        # print("setup finished, EEposition:", self.endEffectorPos)

        self.baseCylinderUid = create_cylinder_object([self.BasePos[0], self.BasePos[1], (self.BasePos[2] - 0.01) / 2],
                                                      radius=0.35, height=(self.BasePos[2] - 0.01),
                                                      color=[0.5, 0.5, 0.5, 1])
        self.contactUids.append(self.baseCylinderUid)

        self.initialize_random_pose()

        self.sucking = False
        self.is_success = False
        self.is_failed = False
        self.is_reached = False
        self.is_collision = False
        self.accumulated_reward = 0
        self.picking_count = 0
        self.placing_count = 0

        return True

    #### reset robot ####
    def reset(self, reset_init_pos=True, init_pos=None, goal_pos=None, reset_with_obj=False):

        if reset_init_pos:
            self.init_EE_pos = self.sample_EE_pose(self.init_state) if init_pos is None else init_pos
            self.initialize_by_EE_pose(self.init_EE_pos)
            self.init_JS_pos = np.array(self.init_jointStates)
        self.resetGoalPose(goal_pos)
        self.resetInitPose(default_pose=False)
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

    def resetInitPose(self, init_pose=None, default_pose=True):
        if default_pose:
            self.init_EE_pos = self.default_goal_pose.copy()
            self.init_pose = np.concatenate((self.init_EE_pos[:3], p.getEulerFromQuaternion(self.goal_EE_pos[3:])[
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

    def resetGoalPose(self, goal_pose=None, default_pose=False):
        if default_pose:
            self.goal_EE_pos = self.default_goal_pose.copy()
            self.goal = np.concatenate((self.goal_EE_pos[:3], p.getEulerFromQuaternion(self.goal_EE_pos[3:])[
                                                              -1:])) if self.useInverseKinematics else self.goal_EE_pos
        elif goal_pose is not None:
            self.goal = goal_pose
            goal_rpy = [0, 0, goal_pose[-1]] if self.useInverseKinematics else goal_pose[3:]
            self.goal_EE_pos = np.concatenate((goal_pose[:3], p.getQuaternionFromEuler(goal_rpy)[:]))
        else:
            self.goal_EE_pos = self.sample_EE_pose(self.goal_state)
            if self.item_picking is not None:
                delta_h = max(self.item_picking.part_size[2] - 0.05, 0)
                self.goal_EE_pos[2] += delta_h
            self.goal = np.concatenate((self.goal_EE_pos[:3], p.getEulerFromQuaternion(self.goal_EE_pos[3:])[
                                                              -1:])) if self.useInverseKinematics else self.goal_EE_pos

        self.goal_pose = self.goal.copy()
        return self.goal

    #### get observation ####
    def getObservationDimension(self):
        return len(self.getObservation_JS())

    def getObservation_JS(self):
        observation = []
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.robotUid, i)
            jointState = p.getJointState(self.robotUid, i)
            qIndex = jointInfo[3]
            if qIndex > 0:
                observation.append(jointState[0])
        self.jointsStates = observation
        return np.array(observation)

    def getObservation_EE(self, useInverseKinematics=None):
        if useInverseKinematics is None:
            useInverseKinematics = self.useInverseKinematics
        state = p.getLinkState(self.robotUid, self.robotEndEffectorIndex)
        pos = state[0]
        orn = state[1]
        rpy = p.getEulerFromQuaternion(orn)
        if useInverseKinematics:
            observation = pos + rpy[-1:]
        else:
            observation = pos + orn
        return np.array(observation)

    def normalize_JS(self, js):
        norm_js = js.copy()
        for i in range(len(js)):
            norm_js[i] = (js[i] - self.ll[i]) / (self.ul[i] - self.ll[i])
        return norm_js

    #### apply action ####
    def getActionDimension(self):
        if (self.useInverseKinematics):
            return 4
        return 6

    def applyAction(self, motorCommands, use_reset=False):
        if len(motorCommands) != self.getActionDimension():
            # print(motorCommands)
            # print(self.getActionDimension())
            print("ERROR: dimension does not match!")

        if self.useInverseKinematics:
            curr_EE = self.getObservation_EE()
            commands_EE = [motorCommands[i] + curr_EE[i] for i in range(len(curr_EE))]
            commands_pos = commands_EE[:3]
            commands_orn = list(p.getQuaternionFromEuler([0, 0, commands_EE[-1]]))
            commands_js = p.calculateInverseKinematics(self.robotUid,
                                                       self.robotEndEffectorIndex,
                                                       commands_pos,
                                                       commands_orn,
                                                       jointDamping=self.jd)
            if not use_reset:
                for i in range(len(commands_js)):
                    idx = self.jointsInfos[i][0]
                    command = commands_js[i]
                    p.setJointMotorControl2(bodyUniqueId=self.robotUid,
                                            jointIndex=idx,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=command,
                                            )
            else:
                self.initialize_by_JS_pose(commands_js)
        else:
            if not use_reset:
                commands_JS = [self.jointsStates[i] + motorCommands[i] for i in range(len(self.jointsStates))]
                for i in range(len(motorCommands)):
                    idx = self.jointsInfos[i][0]
                    command = commands_JS[i]
                    p.setJointMotorControl2(bodyUniqueId=self.robotUid,
                                            jointIndex=idx,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=command,
                                            )
            else:
                commands_JS = [self.jointsStates[i] + motorCommands[i] for i in range(len(self.jointsStates))]
                self.initialize_by_JS_pose(commands_JS)

    def calculStraightAction2Goal(self, goal, type="ee", commands_scale=None):
        commands_scale = commands_scale if commands_scale is not None else self.default_commands_scale
        if type == "js":
            curr_js = self.getObservation_JS()
            delta_js = goal - curr_js
            straight_action_js = [0] * len(delta_js)
            for i in range(len(delta_js)):
                if abs(delta_js[i]) < 0.000001:
                    delta_js[i] = 0
                straight_action_js[i] = min(abs(delta_js[i]), commands_scale) if delta_js[i] >= 0 else -min(
                    abs(delta_js[i]), commands_scale)
                # straight_action_js[i] /= commands_scale
            return np.array(straight_action_js)
        elif type == "ee":
            assert len(goal) == self.getActionDimension()
            curr_ee = self.getObservation_EE()
            if not self.useInverseKinematics:
                curr_ee = np.concatenate((curr_ee[:3], p.getEulerFromQuaternion(curr_ee[3:])))
            delta_ee = goal - curr_ee
            len_delta_ee_pos = np.linalg.norm(delta_ee[:3])
            len_delta_ee_orn = np.linalg.norm(delta_ee[3:])
            len_delta_ee_pos = np.linalg.norm(delta_ee)
            len_delta_ee_orn = np.linalg.norm(delta_ee)
            straight_action_ee = [0] * len(delta_ee)
            for i in range(len(delta_ee)):
                if abs(delta_ee[i]) < 0.00001:
                    delta_ee[i] = 0
                if i < 3:
                    scale_len = commands_scale * (abs(delta_ee[i]) / len_delta_ee_pos)
                else:
                    scale_len = commands_scale * (abs(delta_ee[i]) / len_delta_ee_orn)
                straight_action_ee[i] = min(abs(delta_ee[i]), scale_len) * 1 if delta_ee[i] >= 0 else -min(
                    abs(delta_ee[i]), scale_len)
                # straight_action_ee[i] /= commands_scale
            return np.array(straight_action_ee)
        else:
            return None

    def pickItem(self, part=None):
        assert part is not None
        self.item_picking = part
        self.constrain_cid = p.createConstraint(self.robotUid, self.robotEndEffectorIndex, part.partUid,
                                                part.link_idx_to_pick,
                                                p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        self.contactUids.append(part.partUid)
        self.picking_count += 1
        return self.constrain_cid

    def placeItem(self, part):
        if part is None or self.constrain_cid is None:
            return False
        else:

            p.removeConstraint(self.constrain_cid)
            self.constrain_cid = None
            self.contactUids.remove(part.partUid)
            self.item_picking = None
            self.placing_count += 1
            return True

    #### pose initlization ####
    def initialize_random_pose(self):
        for i in range(len(self.init_jointStates)):
            if i > 0 and i < 7:
                self.init_jointStates[i] += 0.2 * (np.random.random() - 0.5)

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

    def initialize_by_JS_pose(self, jointPoses):
        self.init_jointStates = jointPoses
        for i in range(self.numJoints):
            p.resetJointState(self.robotUid, i, jointPoses[i])
        # print(self.init_jointStates)
        self.initialize_item()
        return True

    def reset_by_JS_pose(self, js):
        js = np.concatenate([js])
        for i in range(self.numJoints):
            p.resetJointState(self.robotUid, i, js[i])
        self.initialize_item()
        return True

    def initialize_by_EE_pose(self, target_pos):
        EE_pos = target_pos[:3]
        if self.useInverseKinematics:
            EE_orn = p.getQuaternionFromEuler([0, 0, EE_pos[-1]])
        else:
            EE_orn = target_pos[3:]

        for jointIndex in range(self.numJoints):
            p.resetJointState(self.robotUid, jointIndex, self.robot_config.init_jointStates[jointIndex])
        if EE_orn is not None:
            jointPoses = p.calculateInverseKinematics(self.robotUid,
                                                      self.robotEndEffectorIndex,
                                                      EE_pos,
                                                      EE_orn,
                                                      jointDamping=self.jd)
        else:
            jointPoses = p.calculateInverseKinematics(self.robotUid,
                                                      self.robotEndEffectorIndex,
                                                      EE_pos,
                                                      jointDamping=self.jd)
        self.init_jointStates = jointPoses
        for i in range(self.numJoints):
            p.resetJointState(self.robotUid, i, jointPoses[i])
        # print(self.init_jointStates)
        self.initialize_item()
        return True

    def reset_by_EE_pose(self, target_pos):
        EE_pos = target_pos[:3]
        if self.useInverseKinematics:
            EE_orn = p.getQuaternionFromEuler([0, 0, EE_pos[-1]])
        else:
            EE_orn = target_pos[3:]

        for jointIndex in range(self.numJoints):
            p.resetJointState(self.robotUid, jointIndex, self.robot_config.init_jointStates[jointIndex])
        if EE_orn is not None:
            jointPoses = p.calculateInverseKinematics(self.robotUid,
                                                      self.robotEndEffectorIndex,
                                                      EE_pos,
                                                      EE_orn,
                                                      jointDamping=self.jd)
        else:
            jointPoses = p.calculateInverseKinematics(self.robotUid,
                                                      self.robotEndEffectorIndex,
                                                      EE_pos,
                                                      jointDamping=self.jd)
        for i in range(self.numJoints):
            p.resetJointState(self.robotUid, i, jointPoses[i])
        self.initialize_item()
        return True

    def initialize_item(self):
        item = self.item_picking
        if item is not None:
            ee = self.getObservation_EE()
            pos = ee[:3]
            if self.useInverseKinematics:
                rz = ee[-1]
                orn = p.getQuaternionFromEuler([0, 0, rz])
            else:
                orn = ee[3:]
            item.resetInitPose(pos, orn, reset_by_ee_pose=True)

    #### collision detection ####
    def update_closest_points(self):
        others_id = [p.getBodyUniqueId(i)
                     for i in range(p.getNumBodies())
                     if p.getBodyUniqueId(i) != self.robotUid]
        part_id = None
        if self.item_picking is not None:
            part_id = self.item_picking.partUid
            others_id.remove(part_id)

        self.closest_points_robot_to_others = [
            sorted(list(p.getClosestPoints(
                bodyA=self.robotUid, bodyB=other_id,
                distance=self.max_distance_from_others)),
                key=lambda contact_points: contact_points[8])
            if other_id != 0 else []
            for other_id in others_id]
        if part_id is not None:
            self.closest_points_part_to_others = [
                sorted(list(p.getClosestPoints(
                    bodyA=part_id, bodyB=other_id,
                    distance=self.max_distance_from_others)),
                    key=lambda contact_points: contact_points[8])
                if other_id != 0 else []
                for other_id in others_id]
        else:
            self.closest_points_part_to_others = []
        self.closest_points_to_others = self.closest_points_robot_to_others + self.closest_points_part_to_others
        self.closest_points_to_self = [
            p.getClosestPoints(
                bodyA=self.robotUid, bodyB=self.robotUid,
                distance=0,
                linkIndexA=link1, linkIndexB=link2)
            for link1, link2 in self.link_pairs]

    def check_collision(self, collision_distance=0.00):
        self.update_closest_points()
        # Collisions with others
        for i, closest_points_to_other in enumerate(
                self.closest_points_to_others):
            if len(closest_points_to_other) > 0:

                for point in closest_points_to_other:
                    if point[1] == self.robotUid and point[2] in self.contactUids:
                        if point[8] < -0.01:
                            self.prev_collided_with = point
                            # print("point in closest_points_to_other::point[8] < 0")
                            # print("hit\t",point[1],point[2])
                            return True, point[2]
                    elif self.item_picking is not None and point[1] == self.item_picking.partUid and point[2] == 1:
                        if point[8] < -0.01:
                            self.prev_collided_with = point
                            # print("point in closest_points_to_other::point[8] < 0")
                            # print("hit\t",point[1],point[2])
                            return True, point[2]
                    else:
                        if point[8] < collision_distance:
                            self.prev_collided_with = point
                            # print("point in closest_points_to_other::point[8] < collision_distance")
                            # print("hit\t",point[1],point[2])
                            return True, point[2]
        # Self Collision
        for closest_points_to_self_link in self.closest_points_to_self:
            for point in closest_points_to_self_link:
                if len(point) > 0:
                    self.prev_collided_with = point
                    # print("closest_points_to_self_link in self.closest_points_to_self:len(point) > 0:")
                    return True, self.robotUid
        self.prev_collided_with = None
        return False, None
