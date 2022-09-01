from pybullet_planning import BASE_LINK, RED, BLUE, GREEN
from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput, \
    reset_simulation, disconnect, set_camera_pose, has_gui, set_camera, wait_for_duration, wait_if_gui, apply_alpha
from pybullet_planning import Pose, Point, Euler
from pybullet_planning import multiply, invert, get_distance
from pybullet_planning import create_obj, create_attachment, Attachment
from pybullet_planning import link_from_name, get_link_pose, get_moving_links, get_link_name, get_disabled_collisions, \
    get_body_body_disabled_collisions, has_link, are_links_adjacent
from pybullet_planning import get_num_joints, get_joint_names, get_movable_joints, set_joint_positions, joint_from_name, \
    joints_from_names, get_sample_fn, get_distance_fn, get_extend_fn, get_joint_positions, check_initial_end, \
    plan_joint_motion, get_difference_fn, get_refine_fn
from pybullet_planning import dump_world, set_pose
from pybullet_planning import get_collision_fn, get_floating_body_collision_fn, expand_links, create_box
from pybullet_planning import pairwise_collision, pairwise_collision_info, draw_collision_diagnosis, body_collision_info

from pybullet_planning import birrt, lazy_prm, plan_lazy_prm
from pybullet_planning import MAX_DISTANCE, DEFAULT_RESOLUTION
import numpy as np


class MotionPlanner():
    def __init__(self, robots=[], parts=[], obj=[]):
        self.robots = robots
        self.robotUids = [robot.robotUid for robot in self.robots]
        self.robot_joints = [get_movable_joints(Uid) for Uid in self.robotUids]
        self.robot_joint_nums = [len(joint_num) for joint_num in self.robot_joints]
        self.robot_joint_names = [get_joint_names(self.robotUids[k], self.robot_joints[k]) for k in range((len(self.robots)))]

        self.parts = parts
        self.partsUids = [part.partUid for part in self.parts]

        self.robotbaseUids = [robot.baseCylinderUid for robot in self.robots]
        self.objUids = obj + self.robotbaseUids

        self.use_self_collisions = True
        self.robot_self_collision_disabled_link_names = [('base_link', 'link1'),
                                                         ('link1', 'link2'), ('link2', 'link3'),
                                                         ('link3', 'link4'), ('link4', 'link5'),
                                                         ('link5', 'link6')]
        self.robot_base_disabled_link_names = [('base_link', 'link0'),
                                               ('link1', 'link0')]

        self.planning_alg_name_list = ["birrt", "lazy_prm"]
        self.update_plannig_alg("birrt")

        self.update_planning_robots(robotUids=self.robotUids)

    def update_plannig_alg(self, alg_name):
        if alg_name not in self.planning_alg_name_list:
            return None
        self.alg_name = alg_name
        return self.alg_name
        # # planning algorithm selection
        # if alg_name == "birrt":
        #     self.plannig_alg = birrt
        # elif alg_name == "lazy_prm":
        #     self.plannig_alg = lazy_prm
        #
        # return self.plannig_alg

    def update_planning_robots(self, robotUids=[], obstacles=None, custom_limits=None, use_inverse_kinermatics=False):
        self.use_inverse_kinermatics = use_inverse_kinermatics
        if obstacles is None:
            obstacles = [None] * len(robotUids)
        if custom_limits is None:
            custom_limits = [{}] * len(robotUids)
        # get planning robots infomation
        robotUids = robotUids
        robot_joints = []
        robot_joint_nums = []
        robot_obstacles = []
        robot_custom_limits = []
        for k, Uid in enumerate(robotUids):
            # robot id
            robot_idx = self.robotUids.index(Uid)
            # robot joint id
            robot_joint = self.robot_joints[robot_idx]
            robot_joints.append(robot_joint)
            # robot joint number
            joint_num = self.robot_joint_nums[robot_idx]
            robot_joint_nums.append(joint_num)
            # robot obstacle list
            if obstacles[k] is None:
                robot_obstacle = self.robotUids + self.partsUids + self.objUids
                # print(robot_obstacle, Uid)
                robot_obstacle.remove(Uid)
            else:
                robot_obstacle = obstacles[k]
            robot_obstacles.append(robot_obstacle)
            # robot joint limits
            robot_custom_limit = self.get_custom_limits_from_name(Uid, custom_limits[k])
            robot_custom_limits.append(robot_custom_limit)

        # update group collision fn
        self.group_collision_fn = self.get_group_collision_fn(robotUids=robotUids, robot_joints=robot_joints,
                                                              robot_joint_nums=robot_joint_nums,
                                                              robot_obstacles=robot_obstacles,
                                                              robot_custom_limits=robot_custom_limits)
        # update group difference fn
        self.group_distance_fn = self.get_group_distance_fn(robotUids=robotUids,
                                                            robot_joints=robot_joints,
                                                            robot_joint_nums=robot_joint_nums)

        # update group sample fn
        self.group_sample_fn = self.get_group_sample_fn(robotUids=robotUids, robot_joints=robot_joints,
                                                        robot_joint_nums=robot_joint_nums,
                                                        robot_custom_limits=robot_custom_limits)

        # update group extend fn
        self.group_extend_fn = self.get_group_extend_fn(robotUids=robotUids, robot_joints=robot_joints,
                                                        robot_joint_nums=robot_joint_nums)

        self.curr_planning_robots = robotUids
        self.curr_total_joint_nums = sum(robot_joint_nums)

        return 0

    def get_group_collision_fn(self, robotUids=[], robot_joints=[], robot_joint_nums=[], robot_obstacles=[],
                               robot_custom_limits=[]):
        collision_fns = []
        for k, Uid in enumerate(robotUids):
            self_colision_links = get_disabled_collisions(Uid, self.robot_self_collision_disabled_link_names)
            robot_base_Uid = self.robotbaseUids[self.robotUids.index(Uid)]
            extra_disabled_collisions = get_body_body_disabled_collisions(Uid, robot_base_Uid,
                                                                          self.robot_base_disabled_link_names)

            fn = get_collision_fn(Uid, robot_joints[k], obstacles=robot_obstacles[k],
                                  self_collisions=self.use_self_collisions,
                                  disabled_collisions=self_colision_links,
                                  extra_disabled_collisions=extra_disabled_collisions,
                                  custom_limits=robot_custom_limits[k], max_distance=MAX_DISTANCE)
            collision_fns.append(fn)

        def group_collision_fn(confs, diagnosis=False):
            assert len(confs) == sum(robot_joint_nums)
            c = []
            idx = 0
            for k in range(len(collision_fns)):
                joint_num = robot_joint_nums[k]
                c.append(collision_fns[k](confs[idx:idx + joint_num], diagnosis=diagnosis))
                idx += joint_num
            return any(c)

        return group_collision_fn

    def get_group_difference_fn(self, robotUids=[], robot_joints=[], robot_joint_nums=[]):
        difference_fns = []
        for k in range(len(robotUids)):
            fn = get_difference_fn(robotUids[k], robot_joints[k])
            difference_fns.append(fn)

        def group_difference_fn(conf2, conf1):
            diff_tuple = tuple()
            idx = 0
            for k in range(len(difference_fns)):
                joint_num = robot_joint_nums[k]
                q1 = conf1[idx:idx + joint_num]
                q2 = conf2[idx:idx + joint_num]
                diff_tuple += difference_fns[k](q2, q1)
                idx += joint_num
            return diff_tuple

        return group_difference_fn

    def get_group_distance_fn(self, robotUids=[], robot_joints=[], robot_joint_nums=[], weights=None):
        total_joint_num = sum(robot_joint_nums)
        if weights is None:
            weights = 1 * np.ones(total_joint_num)
        else:
            assert len(weights) == total_joint_num
        joint_difference_fn = self.get_group_difference_fn(robotUids=robotUids, robot_joints=robot_joints,
                                                           robot_joint_nums=robot_joint_nums)

        def group_distance_fn(conf1, conf2):
            diff = np.array(joint_difference_fn(conf2, conf1))
            return np.sqrt(np.dot(weights, diff * diff))

        return group_distance_fn

    def get_group_sample_fn(self, robotUids=[], robot_joints=[], robot_joint_nums=[], robot_custom_limits=[]):
        sample_fns = []
        for k in range(len(robotUids)):
            fn = get_sample_fn(robotUids[k], robot_joints[k], custom_limits=robot_custom_limits[k])
            sample_fns.append(fn)

        def group_sample_fn():
            samp_tuple = tuple()
            for k in range(len(sample_fns)):
                samp_tuple += sample_fns[k]()
            return samp_tuple

        return group_sample_fn

    def get_group_refine_fn(self, robotUids=[], robot_joints=[], robot_joint_nums=[], num_steps=0):
        joint_difference_fn = self.get_group_difference_fn(robotUids=robotUids, robot_joints=robot_joints,
                                                           robot_joint_nums=robot_joint_nums)
        num_steps = num_steps + 1

        def group_refine_fn(conf1, conf2):
            conf = conf1
            yield conf1
            for i in range(num_steps):
                positions = (1. / (num_steps - i)) * np.array(joint_difference_fn(conf2, conf)) + conf
                conf = tuple(positions)
                yield conf

        return group_refine_fn

    def get_group_extend_fn(self, robotUids=[], robot_joints=[], robot_joint_nums=[], resolutions=None, norm=2):
        # norm = 1, 2, INF
        total_joint_num = sum(robot_joint_nums)
        if resolutions is None:
            resolutions = DEFAULT_RESOLUTION * np.ones(total_joint_num)
        else:
            assert len(resolutions) == total_joint_num
        joint_difference_fn = self.get_group_difference_fn(robotUids=robotUids, robot_joints=robot_joints,
                                                           robot_joint_nums=robot_joint_nums)

        def joint_extend_fn(conf1, conf2):
            steps = int(np.ceil(np.linalg.norm(np.divide(joint_difference_fn(conf2, conf1), resolutions), ord=norm)))
            joint_refine_fn = self.get_group_refine_fn(robotUids=robotUids, robot_joints=robot_joints,
                                                       robot_joint_nums=robot_joint_nums, num_steps=steps)
            return joint_refine_fn(conf1, conf2)

        return joint_extend_fn

    def plan_group_motion(self, robotUids, start_conf, end_conf, **kwargs):
        assert robotUids == self.curr_planning_robots
        assert len(start_conf) == self.curr_total_joint_nums
        assert len(start_conf) == len(end_conf)
        sample_fn = self.group_sample_fn
        distance_fn = self.group_distance_fn
        extend_fn = self.group_extend_fn
        collision_fn = self.group_collision_fn

        # check init end conf
        if not check_initial_end(start_conf, end_conf, collision_fn, diagnosis=False):
            return None

        # calculate motion planning solution
        if self.alg_name == "birrt":
            solution = birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)
        elif self.alg_name == "lazy_prm":
            solution, samples, edges, colliding_vertices, colliding_edges = lazy_prm(
                start_conf, end_conf, sample_fn, extend_fn, collision_fn, num_samples=50, **kwargs)
            # print("path:", path)
            # print("samples:", samples)
            # print("colliding vertices:", colliding_vertices)
            # print("colliding edges:", colliding_edges)
        else:
            print("no planning algorithm selected")
            solution = None
        if solution is None:
            return solution

        # generate trajectory based on solution
        trajectory = []
        conf0 = start_conf
        for conf in solution:
            sub_solution = extend_fn(conf0, conf)
            for path in sub_solution:
                trajectory.append(path)
            conf0 = conf

        return trajectory

    def set_all_joint_position(self, robotUids=[], confs=None):
        idx = 0
        for Uid in robotUids:
            robot_idx = self.robotUids.index(Uid)
            robot_joint = self.robot_joints[robot_idx]
            joint_num = self.robot_joint_nums[robot_idx]
            robot_conf = confs[idx:idx + joint_num]
            set_joint_positions(Uid, robot_joint, robot_conf)
            idx += joint_num
        return 1

    def get_custom_limits_from_name(self, robotUid, joint_limits):
        return {joint_from_name(robotUid, joint): limits
                for joint, limits in joint_limits.items()}
