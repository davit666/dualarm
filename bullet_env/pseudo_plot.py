import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time
from tools import check_point_in_line_segment, check_is_triangle


class PseudoBasePlot():
    def __init__(self, base):
        self.left_bottom = np.array([base[0, 0], base[1, 0]])
        self.left_top = np.array([base[0, 0], base[1, 1]])
        self.right_bottom = np.array([base[0, 1], base[1, 0]])
        self.right_top = np.array([base[0, 1], base[1, 1]])

        self.line_left = None
        self.line_right = None
        self.line_bottom = None
        self.line_top = None


class PseudoRobotPlot():
    def __init__(self, robot_id):
        self.robot_id = robot_id

        self.base = np.array([0, 0])
        self.ee = np.array([0, 0])
        self.rest_pose = np.array([0, 0])

        self.dot_base = None
        self.dot_ee = None
        self.dot_rest_pose = None
        self.line_robot = None

    def update_data(self, ee, rest_pose=None, base=None):
        self.ee = np.array(ee[0:2])
        if rest_pose is not None:
            self.rest_pose = np.array(rest_pose[:2])
        if base is not None:
            self.base = np.array(base[:2])

    def update_plot(self):
        pass


class PseudoPartPlot():
    def __init__(self, part_id, color=None):
        self.pat_id = part_id
        self.color = color

        self.pose = np.array([0, 0])
        self.goal = np.array([0, 0])

        self.dot_pose = None
        self.dot_goal = None
        self.line_path = None

    def update_data(self, pose, goal=None):
        self.pose = np.array(pose[:2])
        if goal is not None:
            self.goal = np.array(goal[:2])

    def update_plot(self):
        pass


class PseudoPlot():
    def __init__(self, img_width=1024, img_height=1024, dpi=128):
        self.img_width = img_width
        self.img_height = img_height
        self.dpi = dpi

        fig_w = img_width // dpi
        fig_h = img_height // dpi
        self.fig = plt.figure(figsize=(8, 8), dpi=300)
        self.ax = self.fig.add_subplot(111)

        self.ax.set_xlim([-1.3, 1.3])
        self.ax.set_ylim([-1.3, 1.3])
        self.ax.set_axis_off()
        self.ax.margins(0, 0)
        self.fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.t_name = str(time.time())
        self.dot_size = 10

    def setup(self, base, robots, parts):
        self.p_base = PseudoBasePlot(base)
        self.p_base.line_left = self.ax.plot([self.p_base.left_bottom[0], self.p_base.left_top[0]],
                                             [self.p_base.left_bottom[1], self.p_base.left_top[1]], 'k-')
        self.p_base.line_top = self.ax.plot([self.p_base.left_top[0], self.p_base.right_top[0]],
                                            [self.p_base.left_top[1], self.p_base.right_top[1]], 'k-')
        self.p_base.line_right = self.ax.plot([self.p_base.right_top[0], self.p_base.right_bottom[0]],
                                              [self.p_base.right_top[1], self.p_base.right_bottom[1]], 'k-')
        self.p_base.line_bottom = self.ax.plot([self.p_base.right_bottom[0], self.p_base.left_bottom[0]],
                                               [self.p_base.right_bottom[1], self.p_base.left_bottom[1]], 'k-')
        self.p_robots = []
        self.p_parts = []
        # robot_colors = ['y','g']
        for i, robot in enumerate(robots):
            p_robot = PseudoRobotPlot(i)
            robot_ee = robot.getObservation_EE()
            robot_goal = robot.goal_pose
            robot_base = robot.BasePos

            p_robot.update_data(robot_ee, rest_pose=robot_goal, base=robot_base)
            p_robot.dot_base = self.ax.plot([p_robot.base[0]], [p_robot.base[1]], 'ro',markersize=self.dot_size)
            p_robot.dot_ee = self.ax.plot([p_robot.ee[0]], [p_robot.ee[1]], 'ro',markersize=self.dot_size)
            p_robot.dot_rest_pose = self.ax.plot([p_robot.rest_pose[0]], [p_robot.rest_pose[1]], 'g*',markersize=self.dot_size)
            p_robot.line_robot = self.ax.plot([p_robot.base[0], p_robot.ee[0]], [p_robot.base[1], p_robot.ee[1]], 'r-',markersize=self.dot_size)

            self.p_robots.append(p_robot)
        for j, part in enumerate(parts):
            p_part = PseudoPartPlot(j)
            part_init = part.getPose()
            part_goal = part.getGoalPose()
            p_part.update_data(part_init, goal=part_goal)
            p_part.dot_pose = self.ax.plot([p_part.pose[0]], [p_part.pose[1]], 'bo',markersize=self.dot_size)
            p_part.dot_goal = self.ax.plot([p_part.goal[0]], [p_part.goal[1]], 'y*',markersize=self.dot_size)
            p_part.line_path = self.ax.plot([p_part.pose[0], p_part.goal[0]], [p_part.pose[1], p_part.goal[1]], 'b--')

            self.p_parts.append(p_part)

        return True

    def update_plot(self, robots, parts):
        for i, robot in enumerate(robots):
            robot_ee = robot.getObservation_EE()
            robot_goal = robot.goal_pose
            robot_base = robot.BasePos

            p_robot = self.p_robots[i]
            p_robot.update_data(robot_ee, rest_pose=robot_goal, base=robot_base)


            p_robot.dot_ee[0].set_xdata([p_robot.ee[0]])
            p_robot.dot_ee[0].set_ydata([p_robot.ee[1]])
            p_robot.dot_rest_pose[0].set_xdata([p_robot.rest_pose[0]])
            p_robot.dot_rest_pose[0].set_ydata([p_robot.rest_pose[1]])
            p_robot.line_robot[0].set_xdata([p_robot.base[0], p_robot.ee[0]])
            p_robot.line_robot[0].set_ydata([p_robot.base[1], p_robot.ee[1]])

        for j, part in enumerate(parts):
            part_init = part.getPose()
            part_goal = part.getGoalPose()

            p_part = self.p_parts[j]
            p_part.update_data(part_init, goal=part_goal)

            p_part.dot_pose[0].set_xdata([p_part.pose[0]])
            p_part.dot_pose[0].set_ydata([p_part.pose[1]])
            p_part.dot_goal[0].set_xdata([p_part.goal[0]])
            p_part.dot_goal[0].set_ydata([p_part.goal[1]])
            p_part.line_path[0].set_xdata([p_part.pose[0], p_part.goal[0]])
            p_part.line_path[0].set_ydata([p_part.pose[1], p_part.goal[1]])

        self.fig.canvas.draw()

    def imshow(self,):
        # img = np.array(self.fig.canvas.renderer._renderer)
        # cv2.imshow("Pseudo Task Plot", img)
        # cv2.waitKey(1)
        return True

    def save(self):
        self.fig.savefig("savefig/fig_"+time.strftime("%Y-%m-%d-%H-%M-%S")+".png")
        print("fig saved")
        time.sleep(1)
        return

