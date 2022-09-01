import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time
from tools import check_point_in_line_segment, check_is_triangle

class Robot_Triangle():
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.base = np.array([0., 0.])
        self.ee = np.array([0., 0.])
        self.goal = np.array([0., 0.])

        self.dot = None
        self.line_be = None
        self.line_bg = None
        self.line_eg = None

    def update_triangle(self, base=None, ee=None, goal=None):
        if base is not None:
            self.base[0] = base[0]
            self.base[1] = base[1]
        if ee is not None:
            self.ee[0] = ee[0]
            self.ee[1] = ee[1]
        if goal is not None:
            self.goal[0] = goal[0]
            self.goal[1] = goal[1]

    def get_triangles(self):
        return np.array([self.base, self.ee, self.goal])


class Robot_Triangle_Plot():
    def __init__(self, img_width=128, img_height=128, dpi=64, refresh_frequency=30, show=False, fill_triangle=False):

        self.img_width = img_width
        self.img_height = img_height
        self.dpi = dpi
        self.refresh_frequency = refresh_frequency
        self.show = show
        self.fill_triangle = fill_triangle

        fig_w = img_width // dpi
        fig_h = img_height // dpi
        self.fig = plt.figure(figsize=(fig_w, fig_h), dpi=self.dpi)
        self.ax = self.fig.add_subplot(111)

        self.ax.set_xlim([-0.8, 0.8])
        self.ax.set_ylim([-1.3, 1.3])
        self.ax.set_axis_off()
        self.ax.margins(0, 0)
        self.fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        self.robot_triangles = []
        self.img_matrix = None

    def create_robot_triangle(self, robot_id, base=None, ee=None, goal=None):
        robot_tri = Robot_Triangle(robot_id)
        robot_tri.update_triangle(base=base, ee=ee, goal=goal)
        self.robot_triangles.append(robot_tri)
        return True

    def update_robot_triangle(self, robot_id, base=None, ee=None, goal=None):
        if not any(robot_id == robot_tri.robot_id for robot_tri in self.robot_triangles):
            return False
        robot_tri = self.robot_triangles[robot_id]
        robot_tri.update_triangle(base=base, ee=ee, goal=goal)
        return True

    def create_plot(self):
        for robot_tri in self.robot_triangles:
            tri_points = robot_tri.get_triangles()
            robot_tri.dot = self.ax.plot(tri_points[1, 0], tri_points[1, 1], 'ko')
            robot_tri.line_be = self.ax.plot(tri_points[:2, 0], tri_points[:2, 1], 'k-')
            robot_tri.line_bg = self.ax.plot(tri_points[[0, 2], 0], tri_points[[0, 2], 1], 'k-')
            robot_tri.line_eg = self.ax.plot(tri_points[1:, 0], tri_points[1:, 1], 'k-')
            if self.fill_triangle:
                # tri = mtri.Triangulation(tri_points[:, 0], tri_points[:, 1], [[0, 1, 2]])
                if check_is_triangle(tri_points[0,:], tri_points[1, :], tri_points[2,:]):
                    robot_tri.trifill = self.ax.tripcolor(tri_points[:, 0],tri_points[:, 1],[0, 1, 2],color = 'red')
                else:
                    robot_tri.trifill = None
        self.fig.canvas.draw()
        return True
    def update_plot(self):
        for robot_tri in self.robot_triangles:
            tri_points = robot_tri.get_triangles()
            self.update_line(robot_tri.dot, tri_points[[1], :])
            self.update_line(robot_tri.line_be, tri_points[:2, :])
            self.update_line(robot_tri.line_bg, tri_points[[0, 2], :])
            self.update_line(robot_tri.line_eg, tri_points[1:, :])

            if self.fill_triangle:
                if robot_tri.trifill is not None:
                    robot_tri.trifill.remove()
                    robot_tri.trifill = None
                # tri = mtri.Triangulation(tri_points[:, 0], tri_points[:, 1], [[0, 1, 2]])
                if check_is_triangle(tri_points[0,:], tri_points[1, :], tri_points[2,:]):
                    robot_tri.trifill = self.ax.tripcolor(tri_points[:, 0],tri_points[:, 1],[0, 1, 2],color = 'red')
                else:
                    robot_tri.trifill = None

        self.fig.canvas.draw()
        # self.ax.cla()
        # self.create_plot()

    def update_line(self, hl, new_data):
        hl[0].set_xdata(new_data[:, 0])
        hl[0].set_ydata(new_data[:, 1])
        # self.fig.draw()
        return True

    def output_img_matrix(self):
        img = np.array(self.fig.canvas.renderer._renderer)
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        imgGray = 255 - np.round(0.29895 * R + 0.58705 * G + 0.11405 * B)
        self.img_matrix = imgGray

        if self.show:
            self.show_img_matrix()

        return self.img_matrix

    def show_img_matrix(self, img_input=None):
        img = img_input if img_input is not None else self.img_matrix
        cv2.imshow("Robot Triangle Plot", img)
        cv2.waitKey(1)
        return True


def im_show(img):
    cv2.imshow("Color Image", img)
    cv2.waitKey(50)
# tri1 = [[0, -1.25], [-0.6, 0.2], [0.3, 0.4]]
# tri2 = [[0, 1.25], [0.3, 0.5], [-0.7, 0.7]]
# tri1 = np.array(tri1)
# tri2 = np.array(tri2)
#
#
# plot = Robot_Triangle_Plot()
# plot.create_robot_triangle(0)
# plot.create_robot_triangle(1)
# plot.create_plot()
#
# for i in range(100):
#     time.sleep(0.1)
#     tri1[:, 0] += 0.01
#     tri2[:, 1] -= 0.01
#     plot.update_robot_triangle(0, base=tri1[0], ee=tri1[1], goal=tri1[2])
#     plot.update_robot_triangle(1, base=tri2[0], ee=tri2[1], goal=tri2[2])
#     plot.update_plot()
#     img = plot.output_img_matrix()
#     plot.show_img_matrix()
