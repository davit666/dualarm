import numpy as np


def remove_duplicates_from_list(x):
    return list(dict.fromkeys(x))


def same_point(p, q, allow_same_point=False):
    if allow_same_point:
        return np.linalg.norm(np.array(p) - np.array(q)) < 1e-6
    return p[0] == q[0] and p[1] == q[1]


def check_point_in_line_segment(s, p1, p2, plot=False, allow_same_point=False):
    '''
    in 2d space, check if point s posits in line segment(p1,p2) exclude p1, p2
    '''
    a1 = p1[1] - p2[1]
    b1 = p2[0] - p1[0]
    c1 = p1[0] * p2[1] - p2[0] * p1[1]
    if plot:
        print(s, p1, p2)
        print(a1 * s[0] + b1 * s[1] + c1)
    if abs(a1 * s[0] + b1 * s[1] + c1) < 1e-6:
        if ((s[0] - p1[0]) * (s[0] - p2[0]) <= 1e-6) and ((s[1] - p1[1]) * (s[1] - p2[1]) <= 1e-6):
            if not allow_same_point and (same_point(s, p1) or same_point(s, p2)):
                return False
            else:
                return True
    return False


def check_is_triangle(p1, p2, p3):
    if same_point(p1, p2):
        return False
    elif same_point(p2, p3):
        return False
    elif same_point(p1, p3):
        return False
    elif check_point_in_line_segment(p1, p2, p3):
        return False
    elif check_point_in_line_segment(p2, p1, p3):
        return False
    elif check_point_in_line_segment(p3, p1, p2):
        return False
    else:
        return True


def get_line_intersection(p1, p2, q1, q2, plot=False):
    '''
    in 2D space, judge if line(p1, p2) intersects with line(q1, q2), return bool and intersect point(if available), only for line segment
    '''
    a1 = p1[1] - p2[1]
    b1 = p2[0] - p1[0]
    c1 = p1[0] * p2[1] - p2[0] * p1[1]

    a2 = q1[1] - q2[1]
    b2 = q2[0] - q1[0]
    c2 = q1[0] * q2[1] - q2[0] * q1[1]

    if plot:
        print(a1, " * x + ", b1, " * y + ", c1, " = 0")
        print(a2, " * x + ", b2, " * y + ", c2, " = 0")

    D = a1 * b2 - a2 * b1

    if D == 0:
        if plot:
            print("line 1 parallel with line 2")
        return False, None
    else:
        x = round((b1 * c2 - b2 * c1) / D, 8)
        y = round((a2 * c1 - a1 * c2) / D, 8)
        s = [x, y]
        if plot:
            print("intersect point is :", s)
        if check_point_in_line_segment(s, p1, p2) and check_point_in_line_segment(s, q1, q2):
            return True, s
        else:
            return False, s


def calcul_triangle_area(p1, p2, p3, minimum_triangle_area=None):
    area = abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0)
    if minimum_triangle_area is not None:
        area = max(area, minimum_triangle_area)
    return area


def check_point_in_triangle(s, p1, p2, p3):
    area_triangle = calcul_triangle_area(p1, p2, p3)

    if area_triangle < 1e-6:
        if check_point_in_line_segment(s, p1, p2) or check_point_in_line_segment(s, p1,
                                                                                 p3) or check_point_in_line_segment(s,
                                                                                                                    p2,
                                                                                                                    p3):
            return True
        else:
            return False
    area_s12 = calcul_triangle_area(s, p1, p2)
    area_s23 = calcul_triangle_area(s, p2, p3)
    area_s13 = calcul_triangle_area(s, p1, p3)
    area_s = area_s13 + area_s23 + area_s12
    if abs(area_triangle - area_s) < 1e-6:
        return True
    else:
        return False


from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt


def calcul_convex_polygon_area_D(p_list=[], plot=False):
    points = np.array(p_list)
    if len(points) < 3:
        if plot:
            print("no polygon created by points. points: ", points)
        return 0
    delaunay = Delaunay(points)
    tris = delaunay.simplices
    tris_area = 0
    if plot:
        plot_polygon_D(points, delaunay)
    for k, tri in enumerate(tris):
        tri_points = [p_list[tri[0]], p_list[tri[1]], p_list[tri[2]]]
        tri_area = calcul_triangle_area(tri_points[0], tri_points[1], tri_points[2])
        if plot:
            print("the {}th triangle is: ".format(k + 1), tri_points)
            print("the area of that triangle is: ", tri_area)
        tris_area += tri_area
    if plot:
        print("total triangle numer is:{} and total area is: {}".format(k + 1, tris_area))
    return tris_area


def plot_polygon_D(points, delaunay):
    plt.triplot(points[:, 0], points[:, 1], delaunay.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()
    return True


def calcul_convex_polygon_area_C(p_list=[], plot=False):
    points = np.array(p_list)
    if len(points) < 3:
        if plot:
            print("no polygon created by points. points: ", points)
        return 0
    hull = ConvexHull(points)
    if plot:
        plot_polygon_C(points, hull)
    hull_area = hull.volume
    if plot:
        print("the convex hull is: ", points[hull.simplices])
        print("the area is: ", hull_area)
    return hull_area


def plot_polygon_C(points, hull):
    plt.plot(points[:, 0], points[:, 1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'r--')
    return True


from itertools import combinations


def get_overlap_convex_polygon(tri1_points=[], tri2_points=[], plot=False):
    assert len(tri1_points) == 3, tri1_points
    assert len(tri2_points) == 3, tri2_points
    overlap_p_points = []
    for p in tri1_points:
        if check_point_in_triangle(p, tri2_points[0], tri2_points[1], tri2_points[2]):
            if not any([same_point(p, k) for k in overlap_p_points]):
                overlap_p_points.append(np.array(p))
    for q in tri2_points:
        if check_point_in_triangle(q, tri1_points[0], tri1_points[1], tri1_points[2]):
            if not any([same_point(q, k) for k in overlap_p_points]):
                overlap_p_points.append(np.array(q))
    comb1 = combinations(tri1_points, 2)
    edges_1 = []
    for pair in comb1:
        edges_1.append(pair)
    comb2 = combinations(tri2_points, 2)
    edges_2 = []
    for pair in comb2:
        edges_2.append(pair)
    for edge_1 in edges_1:
        for edge_2 in edges_2:
            intersect, s = get_line_intersection(edge_1[0], edge_1[1], edge_2[0], edge_2[1], plot=False)
            if intersect:
                if not any([same_point(s, k) for k in overlap_p_points]):
                    overlap_p_points.append(np.array(s))
    # overlap_p_points = remove_duplicates_from_list(overlap_p_points)
    if plot:
        print("overlap_points_set is: ", overlap_p_points)
    return overlap_p_points


def get_overlap_area(tri1_points=[], tri2_points=[], type="D", plot=False, minimun_overlap_area=1e-5,
                     overlap_convex_polygon_points=None):
    assert len(tri1_points) == 3, tri1_points
    assert len(tri2_points) == 3, tri2_points

    if overlap_convex_polygon_points is None:
        overlap_p_points = get_overlap_convex_polygon(tri1_points, tri2_points, plot=plot)
    else:
        overlap_p_points = overlap_convex_polygon_points

    if len(overlap_p_points) == 0:
        return 0
    if type == "D":
        area = calcul_convex_polygon_area_D(overlap_p_points, plot=plot)
    elif type == "C":
        area = calcul_convex_polygon_area_C(overlap_p_points, plot=plot)
    else:
        area = -1
    return max(area, minimun_overlap_area)


def get_edge_length_of_triangle(p_base, p_ee, p_goal):
    base_ee = np.linalg.norm(p_base[:2] - p_ee[:2])
    base_goal = np.linalg.norm(p_base[:2] - p_goal[:2])
    ee_goal = np.linalg.norm(p_ee[:2] - p_goal[:2])
    return [base_ee, base_goal, ee_goal]


def get_base_angle_of_triangle(p_base, p_ee, p_goal, base_axis='y', use_cosin=False):
    if base_axis == 'y':
        v_axis = [1, 0]
    elif base_axis == 'x':
        v_axis = [0, 1]
    else:
        return None
    v_base_ee = [p_ee[0] - p_base[0], p_ee[1] - p_base[1]]
    v_base_goal = [p_goal[0] - p_base[0], p_goal[1] - p_base[1]]

    v_axis = np.array(v_axis)
    v_base_ee = np.array(v_base_ee) / np.linalg.norm(v_base_ee)
    v_base_goal = np.array(v_base_goal) / np.linalg.norm(v_base_goal)

    cos_base_ee = np.dot(v_axis, v_base_ee)
    cos_base_goal = np.dot(v_axis, v_base_goal)
    cos_ee_goal = np.dot(v_base_ee, v_base_goal)

    cos_base_ee = min(1, max(-1, cos_base_ee))
    cos_base_goal = min(1, max(-1, cos_base_goal))
    cos_ee_goal = min(1, max(-1, cos_ee_goal))

    if use_cosin:
        return [cos_base_ee, cos_base_goal, cos_ee_goal]

    angle_base_ee = np.arccos(cos_base_ee)
    angle_base_goal = np.arccos(cos_base_goal)
    angle_ee_goal = np.arccos(cos_ee_goal)
    return [angle_base_ee, angle_base_goal, angle_ee_goal]


# def check_point_in_project_line(s, p1, p2, plot=False):
#     '''
#     in 2d space, check if point s posits in project line(p1,p2) exclude p1, p2, p2 between p1 and s
#     '''
#     a1 = p1[1] - p2[1]
#     b1 = p2[0] - p1[0]
#     c1 = p1[0] * p2[1] - p2[0] * p1[1]
#     if plot:
#         print(s, p1, p2)
#         print(a1 * s[0] + b1 * s[1] + c1)
#     if abs(a1 * s[0] + b1 * s[1] + c1) < 1e-6:
#         if ((p2[0] - p1[0]) * (p2[0] - s[0]) <= 0) and ((p2[1] - p1[1]) * (p2[1] - s[1]) <= 0):
#             if same_point(s, p1) or same_point(s, p2):
#                 return False
#             else:
#                 return True
#     return False


def get_line_intersection_2(p1, p2, q1, q2, plot=False):
    '''
    in 2D space, judge if line(p1, p2) intersects with line(q1, q2), return bool and intersect point(if available), line p is project line and line q is segment
    '''
    a1 = p1[1] - p2[1]
    b1 = p2[0] - p1[0]
    c1 = p1[0] * p2[1] - p2[0] * p1[1]

    a2 = q1[1] - q2[1]
    b2 = q2[0] - q1[0]
    c2 = q1[0] * q2[1] - q2[0] * q1[1]

    if plot:
        print(a1, " * x + ", b1, " * y + ", c1, " = 0")
        print(a2, " * x + ", b2, " * y + ", c2, " = 0")

    D = a1 * b2 - a2 * b1

    if D == 0:
        if plot:
            print("line 1 parallel with line 2")
        s = p2
        if np.linalg.norm(np.array(p1) - np.array(s)) < np.linalg.norm(np.array(p1) - np.array(q1)):
            s = q1
        if np.linalg.norm(np.array(p1) - np.array(s)) < np.linalg.norm(np.array(p1) - np.array(q2)):
            s = q2
        return True, s
    else:
        x = round((b1 * c2 - b2 * c1) / D,8)
        y = round((a2 * c1 - a1 * c2) / D,8)
        s = [x, y]
        if plot:
            print("intersect point is :", s)
            print("a:", check_point_in_line_segment(p2, p1, s, allow_same_point=True))
            print("b:",check_point_in_line_segment(s, q1, q2, allow_same_point=True))
        if check_point_in_line_segment(p2, p1, s, allow_same_point=True) and check_point_in_line_segment(s, q1, q2,
                                                                                                         allow_same_point=True):
            return True, s
        else:
            return False, s


def get_cutting_ratio_of_point_in_triangle(tri_points=[], cutting_point=None, plot=False):
    assert len(tri_points) == 3, tri_points
    assert cutting_point is not None, cutting_point

    intersect, s = get_line_intersection_2(tri_points[0], cutting_point, tri_points[1], tri_points[2], plot=plot)
    if not intersect:
        # print(
        #     "not proper intersection point for the project line and line segment, something goes wrong, recheck code!")
        # print("tri_points:   ", tri_points)
        # print("cutting_point:  ", cutting_point)
        # print("s: ",s)
        s = tri_points[1] if np.linalg.norm(tri_points[0] - tri_points[1]) > np.linalg.norm(
            tri_points[0] - tri_points[2]) else tri_points[2]
    dist0 = np.linalg.norm(tri_points[0] - cutting_point)
    dist1 = np.linalg.norm(tri_points[0] - s)
    if dist0 > dist1 + 1e-6:
        # print("segment p1 q length is larger than segment p1 s length, something goes wrong, recheck code!")
        # print("tri_points:   ", tri_points)
        # print("cutting_point:  ", cutting_point)
        # print("s: ", s)
        # print("dist0-1: ",dist0, dist1)
        return 0
    return max(round((dist1 - dist0) / dist1,8), 0)


def get_cutting_ratio(tri1_points=[], tri2_points=[], plot=False, use_area=True, overlap_convex_polygon_points=None):
    assert len(tri1_points) == 3, tri1_points
    assert len(tri2_points) == 3, tri2_points

    if overlap_convex_polygon_points is None:
        overlap_p_points = get_overlap_convex_polygon(tri1_points, tri2_points, plot=plot)
    else:
        overlap_p_points = overlap_convex_polygon_points

    if len(overlap_p_points) == 0:
        return 0
    cutting_ratio_list = []
    for cutting_point in overlap_p_points:
        cutting_ratio = get_cutting_ratio_of_point_in_triangle(tri_points=tri1_points, cutting_point=cutting_point,
                                                               plot=plot)
        cutting_ratio_list.append(cutting_ratio)
    # print("cutting_ratio_list",cutting_ratio_list)
    max_cutting_ratio = max(cutting_ratio_list)
    if use_area:
        max_cutting_ratio = max_cutting_ratio ** 2
    return max_cutting_ratio


def get_overlap_points(tri1_points=[], tri2_points=[], plot=False):
    assert len(tri1_points) == 3, tri1_points
    assert len(tri2_points) == 3, tri2_points
    overlap_p_points = get_overlap_convex_polygon(tri1_points, tri2_points, plot=plot)
    return overlap_p_points
