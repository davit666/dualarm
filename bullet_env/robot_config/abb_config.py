import math


class ABB():
    # path = "/robot/abb_irb4600/urdf/abb_irb4600.urdf"
    # path = "/robot/new/rs050n/urdf/rs050n.urdf"
    path = "/robot/old/RS050N/urdf/RS050N_2.urdf"

    BasePos = [0, -1.25, 0.61]
    BaseOrn = [0, 0, 0]

    maxVelocity = 1000
    maxForce = 1000

    actionDimension = 6
    robotEndEffectorIndex = 5
    robotGripperIndex = 5

    init_jointStates = [0, 0, 0, 0, 0, 0, 0, 0]
    upper_limits = [180, 140, 135, 360, 145, 360]
    lowwer_limits = [-180, -105, -155, -360, -145, -360]
    # upper_limits = [180,140,135,360,145,360]
    # lowwer_limits = [-180,-140,-135,-360,-145,-360]
    for i in range(len(upper_limits)):
        upper_limits[i] = upper_limits[i] / 180 * math.pi
        lowwer_limits[i] = lowwer_limits[i] / 180 * math.pi

    joint_ranges = None
    rest_poses = None
    joint_dumpings = [0.00001] * actionDimension

    init_pos1_x = [-1, 1]
    init_pos1_y = [-1., 0.5]
    init_pos1_z = [0.5, 1]

    init_pos2_x = [-0.75, 0.75]
    init_pos2_y = [-0.75, 0.5]
    init_pos2_z = [0.55, 0.6]

    default_ee_pose = [sum(init_pos2_x) / 2, init_pos2_y[0], (init_pos2_z[0] + init_pos2_z[1]) / 2, 0, 0, 0, 1]

