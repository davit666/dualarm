

class TrainConfig:
    # env_name = "asynchronous_goal_reaching"
    env_name = "goal_reaching"

    load_model = False
    load_model_path = 'test_models/' + env_name + '/1220/wided_with_obj'
    load_model_path = 'test_models/' + env_name + '/1223/pretrained_widden_net_0.06_coll_with_obj'
    load_model_path = 'test_models/' + env_name + '/1230/multi_policy_0.05_coll_with_obj'
    load_model_path = 'test_models/' + env_name + '/0110/box_input'


    load_model_path = 'test_models/' + env_name + '/0221/3'
    load_model_path = 'test_models/' + env_name + '/0226/test_srr1'
    load_model_path = 'test_models/' + env_name + '/0303/repeat_2'
    load_model_path = 'test_models/' + env_name + '/0305/cut_repeat_2'
    load_model_path = 'test_models/' + env_name + '/0305/cut_repeat_2'
    load_model_path = 'test_models/' + env_name + '/0307/cut_repeat_2'
    load_model_path = 'test_models/' + env_name + '/0318/current_target'
    # load_model_path = 'test_models/' + env_name + '/0318/both_target'
    load_model_path = 'test_models/' + env_name + '/0328/chain2-4'
    # load_model_path = 'test_models/' + env_name + '/0328/cycle3'
    load_model_path = 'test_models/' + env_name + '/0329/tri_obs_tri_reward_1'
    load_model_path = 'test_models/' + env_name + '/0402/64'
    # load_model_path = 'test_models/' + env_name + '/0402/512'
    # load_model_path = 'test_models/' + env_name + '/0404/common_obs_64'
    load_model_path = 'test_models/' + env_name + '/0518/js_control_ee_review_field_r'
    load_model_path = 'test_models/' + env_name + '/0523/j_c_e_r_l_f'
    load_model_path = 'test_models/' + env_name + '/0829/ee_950'
    load_model_path = 'test_models/' + env_name + '/0829/js_950'

    load_model_path = 'test_models/' + env_name + '/0831/common2'

    load_model_path = 'test_models/' + env_name + '/0831/bb2'
    load_model_path = 'test_models/' + env_name + '/0919/policy_0.2_gap'

    ######## model ########
    custom_network = False
    dict_obs = False

    model_date = env_name + "/monitoring/0922_ee/best_policy_0.05_schrodinger_obj/"
    model_name = "actor_64_32cpu_lr_linear_1e-3_ratio_weight_1_coll_dist_0.05_keep_bonus_when_success/"
    alg_name = "PPO"
    reward_type = "delta_dist_field_with_sparse_reward"#"negative_dist_field_with_sparse_reward" #"delta_dist_&_overlap_area_ratio_with_sparse_reward"  #"delta_dist_&_cutting_area_ratio_with_sparse_reward" #
    obs_type = "common_obs_with_obj_bb"#"common_obs" # "common_obs_with_links_dist"  #"obs_with_triangle_features"  #
    action_type = "ee" # "js_control_ee_reward" #

    ######## save & log ########
    log_path = "../../log_datas/"
    save_path = "../models_saved/"
    model_save_freq = 1e6

    ######## cpu ########
    num_cpu = 32
    ######## lr scheduler ########
    use_lr_scheduler = True
    ######## parameter ########
    learning_rate = 1e-3
    n_steps = 2048
    batch_size = 2048
    n_epochs = 10

    total_timesteps = 3e8


class EnvConfig:
    ######## type ########
    #basic
    useInverseKinematics = True
    normalize_pose = True

    #key status
    move_with_obj = True
    fixed_obj_shape = True
    obj_shape_type = "task"#"box" # "random" #
    keep_bonus_when_success = True
    stepback_if_collide = False

    #for img
    fill_triangle = True
    #for sequence task
    sequence_task = False
    #for historic observation
    observation_history_length = 5
    ######## setup ########
    partsBaseSize = [4.8 / 2, 1.65 / 2, 0.35 / 2]
    partsBasePos = [-0, 0, 0 + partsBaseSize[2]]
    partsBaseOrn = [0, 0, 0, 1]
    partsBaseColor = [1, 1, 1, 1]

    beltBaseSize = [0.7 / 2, 4 / 2, 0.05 / 2]
    beltBasePos = [1.5 + beltBaseSize[0], 0, 0.65 + beltBaseSize[2]]
    beltBaseOrn = [0, 0, 0, 1]
    beltBaseColor = [0.75, 0.75, 0.75, 1]

    ######## reward ########
    reward_scale = 5#1    #5

    delta_pos_weight = 1
    delta_orn_weight = 0.1

    coll_penalty_obj = 1#100  #1
    coll_penalty_robot = 1  # 0.25

    reach_bonus_pos = 0.1#100   # 0.1
    reach_bonus_orn = 0.1#100   # 0.1

    delta_area_ratio_weight = 1

    joint_success_bonus = 1#300   # 1

    ######## normalization ########
    action_scale = 0.05

    ######## goal ########
    success_dist_threshold_pos = None
    success_dist_threshold_orn = None

    success_dist_threshold = 0.10
    ######## collision ########
    safety_dist_threshold = 0.05

    ######## render ########
    show_ball_freq = 50

    ######## model ########
    # reward_type = "delta_dist_with_sparse_reward"
    # obs_type = "js+EE+goal"
    ######## triangle ########
    minimum_triangle_area = 1e-5


def load_config():
    ######## train config ########
    train_config = {}
    T = TrainConfig()

    train_config['dict_obs'] = T.dict_obs
    train_config['custom_network'] = T.custom_network
    train_config['env_name'] = T.env_name
    train_config['load_model'] = T.load_model
    train_config['load_model_path'] = T.load_model_path

    train_config['model_date'] = T.model_date
    train_config['model_name'] = T.model_name
    train_config['alg_name'] = T.alg_name
    train_config['reward_type'] = T.reward_type
    train_config['obs_type'] = T.obs_type
    train_config['action_type'] = T.action_type

    train_config['log_path'] = T.log_path
    train_config['save_path'] = T.save_path
    train_config['model_save_freq'] = T.model_save_freq

    train_config['num_cpu'] = T.num_cpu

    train_config['use_lr_scheduler'] = T.use_lr_scheduler
    train_config['learning_rate'] = T.learning_rate
    train_config['n_steps'] = T.n_steps
    train_config['batch_size'] = T.batch_size
    train_config['n_epochs'] = T.n_epochs

    train_config['total_timesteps'] = T.total_timesteps

    ########env config ########
    env_config = {}
    E = EnvConfig()

    env_config['useInverseKinematics'] = E.useInverseKinematics
    env_config['normalize_pose'] = E.normalize_pose

    env_config['move_with_obj'] = E.move_with_obj
    env_config['fixed_obj_shape'] = E.fixed_obj_shape
    env_config['obj_shape_type'] = E.obj_shape_type
    env_config['keep_bonus_when_success'] = E.keep_bonus_when_success
    env_config['stepback_if_collide'] = E.stepback_if_collide

    env_config['sequence_task'] = E.sequence_task
    env_config['fill_triangle'] = E.fill_triangle
    env_config['observation_history_length'] = E.observation_history_length

    env_config['partsBaseSize'] = E.partsBaseSize
    env_config['partsBasePos'] = E.partsBasePos
    env_config['partsBaseOrn'] = E.partsBaseOrn
    env_config['partsBaseColor'] = E.partsBaseColor
    env_config['beltBaseSize'] = E.beltBaseSize
    env_config['beltBasePos'] = E.beltBasePos
    env_config['beltBaseOrn'] = E.beltBaseOrn
    env_config['beltBaseColor'] = E.beltBaseColor

    env_config['delta_pos_weight'] = E.delta_pos_weight
    env_config['delta_orn_weight'] = E.delta_orn_weight
    env_config['coll_penalty_obj'] = E.coll_penalty_obj
    env_config['coll_penalty_robot'] = E.coll_penalty_robot
    env_config['reach_bonus_pos'] = E.reach_bonus_pos
    env_config['reach_bonus_orn'] = E.reach_bonus_orn
    env_config['joint_success_bonus'] = E.joint_success_bonus
    env_config['delta_area_ratio_weight'] = E.delta_area_ratio_weight

    env_config['action_scale'] = E.action_scale

    env_config['success_dist_threshold_pos'] = E.success_dist_threshold_pos
    env_config['success_dist_threshold_orn'] = E.success_dist_threshold_orn
    env_config['success_dist_threshold'] = E.success_dist_threshold
    env_config['safety_dist_threshold'] = E.safety_dist_threshold

    env_config['show_ball_freq'] = E.show_ball_freq
    env_config['reward_scale'] = E.reward_scale

    env_config['minimum_triangle_area'] = E.minimum_triangle_area
    return train_config, env_config




class TaskConfig:
    env_name = "task"
    load_model = False
    task_allocator_load_model_path = 'test_models/' + 'task' + '/0311/1M'
    task_allocator_load_model_path = 'test_models/' + 'task' + '/0313/penalty-1'
    task_allocator_load_model_path = 'test_models/' + 'task' + '/0311/frag_50'
    task_allocator_load_model_path = 'test_models/' + 'task' + '/0314/set_0.1_frag_5_7m'
    ######## model ########
    custom_network = "basic"
    dict_obs = False

    model_date = env_name + "/monitoring/0921/baseline/test"
    model_name = "test/lr_linear_1e-3_cpu16"
    alg_name = "PPO"


    motion_planner_reward_type = "delta_dist_&_cutting_area_ratio_with_sparse_reward" #"delta_dist_with_sparse_reward" #"delta_dist_&_overlap_area_ratio_with_sparse_reward"  #
    motion_planner_obs_type = "common_obs_with_obj_bb"
    motion_planner_action_type = "ee"

    task_allocator_reward_type = "delta_dist_field_with_sparse_reward"
    task_allocator_obs_type = "feature_observation"
    task_allocator_action_type = "MultiDiscrete"

    ######## save & log ########
    log_path = "../../log_datas/"
    save_path = "../models_saved/"
    model_save_freq = 5e4

    ######## cpu ########
    num_cpu = 16
    ######## lr scheduler ########
    use_lr_scheduler = True
    ######## parameter ########
    learning_rate = 1e-3
    n_steps = 2048
    batch_size = 2048
    n_epochs = 10
    total_timesteps = 2e7

    buffer_size = 1000000

    ######## task parameter########

    parts_num = 4
    fragment_length = 5
    maxSteps = 300
    selected_motion_planner_policy = False
    motion_planner_load_model_path = 'test_models/' + 'goal_reaching' + '/0919/policy_0.2_gap'
    motion_planner_load_model_path = 'test_models/' + 'goal_reaching' + '/0921/policy_0.1_gap'
    ######## para retune ########
    if alg_name == "SAC":
        action_type = "Box"
        batch_size = 2048


def load_task_config():
    ######## train config ########
    task_config = {}
    TA = TaskConfig()
    task_config['env_name'] = TA.env_name
    task_config['load_model'] = TA.load_model
    task_config['task_allocator_load_model_path'] = TA.task_allocator_load_model_path

    task_config['custom_network'] = TA.custom_network
    task_config['dict_obs'] = TA.dict_obs

    task_config['model_date'] = TA.model_date
    task_config['model_name'] = TA.model_name
    task_config['alg_name'] = TA.alg_name

    task_config['task_allocator_reward_type'] = TA.task_allocator_reward_type
    task_config['task_allocator_obs_type'] = TA.task_allocator_obs_type
    task_config['task_allocator_action_type'] = TA.task_allocator_action_type
    task_config['motion_planner_reward_type'] = TA.motion_planner_reward_type
    task_config['motion_planner_obs_type'] = TA.motion_planner_obs_type
    task_config['motion_planner_action_type'] = TA.motion_planner_action_type


    task_config['log_path'] = TA.log_path
    task_config['save_path'] = TA.save_path
    task_config['model_save_freq'] = TA.model_save_freq
    task_config['num_cpu'] = TA.num_cpu
    task_config['use_lr_scheduler'] = TA.use_lr_scheduler
    task_config['learning_rate'] = TA.learning_rate
    task_config['n_steps'] = TA.n_steps
    task_config['batch_size'] = TA.batch_size
    task_config['n_epochs'] = TA.n_epochs
    task_config['total_timesteps'] = TA.total_timesteps
    task_config['buffer_size'] = TA.buffer_size

    task_config['parts_num'] = TA.parts_num
    task_config['fragment_length'] = TA.fragment_length
    task_config['maxSteps'] = TA.maxSteps
    task_config['selected_motion_planner_policy'] = TA.selected_motion_planner_policy
    task_config['motion_planner_load_model_path'] = TA.motion_planner_load_model_path

    return task_config












