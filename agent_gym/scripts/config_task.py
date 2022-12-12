import time

class TaskConfig:
    env_name = "pseudo_task"

    load_model = False
    load_model_path = 'test_models/' + env_name
    load_model_path  += '/1212/test1_not_terminate_when_robot_done_with_check_custom_flatten_node/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_MultiDiscrete_2022-12-12-09-13-20/'
    load_model_path += "test1_not_terminate_when_robot_done_with_check_at_the_end"
    ######## model ########
    use_custom_network = True

    model_date = env_name + "/monitoring/" + time.strftime("%m%d")
    custom_network_type = "custom_flatten_all"#"custom_flatten_node"#
    model_name = "real_test/part_done_terminate_10M"

    alg_name = "PPO"
    task_allocator_reward_type = "negative_normalized_cost_with_sparse_success_bonus"
    task_allocator_obs_type = "common_dict_obs"
    task_allocator_action_type = "MultiDiscrete"

    ####### prediction model ########
    #### define input output type to use
    cost_type = "coord_steps"  # "coord_steps"
    obs_type = "norm_ee_only"
    #### set loading path of prediction model
    cost_model_path = "../../generated_datas/good_models/cost/1203/1010_task_datas_100_epochs/norm_ee_only_succ_only_predict_step/512-512-512_ce_adam_rl1e-3_batch_512/2022-12-02-13-13-04/model_saved/2022-12-02-14-04-40.pth"
    mask_model_path = "../../generated_datas/good_models/mask/1203/1024_with_failure_task_datas_100_epochs/norm_ee_only_predict_succ/256-256-256_ce_adam_rl1e-3_batch_512/2022-12-02-11-47-26/model_saved/2022-12-02-12-29-48.pth"

    ######## env config ########
    part_num = 6
    robot_done_freeze = True

    task_type = "random"
    dynamic_task = False

    max_cost_const = 1000
    global_success_bonus = 10
    reward_scale = 1

    partsBaseSize = [4.8 / 2, 1.65 / 2, 0.35 / 2]
    partsBasePos = [-0, 0, 0 + partsBaseSize[2]]
    partsBaseOrn = [0, 0, 0, 1]
    partsBaseColor = [1, 1, 1, 1]

    beltBaseSize = [0.7 / 2, 4 / 2, 0.05 / 2]
    beltBasePos = [1.5 + beltBaseSize[0], 0, 0.65 + beltBaseSize[2]]
    beltBaseOrn = [0, 0, 0, 1]
    beltBaseColor = [0.75, 0.75, 0.75, 1]


    ######## save & log ########
    log_path = "../../log_datas/"
    save_path = "../models_saved/"
    model_save_freq = 1e6

    ######## cpu ########
    num_cpu = 16
    ######## lr scheduler ########
    use_lr_scheduler = True
    ######## parameter ########
    learning_rate = 1e-3
    n_steps = 2048
    batch_size = 2048
    n_epochs = 10

    total_timesteps = 1e7

def load_config():
    ######## train config ########
    task_config = {}
    T = TaskConfig()

    task_config['env_name'] = T.env_name
    task_config['load_model'] = T.load_model
    task_config['load_model_path'] = T.load_model_path

    task_config['use_custom_network'] = T.use_custom_network

    task_config['model_date'] = T.model_date
    task_config['custom_network_type'] = T.custom_network_type
    task_config['model_name'] = T.model_name
    task_config['alg_name'] = T.alg_name
    task_config['task_allocator_reward_type'] = T.task_allocator_reward_type
    task_config['task_allocator_obs_type'] = T.task_allocator_obs_type
    task_config['task_allocator_action_type'] = T.task_allocator_action_type

    task_config['log_path'] = T.log_path
    task_config['save_path'] = T.save_path
    task_config['model_save_freq'] = T.model_save_freq

    task_config['num_cpu'] = T.num_cpu

    task_config['use_lr_scheduler'] = T.use_lr_scheduler
    task_config['learning_rate'] = T.learning_rate
    task_config['n_steps'] = T.n_steps
    task_config['batch_size'] = T.batch_size
    task_config['n_epochs'] = T.n_epochs

    task_config['total_timesteps'] = T.total_timesteps

    ####### prediction model ########

    task_config['cost_type'] = T.cost_type
    task_config['obs_type'] = T.obs_type
    task_config['cost_model_path'] = T.cost_model_path
    task_config['mask_model_path'] = T.mask_model_path
    ######## env config ########
    task_config['part_num'] = T.part_num
    task_config['robot_done_freeze'] = T.robot_done_freeze
    task_config['task_type'] = T.task_type
    task_config['dynamic_task'] = T.dynamic_task
    task_config['max_cost_const'] = T.max_cost_const
    task_config['global_success_bonus'] = T.global_success_bonus
    task_config['reward_scale'] = T.reward_scale
    task_config['partsBaseSize'] = T.partsBaseSize
    task_config['partsBasePos'] = T.partsBasePos
    task_config['partsBaseOrn'] = T.partsBaseOrn
    task_config['partsBaseColor'] = T.partsBaseColor
    task_config['beltBaseSize'] = T.beltBaseSize
    task_config['beltBasePos'] = T.beltBasePos
    task_config['beltBaseOrn'] = T.beltBaseOrn
    task_config['beltBaseColor'] = T.beltBaseColor

    return task_config



