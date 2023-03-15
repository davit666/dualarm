import time


class TaskConfig:
    env_name = "pseudo_task"

    load_model = False
    load_model_path = 'test_models/' + env_name
    # baselines from 0224
    # load_model_path += '/0224/baseline-reward1_4parts_20M_linear_lr_3e-4/RT_NP_CM/NET0220-baseline/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-02-24-17-06-29'
    # load_model_path += '/0224/baseline-reward1_4parts_20M_linear_lr_3e-4/RT_P_CM/NET0220-baseline/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-02-24-17-04-53'

    # ablation for obs in 0224
    # load_model_path += '/0224/no_node_static_obs-reward1_4parts_20M_linear_lr_3e-4/RT_NP_CM/NET0220-baseline/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-02-24-17-09-01'
    # load_model_path += '/0224/no_edge_mask_obs-reward1_4parts_20M_linear_lr_3e-4/RT_NP_CM/NET0220-baseline/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-02-24-17-11-17'
    # load_model_path += '/0224/short_coop_edge_obs-reward1_4parts_20M_linear_lr_3e-4/RT_NP_CM/NET0220-baseline/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-02-24-17-13-53'

    # experiment in 0228 for reward design
    # load_model_path += '/0228/baseline-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-02-28-17-02-34'
    # load_model_path += '/0228/no-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-02-28-17-04-23'
    # load_model_path += '/0228/fix-penalty-no-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-02-28-17-07-54'
    # load_model_path += '/0228/no-step-no-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-02-28-17-06-38'

    # ablation for short train (10M) in 0228
    # load_model_path += '/0301/short-train-test-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-01-11-59-54'
    # load_model_path += '/0301/short-train-test-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-01-12-00-22'

    # ablation for minimum obs and penalty tuning
    # load_model_path += '/0228/large-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220-minimum_obs/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-02-28-19-56-39'
    # load_model_path += '/0228/middle-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220-minimum_obs/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-02-28-19-57-16'
    # load_model_path += '/0228/small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220-minimum_obs/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-02-28-19-57-47'

    # ablation for reward shaping 2 in 0301
    # load_model_path += '/0301/small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-01-19-34-32'
    # load_model_path += '/0301/tiny-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-01-19-35-04'
    # load_model_path += '/0301/small-bonus-no-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-01-19-35-58'
    # load_model_path += '/0301/no-bonus-no-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-01-19-37-13'

    # ablation for reward shaping 3 in 0303
    # load_model_path += '/0302/small-bonus-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-02-15-11-11'
    # load_model_path += '/0302/no-step-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-02-15-12-18'
    # load_model_path += '/0302/no-bomus-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-02-16-11-02'
    # load_model_path += '/0302/no-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-02-16-12-08'

    # ablation for partial predictive in 0303
    # load_model_path += '/0302/only_predict_cost-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-02-19-29-07'
    # load_model_path += '/0302/only_predict_mask-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-02-19-29-54'

    # reward ablation 4 in 0303
    # load_model_path += '/0303/reward-ablaion-4/no-step-no-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-03-09-58-34'
    # load_model_path += '/0303/reward-ablaion-4/normal-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-03-10-00-09'
    # load_model_path += '/0303/reward-ablaion-4/normal-bonus-normal-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-03-10-55-18'
    # load_model_path += '/0303/reward-ablaion-4/normal-bonus-no-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-03-11-01-07'

    # ablation input in 0304
    # load_model_path += '/0303/input-ablation/baselines-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_NP_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-03-16-40-38'

    # load_model_path += '/0303/input-ablation/least-encoder-obs-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-03-16-33-02'
    # load_model_path += '/0303/input-ablation/least-decoder-obs-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-03-16-33-57'

    # ablation mask constraint in 0304
    # load_model_path += '/0303/mask-constraint/neighbour_index-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_NP_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-04-19-45-19'
    # load_model_path += '/0303/mask-constraint/pose_overlap-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_NP_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-04-19-46-04'
    # load_model_path += '/0303/mask-constraint/triangle_overlap-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_NP_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-04-19-46-43'
    # load_model_path += '/0303/mask-constraint/random_number-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_NP_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-04-19-47-43'

    # ablation input in 0306 & 0308
    # load_model_path += '/0306/input-ablation/encoder-no-static-obs-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-06-18-43-19'
    # load_model_path += '/0306/input-ablation/encoder-no-dynamic-obs-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-06-18-43-38'
    # load_model_path += '/0306/input-ablation/decoder-short-obs-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-06-18-46-39'
    # load_model_path += '/0308/input-ablation/encoder-no-static-obs-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-08-17-59-44'
    # load_model_path += '/0308/input-ablation/encoder-short-robot-obs-small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-08-18-01-09'

    # baselines middle task in 0307
    # load_model_path += '/0307/middle-num-tasks/small-penalty-reward_10parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-06-10-19-34'
    # load_model_path += '/0307/middle-num-tasks/small-penalty-reward_10parts_20M_lr_3e-4_linear/RT_NP_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-06-10-21-15'

    # ablation middle task reward in 0307
    # load_model_path += '/0307/middle-num-tasks/normal-bonus-normal-penalty-reward_10parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-07-09-30-45'
    # load_model_path += '/0307/middle-num-tasks/normal-bonus-no-penalty-reward_10parts_20M_lr_3e-4_linear/RT_P_CM/NET0220/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-07-09-31-25'

    # ablation net arch in 0309-0312
    # load_model_path += '/0309/Net-Arch/small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/Flatten/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-09-16-29-43'
    # load_model_path += '/0309/Net-Arch/small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/Only_Coop/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-09-16-30-53'
    # load_model_path += '/0309/Net-Arch/small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/Only_Decoder/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-09-16-35-11'
    # load_model_path += '/0312/Net-Arch/small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/Without_TaskEdge/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-12-20-00-17'
    load_model_path += '/0312/Net-Arch/small-penalty-reward_6parts_20M_lr_3e-4_linear/RT_P_CM/Upgrade_Output/PPO_negative_normalized_cost_with_sparse_success_bonus_norm_ee_only_Discrete_2023-03-12-20-02-00'

    load_model_path += "/model_at_the_end" #  "/model_at_step_17500000"  # "/model_at_step_18999696"  #
    ######## model ########
    use_custom_network = False

    model_date = env_name + "/monitoringx/" + time.strftime("%m%d")
    custom_network_type = "Upgrade_Output"  # "SCAttNE_3_layer_cat_coop_edge"#"custom_self_cross_attention_edge"#"custom_flatten_all"#"custom_flatten_node"#
    model_name = "Net-Arch/small-penalty-reward_6parts_20M_lr_3e-4_linear" \
                 + "/RT_P_CM"

    alg_name = "PPO"
    task_allocator_reward_type = "negative_normalized_cost_with_sparse_success_bonus"
    task_allocator_obs_type = "common_dict_obs"
    task_allocator_action_type = "Discrete"  # "Box"#"MultiDiscrete"

    use_prediction_model = True
    predict_content = ""  #"cost_only" #"both"#  "mask_only" #

    use_mask_constraint = False
    mask_constraint = "random_number"   #"neighbour_index" # "pose_overlap"# "triangle_overlap" # "random_number"

    ####### prediction model ########
    #### define input output type to use
    cost_type = "coord_steps"  # "coord_steps"
    obs_type = "norm_ee_only"
    #### set loading path of prediction model
    cost_model_path = "../../generated_datas/good_models/cost/1203/1010_task_datas_100_epochs/norm_ee_only_succ_only_predict_step/512-512-512_ce_adam_rl1e-3_batch_512/2022-12-02-13-13-04/model_saved/2022-12-02-14-04-40.pth"
    mask_model_path = "../../generated_datas/good_models/mask/1203/1024_with_failure_task_datas_100_epochs/norm_ee_only_predict_succ/256-256-256_ce_adam_rl1e-3_batch_512/2022-12-02-11-47-26/model_saved/2022-12-02-12-29-48.pth"

    ######## env config ########
    part_num = 6

    fix_box_sample = True
    mask_done_task = False
    robot_done_freeze = True
    default_rest_pose = True

    task_type = "random"
    dynamic_task = False

    max_cost_const = 500
    global_success_bonus = 1
    reward_scale = 5

    partsBaseSize = [4.8 / 2, 1.65 / 2, 0.35 / 2]
    partsBasePos = [-0, 0, 0 + partsBaseSize[2]]
    partsBaseOrn = [0, 0, 0, 1]
    partsBaseColor = [1, 1, 1, 1]

    beltBaseSize = [0.7 / 2, 4 / 2, 0.05 / 2]
    beltBasePos = [1.5 + beltBaseSize[0], 0, 0.65 + beltBaseSize[2]]
    beltBaseOrn = [0, 0, 0, 1]
    beltBaseColor = [0.75, 0.75, 0.75, 1]

    ######## save & log ########
    log_path = "../../../../log_datas/"
    save_path = "../models_saved/"
    model_save_freq = 5e5

    ######## cpu ########
    num_cpu = 24
    ######## lr scheduler ########
    use_lr_scheduler = True
    ######## parameter ########
    learning_rate = 3e-4
    n_steps = 2048
    batch_size = 512
    n_epochs = 10

    total_timesteps = 2e7


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

    task_config['use_prediction_model'] = T.use_prediction_model
    task_config['predict_content'] = T.predict_content
    task_config['use_mask_constraint'] = T.use_mask_constraint
    task_config['mask_constraint'] = T.mask_constraint
    task_config['default_rest_pose'] = T.default_rest_pose

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
    task_config['fix_box_sample'] = T.fix_box_sample
    task_config['mask_done_task'] = T.mask_done_task
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
