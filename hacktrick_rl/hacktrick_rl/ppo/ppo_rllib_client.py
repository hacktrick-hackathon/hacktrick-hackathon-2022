import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#!/usr/bin/env python -W ignore::DeprecationWarning

# All imports except rllib
import argparse, os, sys
from hacktrick_ai_py.agents.benchmarking import AgentEvaluator
import numpy as np

# environment variable that tells us whether this code is running on the server or not
LOCAL_TESTING = os.getenv('RUN_ENV', 'production') == 'local'

# Sacred setup (must be before rllib imports)
from sacred import Experiment
ex = Experiment("PPO RLLib")

# Necessary work-around to make sacred pickling compatible with rllib
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Slack notification configuration
from sacred.observers import SlackObserver
if os.path.exists('slack.json') and not LOCAL_TESTING:
    slack_obs = SlackObserver.from_config('slack.json')
    ex.observers.append(slack_obs)

    # Necessary for capturing stdout in multiprocessing setting
    SETTINGS.CAPTURE_MODE = 'sys'

# rllib and rllib-dependent imports
# Note: tensorflow and tensorflow dependent imports must also come after rllib imports
# This is because rllib disables eager execution. Otherwise, it must be manually disabled
import ray
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo.ppo import PPOTrainer
from hacktrick_rl.ppo.ppo_rllib import RllibPPOModel
from hacktrick_rl.rllib.rllib import HacktrickMultiAgent, save_trainer, gen_trainer_from_params


###################### Temp Documentation #######################
#   run the following command in order to train a PPO self-play #
#   agent with the static parameters listed in my_config        #
#                                                               #
#   python ppo_rllib_client.py                                  #
#                                                               #
#   In order to view the results of training, run the following #
#   command                                                     #
#                                                               #
#   tensorboard --log-dir ~/ray_results/                        #
#                                                               #
#################################################################

# Dummy wrapper to pass rllib type checks
def _env_creator(env_config):
    # Re-import required here to work with serialization
    from hacktrick_rl.rllib.rllib import HacktrickMultiAgent 
    return HacktrickMultiAgent.from_config(env_config)

@ex.config
def my_config():
    ### Model params ###

    # Whether the model is for single or collaborative training
    mode = 'single'

    # Whether dense reward should come from potential function or not
    use_phi = True

    # Base model params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3

    # whether to use D2RL https://arxiv.org/pdf/2010.09163.pdf (concatenation the result of last conv layer to each hidden layer);
    D2RL = False
    ### Training Params ###

    num_workers = 1 if not LOCAL_TESTING else 1

    # list of all random seeds to use for experiments, used to reproduce results
    seeds = [0]

    # Placeholder for random for current trial
    seed = 2229

    # Number of gpus the central driver should use
    num_gpus = 0 if LOCAL_TESTING else 1

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    train_batch_size = 12000 if not LOCAL_TESTING else 800

    # size of minibatches we divide up each batch into before
    # performing gradient steps
    sgd_minibatch_size = 2000 if not LOCAL_TESTING else 800

    # Rollout length
    rollout_fragment_length = 400
    
    # Whether all PPO agents should share the same policy network
    shared_policy = True

    # Number of training iterations to run
    num_training_iters = 500 if not LOCAL_TESTING else 2

    # Stepsize of SGD.
    lr = 5e-4

    # Learning rate schedule.
    lr_schedule = None

    # If specified, clip the global norm of gradients by this amount
    grad_clip = 0.1

    # Discount factor
    gamma = 0.99

    # Exponential decay factor for GAE (how much weight to put on monte carlo samples)
    # Reference: https://arxiv.org/pdf/1506.02438.pdf
    lmbda = 0.98

    # Whether the value function shares layers with the policy model
    vf_share_layers = True

    # How much the loss of the value network is weighted in overall loss
    vf_loss_coeff = 1e-4

    # Entropy bonus coefficient, will anneal linearly from _start to _end over _horizon steps
    entropy_coeff_start = 0.2
    entropy_coeff_end = 1e-2
    entropy_coeff_horizon = 3e6

    # Initial coefficient for KL divergence.
    kl_coeff = 0.2

    # PPO clipping factor
    clip_param = 0.05

    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    num_sgd_iter = 8 if not LOCAL_TESTING else 1

    # How many trainind iterations (calls to trainer.train()) to run before saving model checkpoint
    save_freq = 25

    # How many training iterations to run between each evaluation
    evaluation_interval = 50 if not LOCAL_TESTING else 1

    # How many timesteps should be in an evaluation episode
    evaluation_ep_length = 400

    # Number of games to simulation each evaluation
    evaluation_num_games = 1

    # Whether to display rollouts in evaluation
    evaluation_display = False

    # Where to log the ray dashboard stats
    temp_dir = os.path.join(os.path.abspath(os.sep), "tmp", "ray_tmp")

    # Where to store model checkpoints and training stats
    results_dir = DEFAULT_RESULTS_DIR

    # Whether tensorflow should execute eagerly or not
    eager = False

    # Whether to log training progress and debugging info
    verbose = True


    ### BC Params ### Kept only for backward compatability
    # path to pickled policy model for behavior cloning
    bc_model_dir = None

    # Whether bc agents should return action logit argmax or sample
    bc_stochastic = True



    ### Environment Params ###
    # Which hacktrick level to use
    layout_name = "cramped_room"

    # all_layout_names = '_'.join(layout_names)

    # Name of directory to store training results in (stored in ~/ray_results/<experiment_name>)

    params_str = str(use_phi) + "_nw=%d_vf=%f_es=%f_en=%f_kl=%f" % (
        num_workers,
        vf_loss_coeff,
        entropy_coeff_start,
        entropy_coeff_end,
        kl_coeff
    )

    experiment_name = "{0}_{1}_{2}".format("PPO", layout_name, params_str)

    # Rewards the agent will receive for intermediate actions
    rew_shaping_params = {
        "PLACEMENT_IN_CONSTRUCTION_SITE_REW": 3,
        "CONTAINER_PICKUP_REWARD": 3,
        "SOLARLAB_PICKUP_REWARD": 5,
        "CONTAINER_DISP_DISTANCE_REW": 0,
        "CONSTRUCTION_SITE_DISTANCE_REW": 0,
        "SOLARLAB_DISTANCE_REW": 0
    }

    # Max episode length
    horizon = 400

    # Constant by which shaped rewards are multiplied by when calculating total reward
    reward_shaping_factor = 1.0

    # Linearly anneal the reward shaping factor such that it reaches zero after this number of timesteps
    reward_shaping_horizon = 2.5e6

    # Kept only for backward compatability
    bc_schedule = HacktrickMultiAgent.self_play_bc_schedule


    # To be passed into rl-lib model/custom_options config
    model_params = {
        "NUM_HIDDEN_LAYERS" : NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS" : SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS" : NUM_FILTERS,
        "NUM_CONV_LAYERS" : NUM_CONV_LAYERS,
        "D2RL": D2RL
    }

    # to be passed into the rllib.PPOTrainer class
    training_params = {
        "num_workers" : num_workers,
        "train_batch_size" : train_batch_size,
        "sgd_minibatch_size" : sgd_minibatch_size,
        "rollout_fragment_length" : rollout_fragment_length,
        "num_sgd_iter" : num_sgd_iter,
        "lr" : lr,
        "lr_schedule" : lr_schedule,
        "grad_clip" : grad_clip,
        "gamma" : gamma,
        "lambda" : lmbda,
        "vf_share_layers" : vf_share_layers,
        "vf_loss_coeff" : vf_loss_coeff,
        "kl_coeff" : kl_coeff,
        "clip_param" : clip_param,
        "num_gpus" : num_gpus,
        "seed" : seed,
        "evaluation_interval" : evaluation_interval,
        "entropy_coeff_schedule" : [(0, entropy_coeff_start), (entropy_coeff_horizon, entropy_coeff_end)],
        "eager" : eager,
        "log_level" : "WARN" if verbose else "ERROR"
    }

    # To be passed into AgentEvaluator constructor and _evaluate function
    evaluation_params = {
        "ep_length" : evaluation_ep_length,
        "num_games" : evaluation_num_games,
        "display" : evaluation_display,
        "mode" : mode
    }


    environment_params = {
        # To be passed into HacktrickGridWorld constructor

        "mdp_params" : {
            "layout_name": layout_name,
            "rew_shaping_params": rew_shaping_params
        },
        # To be passed into HacktrickEnv constructor
        "env_params" : {
            "horizon" : horizon
        },

        # To be passed into HacktrickMultiAgent constructor
        "multi_agent_params" : {
            "reward_shaping_factor" : reward_shaping_factor,
            "reward_shaping_horizon" : reward_shaping_horizon,
            "use_phi" : use_phi,
            "bc_schedule" : bc_schedule,
            "mode" : mode
        }
    }

    bc_params = {
        "bc_policy_cls" : None, #// BehaviorCloningPolicy,
        "bc_config" : {
            "model_dir" : bc_model_dir,
            "stochastic" : bc_stochastic,
            "eager" : eager
        }
    }

    ray_params = {
        "custom_model_id" : "MyPPOModel",
        "custom_model_cls" : RllibPPOModel,
        "temp_dir" : temp_dir,
        "env_creator" : _env_creator
    }

    params = {
        "model_params" : model_params,
        "training_params" : training_params,
        "environment_params" : environment_params,
        "bc_params" : bc_params,
        "shared_policy" : shared_policy,
        "num_training_iters" : num_training_iters,
        "evaluation_params" : evaluation_params,
        "experiment_name" : experiment_name,
        "save_every" : save_freq,
        "seeds" : seeds,
        "results_dir" : results_dir,
        "ray_params" : ray_params,
        "verbose" : verbose
    }


def run(params):
    # Retrieve the tune.Trainable object that is used for the experiment
    trainer = gen_trainer_from_params(params)

    # Object to store training results in
    result = {}

    # Training loop
    for i in range(params['num_training_iters']):
        if params['verbose']:
            print("Starting training iteration", i)
        result = trainer.train()

        if i % params['save_every'] == 0:
            save_path = save_trainer(trainer, params)
            if params['verbose']:
                print("saved trainer at", save_path)

    # Save the state of the experiment at end
    save_path = save_trainer(trainer, params)
    if params['verbose']:
        print("saved trainer at", save_path)

    return result


@ex.automain
def main(params):
    # List of each random seed to run
    seeds = params['seeds']
    del params['seeds']

    # List to store results dicts (to be passed to sacred slack observer)
    results = []

    # Train an agent to completion for each random seed specified
    for seed in seeds:
        # Override the seed
        params['training_params']['seed'] = seed

        # Do the thing
        result = run(params)
        results.append(result)

    # Return value gets sent to our slack observer for notification
    average_sparse_reward = np.mean([res['custom_metrics']['sparse_reward_mean'] for res in results])
    average_episode_reward = np.mean([res['episode_reward_mean'] for res in results])
    return { "average_sparse_reward" : average_sparse_reward, "average_total_reward" : average_episode_reward }