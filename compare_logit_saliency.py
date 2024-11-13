import copy
from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel, IdentityModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels
from imitation_rl import decompose_policy

import os, time, yaml, argparse
import gym
from procgen import ProcgenGym3Env
import random
import torch

from PIL import Image
from helper_local import latest_model_path
import torchvision as tv

from gym3 import ViewerWrapper, VideoRecorderWrapper, ToBaselinesVecEnv

import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_video_from_array(video_array, output_file, fps=30):
    """
    Converts a 4D numpy array (num_frames, height, width, channels) into an .mp4 file.

    Parameters:
    video_array (np.array): 4D array containing video frames (num_frames, height, width, 3).
    output_file (str): The output video file path.
    fps (int): Frames per second for the video.
    """
    # Get dimensions from the video array
    num_frames, height, width, channels = video_array.shape

    if channels != 3:
        raise ValueError("Expected an array with 3 channels (RGB), but got {} channels.".format(channels))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")#*'mp4v')  # 'mp4v' is for .mp4 format
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = video_array[i]

        # Ensure frame is in the right data type (uint8)
        if frame.dtype != np.uint8:
            frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8)  # Scale to 0-255 if needed

        out.write(frame)  # Write the frame to the video

    out.release()  # Release the video writer object
    print(f"Video saved as {output_file}")


def save_video_with_text(video_array, output_file, texts, fps=30, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Converts a 4D numpy array (num_frames, height, width, channels) into an .mp4 file with overlayed text.

    Parameters:
    video_array (np.array): 4D array containing video frames (num_frames, height, width, 3).
    output_file (str): The output video file path.
    texts (list): List of strings to overlay on each frame.
    fps (int): Frames per second for the video.
    font (int): OpenCV font type for the overlayed text.
    """
    # Get dimensions from the video array
    num_frames, height, width, channels = video_array.shape

    if channels != 3:
        raise ValueError("Expected an array with 3 channels (RGB), but got {} channels.".format(channels))

    if len(texts) != num_frames:
        raise ValueError("The length of the texts list must match the number of frames.")

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")#*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = video_array[i]

        # Ensure frame is in the right data type (uint8)
        if frame.dtype != np.uint8:
            frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8)

        # Add text to the frame
        text = texts[i]
        position = (50, 50)  # Position of the text (x, y)
        font_scale = 1
        color = (255, 255, 255)  # White text
        thickness = 2

        # Overlay the text on the frame
        cv2.putText(frame, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

        # Write the frame to the video
        out.write(frame)

    out.release()  # Release the video writer object
    print(f"Video with overlayed text saved as {output_file}")


def create_barchart_as_numpy(categories, values, width_pixels, height_pixels, dpi=100):
    plt.switch_backend('Agg')

    # # Sample data
    # categories = ['A', 'B', 'C', 'D']
    # values = [3, 7, 2, 5]

    # Calculate size in inches from pixels and DPI
    width_in = width_pixels / dpi
    height_in = height_pixels / dpi

    # Create the bar chart with specified figure size
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    ax.bar(categories, values)
    ax.set_title('Sample Bar Chart')
    ax.set_xlabel('Category')
    ax.set_ylabel('Values')

    # Draw the canvas to the figure
    fig.canvas.draw()

    # Convert canvas to a NumPy array
    img_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(height_pixels, width_pixels, 4)

    # Convert ARGB to RGBA for proper color channel order
    img_array = np.roll(img_array, 3, axis=2)

    plt.close(fig)  # Close the figure to free memory

    return img_array[..., :3]

# # Example usage:
# width_pixels = 640  # Desired width in pixels
# height_pixels = 480  # Desired height in pixels
# barchart_array = create_barchart_as_numpy(width_pixels, height_pixels, dpi=100)
# print(barchart_array.shape)  # Should print (480, 640, 4)


def main(args, barchart=False):


    exp_name = args.exp_name
    env_name = args.env_name
    start_level = args.start_level
    num_levels = args.num_levels
    distribution_mode = args.distribution_mode
    param_name = args.param_name
    device = args.device
    gpu_device = args.gpu_device
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    ####################
    ## HYPERPARAMETERS #
    ####################
    print('[LOADING HYPERPARAMETERS...]')
    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    for key, value in hyperparameters.items():
        print(key, ':', value)

    ############
    ## DEVICE ##
    ############
    if args.device == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #################
    ## ENVIRONMENT ##
    #################
    print('INITIALIZAING ENVIRONMENTS...')
    n_envs = 1

    def create_venv_render(args, hyperparameters, is_valid=False):
        venv = ProcgenGym3Env(num=n_envs,
                          env_name=args.env_name,
                          num_levels=0 if is_valid else args.num_levels,
                          start_level=0 if is_valid else args.start_level,
                          distribution_mode=args.distribution_mode,
                          num_threads=1,
                          render_mode="rgb_array",
                          random_percent=args.random_percent,
                          corruption_type=args.corruption_type,
                          corruption_severity=int(args.corruption_severity),
                          continue_after_coin=args.continue_after_coin,
                          )
        info_key = None if args.agent_view else "rgb"
        ob_key = "rgb" if args.agent_view else None
        if not args.noview:
            venv = ViewerWrapper(venv, tps=args.tps, info_key=info_key, ob_key=ob_key) # N.B. this line caused issues for me. I just commented it out, but it's uncommented in the pushed version in case it's just me (Lee).
        if args.vid_dir is not None:
            venv = VideoRecorderWrapper(venv, directory=args.vid_dir,
                                        info_key=info_key, ob_key=ob_key, fps=args.tps)
        venv = ToBaselinesVecEnv(venv)
        venv = VecExtractDictObs(venv, "rgb")
        normalize_rew = hyperparameters.get('normalize_rew', True)
        if normalize_rew:
            venv = VecNormalize(venv, ob=False) # normalizing returns, but not
            #the img frames
        venv = TransposeFrame(venv)
        venv = ScaledFloatFrame(venv)

        return venv
    n_steps = hyperparameters.get('n_steps', 256)

    #env = create_venv(args, hyperparameters)
    #env_valid = create_venv(args, hyperparameters, is_valid=True)
    env = create_venv_render(args, hyperparameters, is_valid=True)

    ############
    ## LOGGER ##
    ############
    print('INITIALIZAING LOGGER...')
    if args.logdir is None:
        logdir = 'procgen/' + env_name + '/' + exp_name + '/' + 'RENDER_seed' + '_' + \
                 str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join('logs', logdir)
    else:
        logdir = args.logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logdir_indiv_value = os.path.join(logdir, 'value_individual')
    if not (os.path.exists(logdir_indiv_value)) and args.save_value_individual:
        os.makedirs(logdir_indiv_value)
    logdir_saliency_value = os.path.join(logdir, 'value_saliency')
    if not (os.path.exists(logdir_saliency_value)) and args.value_saliency:
        os.makedirs(logdir_saliency_value)
    print(f'Logging to {logdir}')
    logger = Logger(n_envs, logdir)

    ###########
    ## MODEL ##
    ###########
    print('INTIALIZING MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = hyperparameters.get('architecture', 'impala')
    in_channels = observation_shape[0]
    action_space = env.action_space

    model, policy = create_policy(action_space, architecture, device, hyperparameters, in_channels)

    #############
    ## STORAGE ##
    #############
    print('INITIALIZAING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)
    #storage_valid = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)

    ###########
    ## AGENT ##
    ###########
    print('INTIALIZING AGENT...')
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError
    shifted_agent = AGENT(env, policy, logger, storage, device, num_checkpoints, **hyperparameters)

    shifted_agent.policy.load_state_dict(torch.load(args.shifted_agent_file, map_location=device)["model_state_dict"])
    shifted_agent.n_envs = n_envs

    _, unshifted_policy = create_policy(action_space, architecture, device, hyperparameters, in_channels)

    unshifted_agent = AGENT(env, unshifted_policy, logger, storage, device, num_checkpoints, **hyperparameters)

    unshifted_agent.policy.load_state_dict(torch.load(args.unshifted_agent_file, map_location=device)["model_state_dict"])
    unshifted_agent.n_envs = n_envs


    obs = shifted_agent.env.reset()
    hidden_state = np.zeros((shifted_agent.n_envs, shifted_agent.storage.hidden_state_size))
    # done = np.zeros(shifted_agent.n_envs)

    shifted_agent.policy.eval()
    unshifted_agent.policy.eval()
    n_pics = 0
    while n_pics < args.n_pics: # = 256
        act, log_prob_act, value, _, value_saliency_obs = shifted_agent.predict_w_value_saliency(obs, hidden_state, done)
        next_obs, rew, done, info = shifted_agent.env.step(act)

        if done:
            logit_saliency_obs = shifted_agent.predict_for_logit_saliency(obs, act)
            _,_,_, _, value_saliency_obs_unshifted = unshifted_agent.predict_w_value_saliency(obs, hidden_state, done)
            logit_saliency_obs_unshifted = unshifted_agent.predict_for_logit_saliency(obs, act)

            log_prob_act = log_prob_act.detach().cpu().numpy()
            value = value.detach().cpu().numpy()

            obs_copy, v_grad_vid = apply_saliency(obs, value_saliency_obs)
            _, l_grad_vid = apply_saliency(obs, logit_saliency_obs)

            _, unv_grad_vid = apply_saliency(obs, value_saliency_obs_unshifted)
            _, unl_grad_vid = apply_saliency(obs, logit_saliency_obs_unshifted)


            obs_copy = obs_copy.transpose((1,0,2))
            obs = next_obs
            
            
            top_image = np.concatenate((obs_copy, obs_copy), axis=1)
            mid_image = np.concatenate((v_grad_vid, unv_grad_vid), axis=1)
            bot_image = np.concatenate((l_grad_vid, unl_grad_vid), axis=1)
            
            full_image = np.concatenate((top_image, mid_image, bot_image), axis=0)

            plt.imshow(full_image)
            plt.savefig(logdir_saliency_value + f"/sal_img{n_pics}.png")
            n_pics += 1



        #     shifted_agent.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
        #     obs = next_obs

        # _, _, last_val, hidden_state = shifted_agent.predict(obs, hidden_state, done)
        # shifted_agent.storage.store_last(obs, hidden_state, last_val)

        # if args.save_value_individual:
        #     individual_value_idx = save_value_estimates_individual(shifted_agent.storage, epoch_idx, individual_value_idx)

        # if args.save_value:
        #     save_value_estimates(shifted_agent.storage, epoch_idx)
        #     epoch_idx += 1

        # shifted_agent.storage.compute_estimates(shifted_agent.gamma, shifted_agent.lmbda, shifted_agent.use_gae,
        #                                shifted_agent.normalize_adv)


def create_policy(action_space, architecture, device, hyperparameters, in_channels):
    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)
    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
    else:
        raise NotImplementedError
    policy.to(device)
    return model, policy


def apply_saliency(obs, reward_saliency_obs):
    reward_saliency_obs = reward_saliency_obs.swapaxes(1, 3)
    obs_copy = obs.swapaxes(1, 3)
    # plt.imshow(obs_copy.squeeze()*255)
    # plt.show()
    ims_grad = reward_saliency_obs.mean(axis=-1)

    percentile = np.percentile(np.abs(ims_grad), 99.9999999)
    ims_grad = ims_grad.clip(-percentile, percentile) / percentile
    ims_grad = torch.tensor(ims_grad)
    blurrer = tv.transforms.GaussianBlur(
                    kernel_size=5,
                    sigma=5.)  # (5, sigma=(5., 6.))
    ims_grad = blurrer(ims_grad).squeeze().unsqueeze(-1)

    pos_grads = ims_grad.where(ims_grad > 0.,
                                            torch.zeros_like(ims_grad))
    neg_grads = ims_grad.where(ims_grad < 0.,
                                            torch.zeros_like(ims_grad)).abs()


                # Make a couple of copies of the original im for later
    sample_ims_faint = torch.tensor(obs_copy.mean(-1)) * 0.2
    sample_ims_faint = torch.stack([sample_ims_faint] * 3, axis=-1)
    sample_ims_faint = sample_ims_faint * 255
    sample_ims_faint = sample_ims_faint.clone().detach().type(
                    torch.uint8).cpu().numpy()

    grad_scale = 9.
    grad_vid = np.zeros_like(sample_ims_faint)
    pos_grads = pos_grads * grad_scale * 255
    neg_grads = neg_grads * grad_scale * 255
    grad_vid[:, :, :, 2] = pos_grads.squeeze().clone().detach().type(
                    torch.uint8).cpu().numpy()
    grad_vid[:, :, :, 0] = neg_grads.squeeze().clone().detach().type(
                    torch.uint8).cpu().numpy()

    grad_vid = grad_vid + sample_ims_faint
    obs_copy = (obs_copy * 255).astype(np.uint8)

    return obs_copy.squeeze(), grad_vid.swapaxes(0,2).squeeze()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',         type=str, default = 'render', help='experiment name')
    parser.add_argument('--env_name',         type=str, default = 'coinrun', help='environment ID')
    parser.add_argument('--start_level',      type=int, default = int(0), help='start-level for environment')
    parser.add_argument('--num_levels',       type=int, default = int(0), help='number of training levels for environment')
    parser.add_argument('--distribution_mode',type=str, default = 'hard', help='distribution mode for environment')
    parser.add_argument('--param_name',       type=str, default = 'easy-200', help='hyper-parameter ID')
    parser.add_argument('--device',           type=str, default = 'cpu', required = False, help='whether to use gpu')
    parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
    parser.add_argument('--seed',             type=int, default = random.randint(0,9999), help='Random generator seed')
    parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints',  type=int, default = int(1), help='number of checkpoints to store')
    parser.add_argument('--logdir',           type=str, default = None)
    parser.add_argument('--used_learned_next_val',           type=bool, default = False)
    parser.add_argument('--next_val_net_file', type=str)

    #multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    #render parameters
    parser.add_argument('--tps', type=int, default=15, help="env fps")
    parser.add_argument('--vid_dir', type=str, default=None)
    parser.add_argument('--shifted_agent_file', type=str)
    parser.add_argument('--save_value', action='store_true')
    parser.add_argument('--save_value_individual', action='store_true')
    parser.add_argument('--value_saliency', action='store_true')



    parser.add_argument('--random_percent',   type=float, default=0., help='percent of environments in which coin is randomized (only for coinrun)')
    parser.add_argument('--corruption_type',  type=str, default = None)
    parser.add_argument('--corruption_severity',  type=str, default = 1)
    parser.add_argument('--agent_view', action="store_true", help="see what the agent sees")
    parser.add_argument('--continue_after_coin', action="store_true", help="level doesnt end when agent gets coin")
    parser.add_argument('--noview', action="store_true", help="just take vids")



    args = parser.parse_args()



    shifted_agent = "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033"
    # Check actually unshifted:
    unshifted_agent = "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033"
    # unshifted_agent = "logs/train/coinrun/coinrun/2024-10-08__12-16-22__seed_6033"
    args.shifted_agent_file = latest_model_path(shifted_agent)
    args.unshifted_agent_file = latest_model_path(unshifted_agent)
    
    args.exp_name = "coinrun_test" 
    args.env_name = "coinrun_aisc"
    args.distribution_mode = "hard"
    args.param_name = "hard-500"
    args.noview = True
    args.value_saliency = True
    args.level = "block3"
    args.n_pics = 30
    args.fig_only = True
    main(args)
    
