import argparse
import numpy as np
import torch
import os
from pathlib import Path
import sys
import wandb
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

sys.path.append('/home/sunsh16e/flow-policy/')
from flow_policy.pusht.dataset import PushTStateDatasetWithNextObsAsAction
from flow_policy.pusht.dp_state_notebook.all import (
    ConditionalUnet1D, Rollout,
)
from flow_policy.pusht.sfpd import StreamingFlowPolicyDeterministic
from flow_policy.pusht.sfps import StreamingFlowPolicyStochastic

assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Testing script for flow-policy model.")
    parser.add_argument("--integration_steps_per_action", type=int, default=1, help="Integration time steps per action")
    parser.add_argument("--num_tests", type=int, default=100, help="Number of tests to run")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to EMA checkpoint file")
    parser.add_argument("--name", type=str, default="newbatch", help="Name for this experiment")
    parser.add_argument("--save_dir", type=str, default="/home/sunsh16e/flow-policy/push_T/scene_ood/test/",
                        help="Directory to save results")
    parser.add_argument("--k_p", type=float, default=5, help="Position gain")
    parser.add_argument("--k_v", type=float, default=1, help="Velocity gain")
    parser.add_argument("--seed", type=int, default=16, help="Random seed")
    parser.add_argument("--env_seed_base", type=int, default=500, help="Base seed for environment")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps in each test")
    parser.add_argument("--show_bar", action="store_true", help="Show progress bar")
    parser.add_argument("--generate_video", action="store_true", help="Generate video output")
    parser.add_argument("--model_type", type=str, default="sfpd", help="Model used for testing: sfpd or sfps")
    return parser.parse_args()


def main():
    args = parse_args()

    # Parameters
    integration_steps_per_action: int = args.integration_steps_per_action
    num_tests: int = args.num_tests
    ckpt_path: str = args.ckpt_path
    name: str = args.name
    save_dir: Path = Path(args.save_dir)
    k_p: float = args.k_p
    k_v: float = args.k_v
    seed: int = args.seed
    env_seed_base: int = args.env_seed_base
    max_steps: int = args.max_steps
    show_bar: bool = args.show_bar
    generate_video: bool = args.generate_video
    model_type: str = args.model_type

    # Derived parameters
    score_save_path = save_dir / f'{name}_score.pt'
    traj_save_path = save_dir / f'{name}_traj.pt'
    action_save_path = save_dir / f'{name}_action.pt'

    # Parameters
    pred_horizon = 16
    obs_dim = 5
    action_dim = 2
    obs_horizon = 2
    action_horizon = 8

    # Model type
    sfp = model_type == "sfp"
    new_sfp = model_type == "new_sfp"

    # if model_type == "sfp":
    #     print("Testing with SFP")
    # elif model_type == "dp":
    #     print("Testing with DP")
    # elif model_type == "new_sfp":
    #     print("Testing with new SFP")
    # else:
    #     raise ValueError(f"Invalid model type {model_type}, only accept sfp or dp")

    # Load policy weights
    if model_type == "sfpd":
        velocity_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon,
            fc_timesteps=1,
        )
        policy = StreamingFlowPolicyDeterministic(
            velocity_net=velocity_net,
            action_dim=action_dim,
            device='cuda',
        )
        state_dict = torch.load(ckpt_path, map_location='cuda')
        policy.load_state_dict(state_dict)
        policy.cuda()
    elif model_type == "sfps":
        velocity_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon,
            fc_timesteps=2,
        )
        policy = StreamingFlowPolicyStochastic(
            velocity_net=velocity_net,
            action_dim=action_dim,
            device='cuda',
        )
        state_dict = torch.load(ckpt_path, map_location='cuda')
        policy.load_state_dict(state_dict)
        policy.cuda()
    else:
        raise ValueError(f"Invalid model type {model_type}, only accept sfpd or sfps")
    print('Pretrained weights loaded.')

    # Create dataset
    dataset = PushTStateDatasetWithNextObsAsAction(
        pred_horizon=policy.pred_horizon.item(),
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )

    # Run tests
    score_list = []
    all_action_list = []
    all_state_list = []
    state = None

    # Initialize WandB
    wandb.init(
        project="pushT-test",
        config={
            "name": name,
            "unit_int_steps":integration_steps_per_action,
            "num_tests": num_tests,
            "ema_ckpt_path": ckpt_path,
            "save_dir": save_dir,
            "k_p_scale": k_p,
            "k_v_scale": k_v,
            "seed": seed,
            "env_seed_base": env_seed_base,
            "max_steps": max_steps,
        }
    )
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    for t_ix in range(num_tests):
        policy
        env_seed = env_seed_base + t_ix
        print(f"Environment seed: {env_seed}")
        Rollout(policy)
        if sfp or new_sfp:
            score, action_list, state_all = fp_test(
                noise_pred_net=noise_pred_net,
                k_p_scale=k_p,
                k_v_scale=k_v,
                t_span=t_span,
                select_action_indices=select_action_indices,
                device=device,
                env_seed=env_seed,
                obs_horizon=obs_horizon,
                stats=stats,
                show_bar=show_bar,
                state=state,
                generate_video=generate_video,
                max_steps=max_steps
            )
        else: #dp
            # num_diffusion_iters = 100
            # noise_scheduler = DDPMScheduler(
            #     num_train_timesteps=num_diffusion_iters,
            #     # the choise of beta schedule has big impact on performance
            #     # we found squared cosine works the best
            #     beta_schedule='squaredcos_cap_v2',
            #     # clip output to [-1,1] to improve stability
            #     clip_sample=True,
            #     # our network predicts noise (instead of denoised action)
            #     prediction_type='epsilon'
            # )
            # score, action_list, state_all = dp_test(
            #     ema_noise_pred_net=noise_pred_net,
            #     noise_scheduler = noise_scheduler,
            #     k_p_scale=k_p,
            #     k_v_scale=k_v,
            #     device=device,
            #     env_seed=env_seed,
            #     obs_horizon=obs_horizon,
            #     stats=stats,
            #     show_bar=show_bar,
            #     state=state,
            #     generate_video=generate_video,
            #     max_steps=max_steps
            # )


       
        wandb.log({"t_ix":t_ix, "score": score, 
                   "init_state_gx": state_all[0][0], "init_state_gy": state_all[0][1],
                   "init_state_bx": state_all[0][2], "init_state_by": state_all[0][3],
                   "init_state_theta": state_all[0][4],})
        # else:
        #     score, action_list = test(state, ema_noise_pred_net, stats, noise_scheduler, seed, env_seed = None, # seed if randomly generate env
        #  generate_video = False, show_bar = False, video_path = None,
        #  obs_horizon = 2, pred_horizon =16, action_dim = 2, action_horizon = 8, device = torch.device('cuda'),
        #  num_diffusion_iters = 100, return_action = False):


        print(f"Score: {score}, First action: {action_list[0]}")
        score_list.append(score)
        all_action_list.append(action_list)
        all_state_list.append(state_all)

    # Save results
    torch.save(score_list, score_save_path)
    print('Average score:', sum(score_list) / len(score_list))
    torch.save(all_action_list, action_save_path)
    torch.save(all_state_list, traj_save_path)
wandb.finish()

if __name__ == "__main__":
    main()