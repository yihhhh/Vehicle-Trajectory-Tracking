import os, sys
import numpy as np
from copy import deepcopy
import gym
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from drl.agent import PPOAgent
from drl.trainer import onpolicy_trainer
from drl.data import Collector, Storage
from drl.env import VectorEnv, SubprocVectorEnv
from drl.network import ActorProb2, Critic
from drl.utils import Config, set_seed, BaseNormalizer

from sim_env import TracingEnv

def get_args(parser):
    parser.add_argument('--task-name', type=str, default='webot_vehicle')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--replay-size', type=int, default=100000)
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=500)
    parser.add_argument('--collect-per-step', type=int, default=8)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--env-num-train', type=int, default=8)
    parser.add_argument('--env-num-test', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='exp/log')
    parser.add_argument('--modeldir', type=str, default='exp/model')
    parser.add_argument('--cudaid', type=int, default=-1)
    parser.add_argument('--gae-lambda', type=float, default=0.97)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)

    parser.add_argument('--test', action='store_true')


def showResult(traj, timestep, X, Y, delta, xdot, ydot, F, psi, psidot, minDist):
    totalTime = np.linspace(0, len(X) * timestep * 0.001, len(X))
    print('total steps: ', timestep * len(X))

    fig, _ = plt.subplots(nrows=4, ncols=2, figsize=(15, 10))

    plt.subplot(421)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.plot(traj[:, 0], traj[:, 1], 'gray', linewidth=6.0)
    plt.plot(X, Y, 'r')

    plt.subplot(422)
    plt.xlabel('Time (s)')
    plt.ylabel('delta (rad)')
    plt.plot(totalTime[2:], delta[2:], 'r')

    plt.subplot(423)
    plt.xlabel('Time (s)')
    plt.ylabel('xdot (m/s)')
    plt.plot(totalTime[2:], xdot[2:], 'r')

    plt.subplot(424)
    plt.xlabel('Time (s)')
    plt.ylabel('ydot (m/s)')
    plt.plot(totalTime[2:], ydot[2:], 'r')

    plt.subplot(425)
    plt.xlabel('Time (s)')
    plt.ylabel('psi (rad)')
    plt.plot(totalTime[2:], psi[2:], 'r')

    plt.subplot(426)
    plt.xlabel('Time (s)')
    plt.ylabel('psidot (rad/s)')
    plt.plot(totalTime[2:], psidot[2:], 'r')

    plt.subplot(427)
    plt.xlabel('Time (s)')
    plt.ylabel('minDist (m)')
    plt.plot(totalTime[2:], minDist[2:], 'r')

    plt.subplot(428)
    plt.xlabel('Time (s)')
    plt.ylabel('F (N)')
    plt.plot(totalTime[2:], F[2:], 'r')

    fig.tight_layout()

    avgDist = sum(minDist) / len(minDist)
    print('Maximum distance off reference trajectory: ', max(minDist))
    print('Average distance off reference trajectory: ', avgDist)
    plt.show()

def main():
    cfg = Config()
    get_args(cfg.parser)
    cfg.merge()
    cfg.select_device()
    
    # initialize train/test env
    train_env = TracingEnv()

    cfg.s_dim = np.prod(train_env.observation_space.shape)
    cfg.a_dim = np.prod(train_env.action_space.shape)
    cfg.action_range = [train_env.action_space.low[0], train_env.action_space.high[0]]
    
    seed = 7651 # np.random.randint(10000)
    print('seed=',seed)
    set_seed(seed)

    # model
    actor = ActorProb2(cfg.layer_num, cfg.s_dim, cfg.a_dim, hidden_dim=32).to(Config.DEVICE)
    critic = Critic(cfg.layer_num, cfg.s_dim, hidden_dim=32).to(Config.DEVICE)
    actor_optim = torch.optim.Adam(list(actor.parameters()), lr=cfg.actor_lr)
    critic_optim = torch.optim.Adam(list(critic.parameters()), lr=cfg.critic_lr)
    
    agent = PPOAgent(actor, critic,
        actor_optim, critic_optim,
        torch.distributions.Normal, cfg.gamma, \
        gae_factor=cfg.gae_lambda, 
        max_grad_norm=cfg.max_grad_norm,
        eps_clip_ratio=cfg.eps_clip,
        vf_coef=cfg.vf_coef, 
        ent_coef=cfg.ent_coef,
        ignore_done=False,
    )
    
    # save model
    if not os.path.exists(f'{cfg.modeldir}/{cfg.task_name}'):
        os.makedirs(f'{cfg.modeldir}/{cfg.task_name}', 0o777)
    
    save_path = f'{cfg.modeldir}/{cfg.task_name}/ppo'

    # stop trainin when avg reward > threshold
    def stop_fn(r):
        return r > 910

    if not cfg.test:
        # collector
        collector = Collector(agent, train_env, Storage(cfg.replay_size), \
            act_space=train_env.action_space, safety_env=False)
        
        # for logging
        if not os.path.exists(f'{cfg.logdir}/{cfg.task_name}'):
            os.makedirs(f'{cfg.logdir}/{cfg.task_name}', 0o777)
        writer = SummaryWriter(f'{cfg.logdir}/{cfg.task_name}/ppo')

        # train algorithm by trainer
        onpolicy_trainer(agent, collector, collector, cfg.epoch,\
            cfg.step_per_epoch, cfg.collect_per_step, cfg.repeat_per_collect,\
            cfg.env_num_test, cfg.batch_size, save_path, stop_fn, writer, verbose=True)

        print('==========================')
        print('Training completed.')
        print('==========================')

    else:
        agent.load_model(save_path)
        agent.eval()
        collector = Collector(agent, train_env)
        result = collector.collect(n_episode=1)
        showResult(collector.trajectory, collector.timestep*5, collector.XVec, collector.YVec,
                   collector.deltaVec, collector.xdotVec, collector.ydotVec,
                   collector.FVec, collector.psiVec, collector.psidotVec, collector.minDist)

        info = np.array([collector.XVec, collector.YVec,
                   collector.deltaVec, collector.xdotVec, collector.ydotVec,
                   collector.FVec, collector.psiVec, collector.psidotVec, collector.minDist])
        np.savetxt("C:/Users/yihan/myWorkspace/16745/16745-Project/ppo_result.csv", info, delimiter=",", fmt="%1.4f")

        print(result)
        # print(f'Final reward: {result["reward"]:1d}±{np.std(result["reward_list"]):1d}, length: {result["length"]}')
    

if __name__ == "__main__":
    main()
