import torch
import time
import numpy as np

from drl.data import Batch, Storage, CacheBuffer
from drl.env import BaseMultiEnv

class Collector:

    def __init__(self, agent, env, replay=None,
                    act_space=None, store_keywords=None, safety_env=False):
        """Collector

        args:
            agent: agent of reinforcement learning
            env: environment to collect
            replay: experience pool
            store_keywords: variable to be stored after each step of agent
            safety_env: collect cost in each step if True
        
        return:
            None
        """
        self.agent = agent
        self.process_fn = agent.process_fn
        self.env = env
        self._multi_env = False
        self.env_num = 1
        if isinstance(env, BaseMultiEnv):
            self._multi_env = True
            self.env_num = len(env)
        self.replay = replay if replay is not None else Storage(1000)
        self.act_space = act_space
        self.store_keywords = store_keywords
        self.store_cost = safety_env

        self.states = None
        self.has_reset_fn = hasattr(self.agent, 'reset')

        if self._multi_env:
            self._cached_replays = [CacheBuffer() for _ in range(self.env_num)]

        self.collect_step = 0
        self.collect_episode = 0
        self.collect_time = 0.0

        self.reset_env()
        self.reset_replay()

        self.XVec = []
        self.YVec = []
        self.deltaVec = []
        self.xdotVec = []
        self.ydotVec = []
        self.psiVec = []
        self.psidotVec = []
        self.FVec = []
        self.minDist = []
        self.trajectory = env.trajectory
        self.timestep = env.timestep
    

    def reset_env(self):
        self._obs = self.agent.state_normalizer(self.env.reset())
        self._act = self._rew = self._done = self._info = None

        # rew is one-step reward, reward is sum of rew
        if self._multi_env:
            self.cum_reward = np.zeros(self.env_num)
            self.cum_cost = np.zeros(self.env_num)
            self.length = np.zeros(self.env_num)
        else:
            self.cum_reward = self.cum_cost = self.length = 0

    def reset_replay(self):
        if self._multi_env:
            for r in self._cached_replays:
                r.reset()
        self.replay.reset()


    def _reset_states(self, index=None):
        if hasattr(self.agent, 'reset_states'):
            self.agent.reset_states(self.states, index)


    def seed(self, seed=None):
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)

    def render(self, **kwargs):
        if hasattr(self.env, 'render'):
            return self.env.render(**kwargs)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


    def collect(self, n_step=0, n_episode=0, render=0, random=False):

        assert sum([n_step!=0, n_episode!=0]) == 1, \
            'only 1 of n_step or n_episode should > 0'

        start_time = time.time()
        current_step = 0
        current_episode = 0 # np.zeros(self.env_num, dtype=int) if self._multi_env else 0
        reward_list, cost_list, length_list = [], [], []

        while True:
            if self._multi_env:
                batch_data = Batch(obs=self._obs)
            else:
                batch_data = Batch(obs=np.array([self._obs]))

            if random:
                self._act = np.stack([self.act_space.sample() for _ in range(self.env_num)], axis=0)
            else:
                with torch.no_grad():
                    result, self.states = self.agent(batch_data, states=self.states)
            
                if isinstance(result.act, torch.Tensor):
                    result.act = result.act.detach().cpu().numpy()
                self._act = result.act

                if self.store_keywords:
                    for k in self.store_keywords:
                        setattr(self, '_'+k, getattr(result, k).detach().cpu().numpy())
            
            if not self._multi_env:
                self._act = self._act[0]
            obs_next, self._rew, self._done, self._info, info_lst = self.env.step(self._act)
            obs_next = self.agent.state_normalizer(obs_next)

            [delT, X, Y, xdot, ydot, psi, psidot, F, delta, disError] = info_lst

            self.XVec.append(X)
            self.YVec.append(Y)
            self.deltaVec.append(delta)
            self.xdotVec.append(xdot)
            self.ydotVec.append(ydot)
            self.psiVec.append(psi)
            self.psidotVec.append(psidot)
            self.FVec.append(F)
            self.minDist.append(disError)

            if render > 0:
                self.render()
                time.sleep(render)

            self.length += 1
            self.cum_reward += self._rew

            if self.store_cost:
                if self._multi_env:
                    self.cum_cost += np.stack([self._info[_]['cost'] for _ in range(self.env_num)])
                else:
                    self.cum_cost += self._info['cost']

            store_dict = {}
            if self._multi_env:
                for i in range(self.env_num):
                    
                    if (not random) and self.store_keywords:
                        store_dict = {k:getattr(self, '_'+k)[i] for k in self.store_keywords}
                    if self.store_cost:
                        store_dict['cost'] = self._info[i]['cost']

                    self._cached_replays[i].add({
                        'obs': self._obs[i],
                        'act': self._act[i],
                        'rew': self._rew[i],
                        'done': self._done[i],
                        'obs_next': obs_next[i],
                        **store_dict
                    })
                    current_step += 1
                    if self._done[i]:
                        current_episode += 1
                        reward_list.append(self.cum_reward[i])
                        cost_list.append(self.cum_cost[i])
                        length_list.append(self.length[i])
                        self.replay.update(self._cached_replays[i])
                        
                        self.cum_reward[i], self.cum_cost[i], self.length[i] = 0, 0, 0
                        self._reset_states(i)
                        
                        self._cached_replays[i].reset()
                
                if sum(self._done) > 0:
                    obs_next = self.env.reset(np.where(self._done)[0])
                    obs_next[self._done] = self.agent.state_normalizer(obs_next[self._done])
                if n_episode != 0 and current_episode >= n_episode:
                    break        

            else:
                if self.store_keywords:
                    store_dict = {k:getattr(self, '_'+k) for k in self.store_keywords}
                if self.store_cost:
                    store_dict['cost'] = self._info['cost']

                self.replay.add({
                    'obs': self._obs,
                    'act': self._act,
                    'rew': self._rew,
                    'done': self._done,
                    'obs_next': obs_next,
                    **store_dict
                })

                current_step += 1
                if self._done:
                    current_episode += 1
                    reward_list.append(self.cum_reward)
                    cost_list.append(self.cum_cost)
                    length_list.append(self.length)
                    self.cum_reward = self.cum_cost = self.length = 0
                    
                    obs_next = self.agent.state_normalizer(self.env.reset())
                    self._reset_states()

                    if n_episode != 0 and current_episode >= n_episode:
                        break
            
            if n_step != 0 and current_step >= n_step:
                break

            self._obs = obs_next
        
        self._obs = obs_next

        duration = time.time() - start_time

        self.collect_step += current_step
        self.collect_episode += current_episode
        self.collect_time += duration
        
        n_episode = max(1, current_episode)
        
        return_dict = {
            'n_epis': current_episode,
            'n_step': current_step,
            'reward': sum(reward_list) / n_episode,
            'length': sum(length_list),
            'reward_list': reward_list,
            'length_list': length_list
        }
        if self.store_cost:
            return_dict['cost'] = sum(cost_list) / n_episode
            return_dict['cost_list'] = cost_list

        return return_dict

    def sample(self, batch_size):
        batch_data, _ = self.replay.sample(batch_size)
        batch_data = self.process_fn(batch_data, self.replay)
        return batch_data