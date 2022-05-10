import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from copy import deepcopy

from drl.agent import BaseAgent
from drl.data import Batch
from drl.utils import tensor, MeanStdNormalizer

class PPOAgent(BaseAgent):

    def __init__(self, 
                 actor, critic, 
                 actor_optim, critic_optim, 
                 dist_fn,
                 discount_factor=0.99,
                 gae_factor=None,
                 max_grad_norm=.5,
                 eps_clip_ratio=.2,
                 vf_coef=.5,
                 ent_coef=.0,
                 action_range=[-1., 1.],
                 state_normalizer=MeanStdNormalizer(),
                 ignore_done=True,
                ):
        
        super().__init__()
        self._eps = np.finfo(np.float32).eps.item()

        self.actor, self.actor_old = actor, deepcopy(actor)
        self.actor_old.eval()

        self.critic = critic
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim

        self.dist_fn = dist_fn
        self._gamma = discount_factor
        self._lambda = gae_factor
        self._max_grad_norm = max_grad_norm
        self._eps_clip_ratio = eps_clip_ratio
        self._w_vf = vf_coef
        self._w_ent = ent_coef
        self._range = action_range
        self.state_normalizer = state_normalizer

        self.ignore_done = ignore_done
        

    def train(self):
        self.training = True
        self.actor.train()
        self.critic.train()
        self.state_normalizer.unset_read_only()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()
        self.state_normalizer.set_read_only()
    
    def save_model(self, model_path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
        }, f'{model_path}.model')

        with open(f'{model_path}.stats', 'wb') as f:
            pickle.dump(self.state_normalizer.state_dict(), f)
        

    def load_model(self, model_path):
        models = torch.load(f'{model_path}.model')
        self.actor.load_state_dict(models['actor'])
        self.critic.load_state_dict(models['critic'])
        self.actor_optim.load_state_dict(models['actor_optim'])
        self.critic_optim.load_state_dict(models['critic_optim'])

        self.actor_old = deepcopy(self.actor)
        self.actor_old.eval()

        with open(f'{model_path}.stats', 'rb') as f:
            self.state_normalizer.load_state_dict(pickle.load(f))
        

    def process_fn(self, batch, replay):
        returns = self._get_returns(batch)
        batch.update(returns=returns)
        if self.ignore_done:
            batch.done = batch.done * 0.
        return batch

    def _get_returns(self, batch):
        returns = batch.rew.copy()
        # last = 0
        for i in reversed(range(len(returns))):
            if batch.done[i] or i == len(returns) - 1:
                if self.ignore_done:
                    returns[i] = self.critic(batch.obs[i:i+1]).squeeze().detach().cpu().numpy()
                else:
                    returns[i] = batch.rew[i]
            else:
                returns[i] += self._gamma * last
            last = returns[i]
        return returns
        

    def __call__(self, batch, states=None, actor_name='actor'):
        if actor_name == 'actor':
            logits = self.actor(batch.obs)
        elif actor_name == 'actor_old':
            logits = self.actor_old(batch.obs)
        
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
            
        else:
            dist = self.dist_fn(logits)
        
        act = dist.sample()
        if self._range:
            act = act.clamp(self._range[0], self._range[1])
        
        return Batch(logits=logits, act=act, dist=dist, log_p_act=dist.log_prob(act)), None

    def sync_weights(self):
        for o, n in zip(self.actor_old.parameters(),
                        self.actor.parameters()):
            o.data.copy_(n.data)

    def learn(self, batch, batch_size=None, repeat=1):
        
        actor_losses, vf_losses, ent_losses = [], [], []
        
        batch.act = tensor(batch.act)
        batch.returns = tensor(batch.returns).unsqueeze(-1)
        
        # compute reward Advantage
        vs_old = self.critic(batch.obs).detach()
        vs_next_old = self.critic(batch.obs_next).detach()
        if self._lambda is None: 
            adv = batch.returns - vs_old
        else: # use GAE
            # get TD error first
            td_error = tensor(batch.rew).unsqueeze(-1) + \
                self._gamma * tensor(1. - batch.done).unsqueeze(-1) * vs_next_old - vs_old
            
            adv = td_error
            last = 0
            for i in reversed(range(adv.shape[0])):
                if not batch.done[i]:
                    adv[i] += self._gamma * self._lambda * last
                last = adv[i]
        
        adv = (adv - adv.mean()) / (adv.std() + 1e-12)
        adv = adv.detach()
        batch.update(adv=adv)

        for _ in range(repeat):
            for b in batch.sampler(batch_size):
                dist = self(b)[0].dist
                dist_old = self(b, actor_name='actor_old')[0].dist
                ratio = torch.exp((dist.log_prob(b.act) - dist_old.log_prob(b.act)).sum(-1, keepdim=True))
                
                # update critic
                vf_loss = self._w_vf * F.mse_loss(self.critic(b.obs), b.returns)
                vf_losses.append(vf_loss.item())

                self.critic_optim.zero_grad()
                vf_loss.backward()
                nn.utils.clip_grad_norm_(list(self.critic.parameters()),
                                         self._max_grad_norm)
                self.critic_optim.step()

                # update actor
                # actor loss
                surr1 = ratio * b.adv
                surr2 = ratio.clamp(1. - self._eps_clip_ratio, \
                    1. + self._eps_clip_ratio) * b.adv
                clip_loss = torch.min(surr1, surr2).mean()

                e_loss = dist.entropy().mean()
                ent_losses.append(e_loss.item())
                
                actor_loss = - (clip_loss + self._w_ent * e_loss)
                if actor_loss.item() < -1000:
                    print('b.adv', b.adv.mean())
                    print(b.adv)
                    print('ratio', ratio.shape)
                    print(ratio)
                    raise RuntimeError('-------------')
                actor_losses.append(actor_loss.item())
                
                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()),
                                         self._max_grad_norm)
                self.actor_optim.step()

        # self.sync_weights()

        return {
            'actor_loss': actor_losses,
            'actor_loss/ent': ent_losses,
            'critic_loss/r_critic': vf_losses,
            # 'mse/r_mse': np.mean(vf_losses),
        }
