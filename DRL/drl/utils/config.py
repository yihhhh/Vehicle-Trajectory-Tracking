import torch
import argparse

class Config:
    DEVICE = torch.device('cpu')

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_argument = self.parser.add_argument

        self.task_name = 'untitled'
        self.seed = None
        self.replay_size = 20000
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.gamma = 0.99 # discount
        self.tau = 0.005 # soft parameter synchronize
        self.epoch = 100
        self.step_per_epoch=2400
        self.collect_per_step=4
        self.batch_size=128
        self.layer_num=1
        self.env_num_train=8
        self.env_num_test=10
        self.logdir='../exp/log'

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])

    def select_device(self):
        if not hasattr(self, 'cudaid'):
            return
        if self.cudaid >= 0:
            Config.DEVICE = torch.device('cuda:%d' % (self.cudaid))
        else:
            Config.DEVICE = torch.device('cpu')