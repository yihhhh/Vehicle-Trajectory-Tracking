import time
import numpy as np
from tqdm import tqdm

from drl.trainer.utils import test_episode, gather_info

def onpolicy_trainer(agent, train_collector, test_collector, max_epoch,
                     step_per_epoch, collect_per_step, repeat_per_collect,
                     episode_per_test, batch_size, save_path,
                     stop_fn=None, writer=None, verbose=True
                     # train_fn=None, test_fn=None
                    ):
    global_step = 0 # the times of actor/critic optimization (for each batch)
    global_length = 0 # the times of env taking action to step
    best_epoch, best_reward = -1, -1

    # print('------ Start warm-up. ------')
    # train_collector.collect(n_episode=8, random=True)
    # agent.learn(train_collector.sample(0), batch_size, repeat_per_collect)
    # print('------ End warm-up. ------')

    start_time = time.time()

    for epoch in range(1, max_epoch + 1):
        
        postfix_data = {}
        # train
        agent.train()

        with tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}') as t:
            while t.n < t.total:
                # 1. collect data from environment
                result = train_collector.collect(n_episode=collect_per_step)
                
                # 2. train policy
                losses = agent.learn(train_collector.sample(0), batch_size, repeat_per_collect)
                agent.sync_weights()
                train_collector.reset_replay()
                
                
                # 3. save training logs
                if writer:
                    # 3.1 logs of actor/critic 
                    step = 1
                    for k in losses.keys():
                        if isinstance(losses[k], list):
                            step = max(step, len(losses[k]))
                    global_step += step

                    for k in losses.keys():
                        loss_value = losses[k][-1] if isinstance(losses[k], list) else losses[k]
                        writer.add_scalar(k, loss_value, global_step=global_step)

                    # 3.2 logs of interaction with env
                    if 'reward_list' in result:
                        for r, l in zip(result['reward_list'], result['length_list']):
                            global_length += l
                            writer.add_scalar('reward', r, global_step=global_length)
                    else:
                        global_length += result['length']
                        writer.add_scalar('reward', result['reward'], global_step=global_length)
                
                postfix_data['avg_reward'] = f'{result["reward"]:.2f}'
                
                t.update(step)
                t.set_postfix(**postfix_data)

            if t.n <= t.total:
                t.update()
        
        # save model
        agent.save_model(save_path)

        # test
        result = test_episode(agent, test_collector, episode_per_test)
        if best_epoch == -1 or best_reward < result['reward']:
            best_reward = result['reward']
            best_epoch = epoch
        if verbose:
            print(f'Epoch #{epoch}: test_reward: {result["reward"]:.3f}, '
                  f'best_reward: {best_reward:.3f} in #{best_epoch}')
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(start_time, train_collector, test_collector, best_reward)