# Reinforcement Learning Method for Webots Vehicle
## Webots-vehicle env
Following instructions below to configure webots-vehicle env
1. Download Webots and its python lib.
2. Download this project code.
3. Modify `vehicle_proj_dir` in [sim_env.py](sim_env.py).
4. Test your webots-vehicle env.
    1. start webots project 
        ```
        webots vehicle-Trajectory-Tracking/DRL/worlds/automotive_new.wbt
        ```
    2. make sure to use external control mode. If not, select `DEF TESLA ROBOT > controller` and change it to `<extern>`.
    3. run 
        ```
        python sim_env.py
        ```
        to test env.

## Train a RL algorithm
- run `python run_ppo.py` to train policy.
- run `python run_ppo.py --test` to test policy.
You can also use GPU to accelerate train and infer by adding `--cudaid 0`. We only apply 3-layer MLP as our model for actor/critic so it only requires very low CUDA memory.
