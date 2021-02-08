# DICE: The DIstribution Correction Estimation Library

This library unifies the distribution correction estimation algorithms for off-policy evaluation, including:
* [DualDICE: Behavior-Agnostic Estimation of Discounted Stationary Distribution Corrections](https://arxiv.org/abs/1906.04733)
* [GenDICE: Generalized Offline Estimation of Stationary Values](https://arxiv.org/abs/2002.09072)
* [Reinforcement Learning via Fenchel-Rockafellar Duality](https://arxiv.org/abs/2001.01866)
Please cite these work accordingly upon using this library.

## Summary
Existing DICE algorithms are the results of particular regularization choices in the Lagrangian of the Q-LP and d-LP policy values.
![Regularized Lagrangian](figures/reg_lang.png)*Choices of regularization (colored) in the Lagrangian.*

These choices navigate the trade-offs between optimization stability and estimation bias.
![Estimation bias](figures/est_bias.png)*Estimation bias given the choices of regularization.*

## Install

Navigate to the root of project, and perform:

    pip3 install -e .

To run taxi, download the pretrained policies and place them under policies/taxi:

    git clone https://github.com/zt95/infinite-horizon-off-policy-estimation.git
    cp -r infinite-horizon-off-policy-estimation/taxi/taxi-policy policies/taxi

## Run DICE Algorithms
First, create datasets using the policy trained above:

    for alpha in {0.0,1.0}; do python3 scripts/create_dataset.py --save_dir=./tests/testdata --load_dir=./tests/testdata/CartPole-v0 --env_name=cartpole --num_trajectory=400 --max_trajectory_length=250 --alpha=$alpha --tabular_obs=0; done

Run DICE estimator:

    python3 scripts/run_neural_dice.py --save_dir=./tests/testdata --load_dir=./tests/testdata --env_name=cartpole --num_trajectory=400 --max_trajectory_length=250 --alpha=0.0 --tabular_obs=0

To recover DualDICE, append the following to the above python command:

    --primal_regularizer=0. --dual_regularizer=1. --zero_reward=1 --norm_regularizer=0. --zeta_pos=0

To recover GenDICE, append the following to the above python command:

    --primal_regularizer=1. --dual_regularizer=0. --zero_reward=1 --norm_regularizer=1. --zeta_pos=1

The configuration below generally works the best:

    --primal_regularizer=0. --dual_regularizer=1. --zero_reward=0 --norm_regularizer=1. --zeta_pos=1
