# RL A3C Pytorch

![A3C LSTM playing Breakout-v0](https://raw.githubusercontent.com/dgriff777/rl_a3c_pytorch/master/Breakout.gif)
![A3C LSTM playing MsPacman-v0](https://raw.githubusercontent.com/dgriff777/rl_a3c_pytorch/master/MsPacman.gif)
![A3C LSTM playing Pong-v0](https://raw.githubusercontent.com/dgriff777/rl_a3c_pytorch/master/Pong.gif)

This is a repository includes my implementation with reinforcement learning using Asynchronous Advantage Actor-Critic (A3C) 

### A3C LSTM

I implemented an A3C LSTM model and trained it in the atari 2600 environments provide in the Openai Gym. I have only trained model on environemnts Pong-v0, MsPacman-v0, and Breakout-v0 but had very good performance and currently hold top scores on openai gym leaderboard for those games.

link to the my Gym environment evaluations below

- ![A3C LSTM playing Breakout-v0](https://gym.openai.com/evaluations/eval_CyVPHgs0S22DiZsWXoPFw)
- https://gym.openai.com/envs/Pong-v0
- https://gym.openai.com/envs/MsPacman-v0

## Requirements

- Python 2.7+
- Openai Gym and Universe
- Pytorch

## Training
To train agent on Pong-v0 environment with 32 different worker threads:

```
OMP_NUM_THREADS=1 python main.py --env-name Pong-v0" --num-processes 32
```

## Evaluation
To run a 100 episode gym evaluation with trained model
```
python gym_eval.py --env-name Pong-v0 --num-episodes 100
```

## Project Reference

- https://github.com/ikostrikov/pytorch-a3c

## README still under construction