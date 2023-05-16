---
firstpage:
lastpage:
---

# Pen

These datasets were generated with the [`FrankaKitchen-v1`](https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/) environment, originally hosted in the [`D4RL`](https://github.com/aravindr93/hand_dapg)[1] and [`relay-policy-learning](https://github.com/google-research/relay-policy-learning)[2] repository. The goal of the `FrankaKitchen` environment is to interact with the various objects in order to reach a desired state configuration. The objects you can interact with include the position of the kettle, flipping the light switch, opening and closing the microwave and cabinet doors, or sliding the other cabinet door. The desired goal configuration for all datasets is to complete 4 subtasks: open the microwave, move the kettle, flip the light switch, and slide open the cabinet door.

There are three types of datasets:

```{toctree}
:hidden:
kitchen/partial.md
kitchen/combined.md
kitchen/mixed.md
```

## References

[1] Fu, Justin, et al. ‘D4RL: Datasets for Deep Data-Driven Reinforcement Learning’. CoRR, vol. abs/2004.07219, 2020, https://arxiv.org/abs/2004.07219.

[2] Gupta, A., Kumar, V., Lynch, C., Levine, S., & Hausman, K. (2019). Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning. arXiv preprint arXiv:1910.11956.
