# Reinforcement Learning Playground
Basically, my own take on the OpenAI Gym.

The playround is composed of three core elements: an environment, a controller, and a learning task. The learning task is the glue between the controller and the environment, and moves actions and rewards back and forth between the other two components.

Currently, there are two environments: the discrete and the continuous version of the pole cart. The only controller so far is the Deep Q Network [1].


```
[1] Human-level control through deep reinforcement learning.
V. Mnih, K. Kavukcuoglu, D. Silver, A. Rusu, J. Veness, M. Bellemare, A. Graves,
M. Riedmiller, A. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik,
I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. 
Nature 518 (7540): 529-533 (2015)
```
