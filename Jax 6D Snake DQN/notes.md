For RL: Always add a renderer! You need to see what the model is doing, and play as the model given what the model sees.

Always show eval during training

Jit the entire training function

Make batched types (GameStateBatch) and operate with tree_util where necessary

When starting a project, minimize time spent messing with hyperparameters. Put 98% of your time/effort into making the training code faster and eliminating bugs.