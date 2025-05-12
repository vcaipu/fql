# Flow Q-Learning (FQL)

This repository contains two subdirectories:

**JAX**: Our experiments changing the distillation loss metric, built on top of the original implementation, found [here](https://github.com/seohongpark/fql)

**PyTorch**: The latest iteration we tried to implement the FQL algorithm in PyTorch. This implementation very closely follows the original JAX code structure, again found [here](https://github.com/seohongpark/fql), but uses PyTorch as the backend.

## Overview

Flow Q-Learning (FQL) is a simple and performant offline reinforcement learning method that leverages an expressive flow-matching policy to model arbitrarily complex action distributions in data. The key insight is to train a separate one-step policy that maximizes values while distilling from a flow model trained with behavioral cloning.

