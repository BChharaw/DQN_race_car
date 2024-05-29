# Implementation of a DQN in Pytorch using gymnasium
Solving the car racing problem in OpenAI Gym using Proximal Policy Optimization (PPO). This problem has a real physical engine in the back end. You can achieve real racing actions in the environment, like drifting. 

## Requirement
To run the code, you need
- [pytorch 0.4](https://pytorch.org/)
- [gym 0.10](https://github.com/openai/gym)
- [visdom 0.1.8](https://github.com/facebookresearch/visdom)

## Method
Every action will be repeated for 8 frames. To get velocity information, state is defined as adjacent 4 frames in shape (4, 96, 96). Use a two heads FCN to represent the actor and critic respectively. The actor outputs α, β for each actin as the parameters of Beta distribution. 
<div align=center><img src="img/network.png" width="30%" /></div>

## Training
To train the agent, run```python train.py```
To test, run ```python run.py```.
Ensure parameters are set as desired in ```parameters.py```

### Gamma (γ)

**Description:** The discount factor for future rewards.\
**Impact:** Determines the importance of future rewards. A value close to 1 considers future rewards heavily, while a value close to 0 focuses more on immediate rewards.\
**Tuning:**

-   Start with a value around 0.99.
-   If the agent is too short-sighted, increase γ (e.g., to 0.995).
-   If the agent is too long-sighted and performs poorly in the short term, decrease γ (e.g., to 0.9).

### Max Gradient Norm (max_grad_norm)

**Description:** The maximum norm of gradients for gradient clipping.\
**Impact:** Prevents the gradient from becoming too large, which can destabilize learning.\
**Tuning:**

-   Start with a value around 0.5.
-   If training is unstable, try lowering it (e.g., to 0.1).
-   If gradients are too small, increase it cautiously (e.g., to 1.0).

### Clip Parameter (clip_param)

**Description:** The clipping parameter for the PPO objective function.\
**Impact:** Controls the range of policy updates to ensure stable training.\
**Tuning:**

-   Start with a value around 0.2.
-   If the policy updates are too aggressive, decrease the clip parameter (e.g., to 0.1).
-   If the policy updates are too conservative, increase it slightly (e.g., to 0.3).

### PPO Epoch (ppo_epoch)

**Description:** The number of epochs to update the policy using the collected data.\
**Impact:** More epochs can lead to better learning but can also increase the risk of overfitting.\
**Tuning:**

-   Start with a value around 10.
-   If the agent overfits to the training data, reduce the number of epochs (e.g., to 5).
-   If the agent underfits and learns slowly, increase the number of epochs (e.g., to 15).

### Buffer Capacity

**Description:** The size of the buffer for storing experience tuples.\
**Impact:** A larger buffer can store more experiences, which can improve the quality of training but requires more memory.\
**Tuning:**

-   Start with a value around 5000.
-   If training is slow and the agent doesn't have enough diverse experiences, increase the buffer size (e.g., to 10000).
-   If memory usage is too high, decrease the buffer size (e.g., to 2000).

### Batch Size

**Description:** The number of samples to draw from the buffer for each update.\
**Impact:** Larger batch sizes provide more stable updates but require more computation per update.\
**Tuning:**

-   Start with a value around 64.
-   If updates are too noisy, increase the batch size (e.g., to 128).
-   If updates are too slow and require too much computation, decrease the batch size (e.g., to 32).

### Run Episodes (run_episodes)

**Description:** The number of episodes to run during training.\
**Impact:** Determines how long the agent is trained.\
**Tuning:**

-   Start with a value around 100.
-   If the agent doesn't seem to learn enough, increase the number of episodes (e.g., to 200).
-   If the agent converges too early or if training takes too long, decrease the number of episodes (e.g., to 50).

### T Timesteps (t_timesteps)

**Description:** The number of timesteps per episode.\
**Impact:** Controls the duration of each episode.\
**Tuning:**

-   Start with a value around 1000.
-   If episodes end too quickly for meaningful learning, increase the number of timesteps (e.g., to 2000).
-   If episodes are too long and computationally expensive, decrease the number of timesteps (e.g., to 500).

### Convolutional Layers

#### Number of Filters:

**Impact:** More filters can capture more complex features, but also increase computational cost and risk of overfitting.\
**Tuning:**

-   Increase the number of filters in each layer (e.g., from 8 to 16, 16 to 32, etc.).
-   Experiment with increasing filters more gradually (e.g., 8 → 16 → 32 → 64 → 128 → 256).

#### Kernel Size:

**Impact:** Larger kernels can capture more context but may lose finer details.\
**Tuning:**

-   Try larger kernels for earlier layers (e.g., kernel_size=5 instead of 3).
-   Use smaller kernels in deeper layers to focus on fine details.

#### Stride:

**Impact:** Larger strides reduce the spatial dimensions faster, which can lead to loss of information.\
**Tuning:**

-   Experiment with smaller strides (e.g., stride=1 instead of 2) in earlier layers to retain more spatial information.

#### Activation Functions:

**Impact:** Different activations can affect the learning dynamics.\
**Tuning:**

-   Experiment with other activation functions like LeakyReLU, ELU, or SELU.

### Fully Connected Layers

#### Number of Neurons:

**Impact:** More neurons can capture more complex patterns but increase the risk of overfitting.\
**Tuning:**

-   Increase the number of neurons in fully connected layers (e.g., from 100 to 200).

#### Number of Layers:

**Impact:** More layers can increase the model's capacity but also make it harder to train.\
**Tuning:**

-   Add additional fully connected layers (e.g., an additional layer with nn.Linear(100, 100) and ReLU).

### Regularization

#### Dropout:

**Impact:** Dropout helps prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.\
**Tuning:**

-   Add dropout layers after convolutional or fully connected layers (e.g., nn.Dropout(p=0.5)).

#### Batch Normalization:

**Impact:** Batch normalization can stabilize and accelerate training.\
**Tuning:**

-   Add batch normalization layers after convolutions (e.g., nn.BatchNorm2d(8)).


## Credit
This codebase is a continuation and maintenance of the work by https://github.com/xtma/pytorch_car_caring/tree/master

