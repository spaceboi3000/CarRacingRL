# Dreamer V1 - CarRacing

**Semester Project for Neurofuzzy Control at ECE NTUA**

**Author:** Nikolaos Moraitis

---

A PyTorch implementation of the [Dreamer](https://arxiv.org/abs/1912.01603) algorithm (Hafner et al., 2020) applied to the CarRacing-v3 environment from Gymnasium.

Dreamer learns a world model of the environment and uses it to "dream" (imagine) trajectories in latent space, training an actor-critic policy entirely within these imagined rollouts.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Architecture Overview](#architecture-overview)
- [Known Issues: Slow Car Behavior](#known-issues-slow-car-behavior)
- [Hyperparameters](#hyperparameters)
- [References](#references)

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU training works)

### Step 1: Create a Virtual Environment (Recommended)

```bash
python -m venv dreamer-env
source dreamer-env/bin/activate  # Linux/macOS
# or
dreamer-env\Scripts\activate     # Windows
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision
pip install gymnasium[box2d]
pip install opencv-python
pip install numpy
pip install pandas
pip install matplotlib
```

Or install all at once:

```bash
pip install torch torchvision gymnasium[box2d] opencv-python numpy pandas matplotlib
```

### Step 3: Verify Installation

```bash
python -c "import gymnasium as gym; env = gym.make('CarRacing-v3'); print('CarRacing environment loaded successfully!')"
```

---

## Project Structure

```
.
├── train.py       # Main training script
├── test.py        # Evaluation script with dream visualization
├── dreamer.py     # Dreamer agent (encoder, decoder, actor, critic, reward model)
├── RSSM.py        # Recurrent State-Space Model (latent dynamics)
├── convVAE.py     # Convolutional encoder and decoder networks
├── replay.py      # Experience replay buffer
└── README.md      # This file
```

---

## Usage

### Training

To train the Dreamer agent from scratch:

```bash
python train.py
```

Training will:
1. Seed the replay buffer with 5 random episodes
2. Alternate between collecting environment experience and training
3. Save the best model to `dreamer_best.pth`
4. Save periodic checkpoints as `dreamer_step{N}.pth`
5. Log training metrics to `training_log.csv`

**Training time:** Expect 5-10+ hours on a GPU for decent performance (paper uses 5M steps for control suite tasks).

### Evaluation

#### Standard Evaluation (with rendering)

```bash
python test.py --model dreamer_best.pth --episodes 5
```

#### Evaluation with Dream Visualization

See what the agent "imagines" in real-time:

```bash
python test.py --model dreamer_best.pth --episodes 3 --show-dreams
```

This displays:
- Real RGB observation from the environment
- Preprocessed grayscale input (64×64)
- Agent's reconstruction (what it "sees")
- Imagined future frames (the agent's "dreams")

#### Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Path to saved model weights | `dreamer_best1.5.pth` |
| `--episodes` | Number of evaluation episodes | `5` |
| `--no-render` | Disable environment rendering | `False` |
| `--stochastic` | Use stochastic policy (sample vs. mean) | `False` |
| `--show-dreams` | Enable real-time dream visualization | `False` |
| `--imagination-horizon` | Future steps to imagine (dream mode) | `5` |

---

## Architecture Overview

### Components

1. **Encoder**: CNN that compresses 64×64 grayscale images → 1024-dim embedding
2. **RSSM**: Recurrent State-Space Model maintaining:
   - Deterministic state `h` (GRU hidden, 200-dim)
   - Stochastic state `z` (sampled latent, 30-dim)
3. **Decoder**: Transposed CNN reconstructing images from state `[h, z]`
4. **Reward Model**: MLP predicting rewards from state
5. **Actor**: Policy network outputting tanh-squashed Gaussian actions
6. **Critic**: Value function estimating expected returns

### Training Loop

```
1. Collect experience in environment → Store in replay buffer
2. Sample sequences from buffer
3. Train World Model: Reconstruction + Reward prediction + KL divergence
4. Train Actor-Critic: Imagine trajectories using world model, optimize via λ-returns
5. Repeat
```

---

## Known Issues: 
### Slow Car Behavior

If your trained agent drives very slowly or barely accelerates, this is a common issue with this implementation. Here are the primary causes and solutions:

### Slow training
Due to lack of GPU we cant iterate multiple times and improve  the algorithm

## Hyperparameters

Key hyperparameters (from `train.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `STOCH_DIM` | 30 | Stochastic latent dimension (z) |
| `DET_DIM` | 200 | Deterministic RNN hidden dimension (h) |
| `HIDDEN_DIM` | 300 | MLP hidden layer size |
| `EMBED_DIM` | 1024 | Image embedding dimension |
| `HORIZON` | 15 | Imagination rollout length |
| `GAMMA` | 0.99 | Discount factor |
| `LAMBDA` | 0.95 | GAE λ for value targets |
| `FREE_NATS` | 3.0 | KL divergence threshold |
| `BATCH_SIZE` | 50 | Training batch size |
| `SEQ_LEN` | 50 | Sequence length for training |
| `ACTION_REPEAT` | 2 | Frames per action |

---

## References

1. Hafner, D., et al. (2020). **Dream to Control: Learning Behaviors by Latent Imagination**. ICLR 2020. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)

2. Hafner, D., et al. (2019). **Learning Latent Dynamics for Planning from Pixels**. ICML 2019. [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)

3. Ha, D., & Schmidhuber, J. (2018). **World Models**. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)

---

## License

This project is for educational purposes as part of the Neurofuzzy Control course at ECE NTUA.