# Pig Tac Toe RL Agent

A reinforcement learning agent that plays Pig Tac Toe — a twist on tic-tac-toe played on a 4x4 grid where players can gamble ("roll again") to place multiple pieces per turn, but risk losing them all. The player with the most 3-in-a-lines at the end wins.

This project was built as a final project for a reinforcement learning course. The goal: train an AI from scratch using RL fundamentals and compete on a class leaderboard.

## How It Works

The agent uses **Monte Carlo learning** with two neural networks:

- **Value network** — evaluates board states to decide *where* to place a piece
- **Q-function network** — evaluates the *stop vs. roll again* decision, with separate outputs for each action trained using one-hot masking

Both networks are 3-layer feedforward nets (input → 64 hidden → 64 hidden → output) built in JAX, trained with the Adam optimizer.

### Training Tricks

- **Self-play** — the agent trains against a snapshot of itself, updating the opponent once the average loss drops below 0.1 over the last 50 games
- **Board symmetry augmentation** — every board state generates 8 training samples via rotations and reflections, since Pig Tac Toe is fully symmetric
- **Hand-crafted features** — potential 2-in-a-rows and 3-in-a-rows for both players are fed as extra inputs alongside the one-hot board encoding, significantly accelerating learning

## Results

Final agent achieved **38% win rate vs. the course bot** and **75% vs. the random agent**, placing in the top 3 on the class leaderboard.

## Files

| File | Description |
|------|-------------|
| `training.py` | Full training loop — game simulation, neural network definitions, self-play logic |
| `calculator.py` | Serialized trained model weights |
| `humanReport.pdf` | Writeup documenting the iterative development process |

## Stack

Python, JAX, Optax, einops
