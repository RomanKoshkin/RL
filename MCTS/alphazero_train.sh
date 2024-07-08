#!/bin/bash

# This script is used to train the AlphaZero model using MCTS.
for runid in {1..10}; do
    python3 alphazero.py \
        --runid $runid &
done