# DynaGradTree: Online Decision Trees with Dynamic Feature Spaces

This repository contains the implementation code for online learning with Variable Feature Space (VFS) scenarios.

## Overview

DynaGradTree is a gradient-based decision tree designed to handle dynamic feature spaces in online learning environments. Unlike traditional decision trees that require a fixed feature set, DynaGradTree can seamlessly accommodate new features as they arrive in the data stream without requiring model reconstruction.

The key innovation is a two-layer masking architecture that combines reserved space allocation with dynamic expansion capabilities, allowing the model to handle both pre-allocated feature growth and unexpected new features beyond initial capacity.

## Repository Structure

The implementation consists of three main Python files:

- **DynaGradTree.py**: Core model implementation containing the complete architecture including the two-layer masking system, learned surrogate splits mechanism, and dynamic expansion capabilities.

- **dynagradtree_learner.py**: Online learning adapter that wraps DynaGradTree into a standard incremental learning interface with predict-then-update protocol.

- **run_covertype_experiment.py**: Example experimental code demonstrating usage of DynaGradTree in practice.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Here is a basic example of using DynaGradTree:

```python
from dynagradtree_learner import create_gradtree_medium

# Create a DynaGradTree instance
model = create_gradtree_medium(
    max_features_reserve_ratio=1.5,  # Reserve 1.5x initial features
    n_classes=2,                      # Binary classification
    batch_size=100,                   # Batch size for training
    depth=6,                          # Tree depth
    epochs=3                          # Training epochs per batch
)

# Online learning loop
for X_batch, y_batch in data_stream:
    # Predict on current batch
    predictions = model.predict(X_batch)
    
    # Update model with current batch
    model.update(X_batch, y_batch)
```

## Key Features

- **Dynamic Feature Space Handling**: Seamlessly accommodates new features without model reconstruction through a two-layer masking architecture.

- **Learned Surrogate Splits**: Intelligently handles missing features through learned substitution strategies rather than simple imputation.

- **Online Learning Compatible**: Implements standard predict-then-update protocol for integration with online learning frameworks.

- **Interpretable**: Maintains tree-based interpretability while supporting dynamic feature spaces.

## Requirements

This implementation requires Python 3.8 or higher along with TensorFlow 2.x and standard scientific Python packages. See requirements.txt for complete dependency list.

## License

This code is provided for research purposes in conjunction with our IJCAI 2026 submission.