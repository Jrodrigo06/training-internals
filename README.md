# training-internals

Learning how neural network training actually works by building it from scratch in C++. Starting with basic linear regression and gradually adding layers, nonlinearity, and visualizations to understand why training breaks.

## What I've built so far

**Basic training mechanics**
- Implemented linear regression with manual gradient derivation (MSE loss, chain rule)
- Trained on multiple data points - model finds best-fit line through compromise

**Added structure**
- Refactored into DenseLayer struct separating forward/backward responsibilities
- Implemented activation caching (store input during forward for backward pass)
- Split gradient computation: main calculates dL/dŷ, layer uses chain rule for dL/dw

**Added ReLU nonlinearity**
- Implemented ReLU activation with conditional gradient flow
- Discovered dying ReLU problem through experiments

## Experiments/Tests

**Learning rate sensitivity**
- LR = 0.01: smooth convergence to loss ≈ 0
- LR = 0.1: faster convergence (1 epoch vs 200)
- LR = 1.0: divergence - loss explodes to infinity, then NaN

**Dying ReLU conditions**
- Init `w=1.0, b=-5.0` + ReLU: neuron dies (all z < 0, gradients = 0, params frozen forever)
- Init `w=1.0, b=-5.0` without ReLU: learns normally (linear always allows gradient flow)
- Init `w=1.0, b=-1.0` + ReLU: neuron survives (3/4 points have z > 0, providing gradient)
- Insight: neuron only dies if z ≤ 0 for ALL training data

## Running it
```bash
cd cpp_neural_engine
g++ main.cpp -o train && ./train
```