# training-internals

Learning how neural network training actually works by building it from scratch in C++. Starting with basic linear regression and gradually adding layers, nonlinearity, and visualizations to understand why training breaks.

## What I've built so far

- Linear regression with manual backprop (no libraries, just math)
- Tested learning rate sensitivity - watched it diverge with LR too high
- Trained on multiple data points, saw the model find best-fit compromise

- Refactored into DenseLayer struct to separate responsibilities
- Implemented caching (store input during forward for use in backward)
- Split gradient computation: main calculates dL/dŷ, layer calculates dL/dw using chain rule

## Running it
```bash
cd cpp_neural_engine
g++ main.cpp -o train && ./train
```