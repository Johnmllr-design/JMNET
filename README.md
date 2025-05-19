# 🧠 JMNet – A Neural Network From Scratch

This repo contains a self-made neural network I built in Python from scratch — no TensorFlow, no PyTorch, just the handmade application of the chain rule, the derivative, the dot product operations which I learned in class (calc 1) to a real world problem that I found interesting (sleep habitts).  I built this mostly for fun and to get a deeper understanding of how neural nets actually work under the hood. It supports forward and backward propagation, stochastic gradient descent, and training with real data.

You can try it out with the sample dataset in `main.py`, or plug in your own dataset and see how it performs.

---

## 🔧 Features

- ✅ **Custom neural net architecture** – flexible input/output layers, supports the sigmoid activation function for versatility and simplicity in regression problems
- ✅ **Sigmoid activation** – with full forward/backward support
- ✅ **Stochastic Gradient Descent (SGD)** – trains using backpropagation and the chain rule to figure the rate of change as we adjust neuron weight and biases through each epoch and input of training.
- ✅ **No external ML libraries** – everything built from scratch using NumPy
- ✅ **3D Visualization tools** – compare model predictions and actual labels in 3D, and plot error magnitudes in color to see where the model's doing well (or not)

---

## 📊 What I Have Added Recently (as of May 18, 2025)

- Added time normalization that makes midnight land at `0.5` instead of `0.0`, which helps the model learn more intuitively from sleep data (bedtime/wakeup)
- Built out 3D plots to visualize:
  - Predictions
  - Ground truth labels
  - Error magnitude (absolute difference between prediction and label)

---

## 🚀 want to try it out?

1. Clone the repo
2. Run `main.py` to train and visualize the sample dataset within the repository
3. Want to try your own data? Replace the input arrays and rerun — the rest takes care of itself

---

## 📌 My Next-up plans

- Add support for more activation functions (ReLU, tanh)
- Add support for Adam optimizer
- Clean up training interface and maybe make it a tiny library

---

This has been a fun, knowledge enriching build so far. If you're into building ML stuff from scratch too or have suggestions, feel free to open an issue or fork it.

— John M.
