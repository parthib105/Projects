# 🧠 Neural Network from Scratch (on MNIST)

This project implements a **simple neural network from scratch** using NumPy, trained on the MNIST dataset (handwritten digits).

## 📦 Project Structure

```text
├── data/ # MNIST data stored here
├── main.py # Entry-point for training
├── neural_network/ # Core neural net components
│ ├── activation_functions.py
│ ├── loss_functions.py
│ ├── model.py
│ └── init.py
├── requirements.txt # Dependencies
└── README.md # You're reading it!
```


## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py

🧠 Features
- Manual implementation of:
    - ReLU and Softmax
    - Cross-Entropy Loss
    - Backpropagation and Gradient Descent
- MNIST dataset loading using torchvision
- Training loop from scratch


✅ TODO
- Add accuracy metric
- Add support for other datasets
- Add visualization of weights/loss