# 🧠 Neural Network for MNIST Classification

![Python](https://img.shields.io/badge/python-3.x-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/pytorch-1.x-orange?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

This repository contains a **PyTorch** implementation of a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.  
It supports both **training** and **testing** modes with fully configurable hyperparameters.

---

## 🚀 Features

✅ Train a CNN on MNIST with configurable hyperparameters  
✅ Use data augmentation (rotation, horizontal flip)  
✅ Save and load model checkpoints   
✅ Monitor loss using TensorBoard   
✅ Flexible architecture: adjustable hidden layers, filters, dropout, and activation   

---

## ⚙️ Requirements

- Python 3.x  
- PyTorch  
- torchvision  
- numpy  
- h5py  
- tensorboard

Install all dependencies:

```bash
pip install torch torchvision numpy h5py tensorboard

**Train**
python main.py -mode train -num_epoches 25 -batch_size 100

**Test**
python main.py -mode test -batch_size 100
