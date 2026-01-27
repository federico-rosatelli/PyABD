# Fast & Unsupervised Action Boundary Detection (ABD)

![University of Bonn](https://www.uni-bonn.de/de/universitaet/medien-universitaet/medien-presse-kommunikation/medien-corporate-design/uni_bonn_logo_standard_logo.jpg/images/image/large)

**University of Bonn** **Author:** Federico Rosatelli

---

## About the Project

This repository contains a PyTorch implementation of the CVPR 2022 paper **"Fast and Unsupervised Action Boundary Detection for Action Segmentation"**. 

The goal of this project is to solve the Temporal Action Segmentation task without relying on frame-wise ground truth during training. Instead of supervised learning, the method utilizes inherent feature similarities to detect **Action Boundaries** (Change Points) and clusters temporal segments to label actions. This approach is designed to be efficient, low-latency, and applicable to both offline analysis and online streaming scenarios.

## Key Features

* **Unsupervised Learning**: Does not require frame-by-frame annotations, leveraging the internal consistency of action features.
* **Boundary Detection**: Implements a signal processing approach to find local minima in cosine similarity (Change Point Detection).
* **Robust Refinement**: Uses a **Weighted Merge** strategy (Hierarchical Agglomerative Clustering) to fix over-segmentation while preserving temporal duration weights.
* **Dual Mode**:
    * **Offline**: Processes the entire video at once for maximum accuracy (MoF/F1 evaluation).
    * **Online (Simulated)**: A buffer-based streaming processor that respects causal latency for real-time applications.

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Getting Started
To run the application locally, ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
```