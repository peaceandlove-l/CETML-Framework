# CETML-Framework
# Align and Purify: Cross-Domain Edge-Detail-Aware Transfer Mutual Learning Framework

This repository provides the code and results for "Align and Purify: Cross-Domain Edge-Detail-Aware Transfer Mutual Learning Framework for RGB-D Indoor Semantic Segmentation"

---

## Requirements

- Python 3.7+
- PyTorch 1.5+
- CUDA 10.2+
- TensorboardX 2.1+
- opencv-python

If anything goes wrong with the environment, please check `requirements.txt` for details.

---

## Architecture and Details

<p align="center">
  <img src="assets/mutual learning.png" width="800">
</p>
<p align="center">
  <img src="assets/SGCNet.png" width="800">
</p>
<p align="center">
  <img src="assets/FATNet.png" width="800">
</p>
<p align="center">
  <img src="assets/RFF.png" width="800">
</p>
### Overview

The proposed **CETML** adopts a Cross-Domain Edge-Detail-Aware Transfer Mutual Learning Framework.  
It includes Four main components:

1. **Geometric Transfer Learning (GTL)** – Enhances feature alignment between RGB and thermal branches.  
2. **Boundary Boosting Algorithm (BBA)** – Strengthens edge localization by introducing spatial boundary cues.  
3. **Semantic Transfer Learning (STL)** – Refines cross-modal semantic understanding through joint optimization.

---


