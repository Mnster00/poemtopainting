# Chinese Poem-to-Painting Generation via Cycle-consistent Adversarial Networks

This project implements a semi-supervised approach for bidirectional translation between classical Chinese poems and paintings using cycle-consistent adversarial networks. The model can generate paintings from poems and vice versa, capturing the symbolic essence of artistic expression in both modalities.

## Table of Contents

- [Introduction](#introduction)
- [Structure](#Structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)

## Introduction

Classical Chinese poetry and painting represent an important part of the world's cultural heritage. This project aims to computationally capture the intricate relationship between these two art forms by learning bidirectional mappings that enforce semantic alignment between the visual and textual modalities.


## Structure
   ```
   project/
   ├── config.py
   ├── data/
   │   └── __init__.py
   │   └── dataset.py
   │   └── dataloader.py
   ├── models/
   │   └── __init__.py
   │   └── encoder.py
   │   └── generator.py
   │   └── discriminator.py
   ├── utils/
   │   └── __init__.py
   │   └── losses.py
   │   └── metrics.py
   ├── train.py
   ├── evaluate.py
   ├── inference.py
   └── requirements.txt
   ```


## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/chinese-poem-painting-generation.git
   cd chinese-poem-painting-generation
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. To train the model:
   ```
   python train.py
   ```

2. To evaluate the model:
   ```
   python evaluate.py
   ```

3. To generate a poem from a painting:
   ```
   python inference.py --mode poem --input path/to/painting.jpg
   ```

4. To generate a painting from a poem:
   ```
   python inference.py --mode painting --input "你的诗句在这里"
   ```


## Training

The training process involves alternating between supervised training on paired data and unsupervised training on unpaired data. The model is trained to minimize cycle-consistency losses, adversarial losses, and supervised reconstruction losses.

Refer to `train.py` for the detailed training procedure and hyperparameters.


## Results

