# Chinese Poem-to-Painting Generation via Cycle-consistent Adversarial Networks

This project implements a semi-supervised approach for bidirectional translation between classical Chinese poems and paintings using cycle-consistent adversarial networks. The model can generate paintings from poems and vice versa, capturing the symbolic essence of artistic expression in both modalities.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Classical Chinese poetry and painting represent an important part of the world's cultural heritage. This project aims to computationally capture the intricate relationship between these two art forms by learning bidirectional mappings that enforce semantic alignment between the visual and textual modalities.

The key features of this approach include:

- Semi-supervised learning using both paired and unpaired data
- Cycle-consistent adversarial networks for robust cross-modal translation
- Novel evaluation metrics to assess quality, diversity, and consistency

## Features

- Poem-to-Painting generation
- Painting-to-Poem generation
- Semi-supervised training using cycle-consistency
- Customizable model architecture
- Comprehensive evaluation metrics

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

## Dataset

We use the Chinese Painting Description Dataset (CPDD), which includes 3,217 Chinese poems and corresponding paintings. The dataset comprises various painting categories across different historical periods.

To use your own dataset, modify the `data/dataset.py` file to load and preprocess your data accordingly.

## Model Architecture

The model consists of several key components:

- Painting and Poem Encoders
- Painting and Poem Generators
- Painting and Poem Discriminators

These components work together in a cycle-consistent adversarial framework to learn bidirectional mappings between the painting and poem domains.

## Training

The training process involves alternating between supervised training on paired data and unsupervised training on unpaired data. The model is trained to minimize cycle-consistency losses, adversarial losses, and supervised reconstruction losses.

Refer to `train.py` for the detailed training procedure and hyperparameters.

## Evaluation

We employ several metrics to evaluate the quality, diversity, and semantic consistency of the generated poems and paintings:

- Fréchet Inception Distance (FID) for paintings
- BLEU and METEOR scores for poems
- Perplexity for linguistic quality
- Distribution Consistency Error (DCE) for cross-modal alignment

Refer to `evaluate.py` for the implementation of these metrics.

## Results

(Include some sample results, visualizations, or performance metrics here)

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For any questions or issues, please open an issue on GitHub or contact the project maintainers.
