# mlp-cnn-fashion-mnist
McGill COMP 551 - Mini-Project 3: Implementing Multi-Layer Perceptrons (MLP) from scratch with backpropagation, and comparing performance against CNN and ResNet architectures using PyTorch on the Fashion-MNIST dataset.

- Gabriel Caballero (261108565)
- Adam Dufour (261193949)

## Prerequisites
* Python 3.8+
* Packages in requirements.txt
```bash
pip install -r requirements.txt
```

![](figures/comparison-mlp-cnn-resnet.png)

## Project
The notebook is structured in 4 main tasks:
* Task 1: Obtaining and cleaning data
![training-samples](figures/raw-training-samples.png)
  * Whitening
    * ![pca-reconstructions](figures/pca-reconstructions.png)
  * PCA
    * ![preprocessed-example](figures/preprocessed-example.png)
  * Data augmentation
    * ![augmented-samples](figures/augmented-samples.png)
* Task 2: MLP implementation from scratch
  * ![models-performance-comparison](figures/models-performance-comparison.png)
* Task 3: Running the experiments
  * Regularization
  * CNN comparison
    * ![cnn-comparison](figures/cnn-accuracy-train-time.png)
  * ResNet18 comparison (transfer learning)

## Running the Code
Launch the Jupyter Notebook / JupyterLab from your terminal (or use PyCharm/VSCode) and click run, all the tests will start running automatically
