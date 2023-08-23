# Music Genre Classification

**Language:** Python

**Framework:** PyTorch

**Platform:** Jupyter Notebook, PyCharm

**Cloud Service:** Google Colab

## Overview

In the context of music genre classification, we address three key challenges.

- Firstly, misclassification prevails due to the intricate nature of this task, making conventional classification methods based on artists or publishers inadequate for accurate labeling.
- Secondly, data processing can hamper performance by disregarding essential features, while excessive features result in computational overhead. This highlights the need for efficient feature extraction.
- Lastly, the demand for memory-efficient models arises, as the deployment on low-power hardware like mobile devices requires careful consideration.

To overcome these challenges, our proposal encompasses a multi-faceted approach. We employ a **multi-modal neural network** amalgamating a ResNet18 convolutional neural network and LSTM recurrent neural network. This model uses both image data, such as Mel Spectrograms, and time series data like MFCCs, leading to a significant 10% accuracy enhancement.

Our experiments involved PyTorch on Google Colab for acceleration. Baseline models were trained on singular data types to predict music genres. Notably, ResNet18 models trained on wave plots or Zero Crossing Rate plots showed limited accuracy, around 50%. Conversely, the model trained on Mel Spectrograms achieved over 80% accuracy due to richer data representation. Further investigations are ongoing to determine whether our proposed multi-modal network outperforms these baselines.

To **optimize audio signal processing**, we divided audio signals into smaller segments, extracting MFCCs for feature representation. By employing a 9-layer CNN model trained with a batch size of 16 and 40 epochs, we achieved around 95% accuracy. Efficient model development was pursued through advanced architectures, utilizing ResNet18 for image-based representation and an MLP for numerical features, incorporating dropout layers for regularization. Model compression techniques, including pruning, quantization, and knowledge distillation, were applied to ensure deployment on memory-constrained devices.

The quantized model exhibited comparable accuracy (0.7273) to the original (0.7374), while enhancing inference speed (0.14791101 seconds) and achieving a 3.93 compression ratio. For music feature models, both original and pruned achieved 0.9274 accuracy with a 5.49 compression ratio. **Pruning** effectively eliminated non-critical weights without significant performance degradation. Through these comprehensive efforts, we effectively addressed the challenges in music genre classification, enhancing accuracy and efficiency for practical deployment.
