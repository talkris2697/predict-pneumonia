# predict-pneumonia

# Pneumonia Detection Using CNN

This project focuses on improving a Convolutional Neural Network (CNN) for pneumonia detection from chest X-ray images. The work began with a base implementation from a Kaggle notebook ([Intro to CNN using Keras to Predict Pneumonia](https://www.kaggle.com/code/sanwal092/intro-to-cnn-using-keras-to-predict-pneumonia)) and progressed through a series of optimizations to enhance its performance. This project was developed as part of a Neural Networks course.

---

## Overview

### Objective:
Improve the baseline pneumonia classification results by applying simple yet effective modifications, such as:
- Adding additional layers to the architecture.
- Experimenting with different pre-trained models like InceptionV3.
- Conducting research to identify effective methods for transfer learning and fine-tuning.

### Key Findings:
1. **Pre-trained InceptionV3 Model:** After fine-tuning the InceptionV3 model, the accuracy and F1 score improved significantly compared to the baseline model.
2. **Training Time vs. Performance:** Training on Google Colab with a more complex model like InceptionV3 took significantly longer but yielded only a slight improvement over a simpler architecture. This raises important considerations about resource efficiency and trade-offs in model selection.

---

## Dataset
The dataset used for this project is sourced from the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). It contains labeled images under two categories:
- **NORMAL**
- **PNEUMONIA**

The dataset is divided into training, validation, and test sets.

---

## Methodology

1. **Baseline Implementation:**
   - Replicated the architecture and workflow from the Kaggle notebook.
   - Observed issues with overfitting, a need for better data splitting between training, validation, and test sets, and suboptimal performance metrics.

2. **Improvements Made:**
   - **Additional Layers:** Added fully connected and dropout layers to mitigate overfitting and enhance feature extraction.
   - **Pre-trained Models:** Incorporated InceptionV3 for transfer learning. Fine-tuned the model’s top layers after freezing the base layers, followed by unfreezing selective layers for further improvement.
   - **Loss Function:** Binary crossentropy was used to address the binary classification problem.
   - **Optimizer:** Adam optimizer was selected with an adaptive learning rate.

3. **Evaluation Metrics:**
   - Accuracy
   - F1 Score
   - Confusiion Matrix

---

## Results
| Model                  | Accuracy (%) | F1 Score | Training Time |
|------------------------|--------------|----------|---------------|
| Baseline CNN           | 75.86        | 0.72     | ~10 min       |
| Improved InceptionV3   | 88.89        | 0.83     | ~2 hrs      |




---

## Discussion

The addition of layers and the use of a pre-trained InceptionV3 model led to better performance. However, the improvement in accuracy and F1 score came at the cost of significantly longer training times.

### Key Takeaways:
- **Complex Models vs. Simple Architectures:** While complex pre-trained models like InceptionV3 yield slightly better results, their training time might not justify the minor improvements in some cases. This highlights the importance of considering the context and available computational resources.
- **Simple Improvements Can Go a Long Way:** Adding layers and incorporating regularization techniques like dropout proved effective in improving the baseline model.


## Future Work
- Explore additional pre-trained models such as ResNet50 or EfficientNet.
- Implement techniques like data augmentation and learning rate schedulers to further improve model generalization.
- Investigate the impact of using a larger or more diverse dataset.

---

## Acknowledgments
Special thanks to the authors of the original Kaggle notebook for providing a starting point and to the creators of the Chest X-Ray dataset.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to raise issues or contribute improvements via pull requests!

