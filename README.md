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
2. **Training Time vs. Performance:** Training on Google Colab with a more complex model like InceptionV3 took significantly longer but showed a slightly lower accuracy improvement over a simpler architecture. This raises important considerations about resource efficiency and trade-offs in model selection.

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
   - Confusion Matrix

---

## Results
| Model                  | Accuracy (%) | F1 Score | Training Time |
|------------------------|--------------|----------|---------------|
| Baseline CNN           | 95.39        | 0.70     | ~10 min       |
| Improved InceptionV3   | 93.08        | 0.72     | ~2 hrs        |

---

## Discussion

The evaluation of model performance provides insights into the trade-offs between simplicity, complexity, and resource efficiency:

### Baseline CNN
The Baseline CNN achieved a higher accuracy of **95.39%** compared to the Improved InceptionV3. However, its F1 score of **0.70** suggests potential challenges in balancing precision and recall, especially when dealing with imbalanced data. This discrepancy highlights the model’s potential limitations in distinguishing between the two classes effectively.

Despite this, the Baseline CNN’s training time of **~10 minutes** makes it a practical choice for rapid development and iteration, especially in environments with limited computational resources.

### Improved InceptionV3
The Improved InceptionV3 model displayed a slightly lower accuracy of **93.08%**, potentially due to overfitting or challenges in generalization. However, its F1 score of **0.72** demonstrates improved balance between precision and recall, making it a better choice for applications where false positives and false negatives carry significant importance.

The model’s training time of **~2 hours** reflects the trade-off between complexity and computational cost. In scenarios with access to more powerful hardware, additional epochs or further fine-tuning could enhance both accuracy and F1 score, potentially surpassing the Baseline CNN.

### Key Takeaways:
- The Baseline CNN offers quick training and high accuracy, making it suitable for scenarios prioritizing efficiency.
- The Improved InceptionV3 model provides better-balanced predictions, but its longer training time makes it less practical without sufficient computational resources.

### Recommendations:
To maximize the potential of the Improved InceptionV3, future efforts could include:
- Utilizing more advanced hardware to reduce per-epoch training times.
- Increasing the number of epochs to explore its potential for achieving higher accuracy and F1 scores.

---

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

