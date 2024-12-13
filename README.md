# Machine_learning_and_Neural_Network

Below is a **README.md** file for the GitHub repository related to your face mask detection project:

```markdown
# Face Mask Detection using Convolutional Neural Networks (CNN)

This project demonstrates the creation of a Convolutional Neural Network (CNN) to detect whether a person is wearing a face mask or not. The implementation uses TensorFlow/Keras for building and training the model, and the dataset is sourced from Kaggle.



## Overview
The goal of this project is to:
1. Build a CNN model to classify images into two categories: **with mask** and **without mask**.
2. Use the model for real-time or image-based predictions to enhance public safety and compliance monitoring.

---

## Dataset
The dataset is sourced from Kaggle: [Face Mask Dataset by Omkar Gurav](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset). It contains:
- **3,725 images** of people wearing masks.
- **3,828 images** of people not wearing masks.

---

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
   cd face-mask-detection
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your Kaggle API key file (`kaggle.json`) in the project directory.

4. Download and extract the dataset:
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   kaggle datasets download -d omkargurav/face-mask-dataset
   unzip face-mask-dataset.zip -d data/
   ```

---

## Implementation
### Preprocessing
- Images are resized to **128x128** pixels.
- Labels are assigned:
  - `1` for **with_mask** images.
  - `0` for **without_mask** images.
- Data is normalized by dividing pixel values by 255.
- The dataset is split into **80% training** and **20% testing**.

### Model Architecture
The CNN architecture consists of:
1. **Conv2D** layers with ReLU activation for feature extraction.
2. **MaxPooling2D** for dimensionality reduction.
3. Fully connected **Dense** layers for classification.
4. **Dropout** layers to prevent overfitting.
5. Final output layer with a **sigmoid** activation for binary classification.

---

## Evaluation
The model is evaluated using:
- **Training and validation accuracy/loss graphs**.
- **Test accuracy** on unseen data.

---

## Usage
### Train the Model
Run the provided script to train the model:
```bash
python train_model.py
```

### Predict Using the Model
To predict an image:
1. Provide the path of the input image.
2. The model will display the image and print whether the person is wearing a mask.

Example:
```bash
Path of the image to be predicted: /content/data/with_mask/with_mask_1545.jpg
```

Output:
```
The person in the image is wearing a mask.
```

---

## Results
- **Training Accuracy**: Achieved during the training phase.
- **Validation Accuracy**: Monitored during the training phase.
- **Test Accuracy**: Evaluated on the testing set.

Graphs:
1. **Training vs Validation Loss**
2. **Training vs Validation Accuracy**

---

## Future Scope
1. **Real-Time Detection**: Integrate with CCTV systems for live monitoring.
2. **Multi-Class Classification**: Add categories for incorrect mask usage.
3. **Transfer Learning**: Use pre-trained models like MobileNetV2 for enhanced performance.

---

## Acknowledgments
- Dataset: [Kaggle Face Mask Dataset by Omkar Gurav](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- Frameworks and Libraries: TensorFlow, Keras, OpenCV, Matplotlib

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
```

### Notes:
1. Replace `"Deepakvishwakarma1"` in the GitHub URL with your actual username.
