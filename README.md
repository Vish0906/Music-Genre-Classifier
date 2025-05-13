# Music-Genre-Classifier
Music Genre classification project using TensorFlow/Keras with MFCC features, implementing dense and CNN models, and evaluating performance using scikit-learn and Matplotlib.

# Speech Classification using MFCC Features and Deep Learning

This project performs **speech command classification** using audio data by extracting **MFCC (Mel-Frequency Cepstral Coefficients)** features and training deep learning models using **TensorFlow/Keras**. Two types of neural networks are explored: a **Dense Neural Network (DNN)** and a **Convolutional Neural Network (CNN)**.

---

## Libraries Used

- `librosa` – For audio loading and MFCC feature extraction
- `numpy`, `os`, `random` – For numerical and file handling operations
- `scikit-learn` – For data preprocessing and evaluation metrics
- `matplotlib`, `seaborn` – For plotting accuracy, confusion matrix, etc.
- `tensorflow` (Keras) – For building and training DNN and CNN models

---

## Dataset

The dataset consists of `.wav` audio files organized in folders where each folder name represents a **label** (e.g., "yes", "no", "up", "down", etc.). The dataset is preprocessed by:

1. Loading audio files
2. Converting them into MFCCs
3. Padding/truncating them to a consistent shape
4. One-hot encoding the labels

---

## Models

### 1. Dense Neural Network (DNN)
- Input: Flattened MFCCs
- Architecture: Fully connected layers with dropout
- Output: Softmax over classes

### 2. Convolutional Neural Network (CNN)
- Input: 2D MFCC array
- Architecture: Multiple Conv2D + MaxPooling2D layers followed by dense layers
- Output: Softmax over classes

---

## Evaluation

- **Accuracy** and **Loss** are plotted for both training and validation.
- **Confusion Matrix** is visualized using seaborn to identify model strengths and weaknesses across different labels.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/speech-classification-mfcc.git
   cd speech-classification-mfcc
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Organize `.wav` files into labeled folders inside a directory (e.g., `dataset/yes`, `dataset/no`, etc.)

4. Run the script:
   ```bash
   python main.py
   ```

---

## Notes

- You can toggle between CNN and Dense model in the script.
- MFCC parameters (e.g., number of coefficients, frame length) can be tuned for better performance.
- Confusion matrix and accuracy/loss plots will be saved after training.

---

## Sample Output

- Confusion Matrix  
- Training vs Validation Accuracy/Loss plots

---

## Future Improvements

- Add more robust data augmentation (e.g., noise injection, pitch shifting)
- Support for real-time prediction
- Integration with a UI or voice assistant

---

## License

This project is open-source and available under the MIT License.
