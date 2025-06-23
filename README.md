# Biomedical-Informatics-using-GANs
# 🧠 Skin Cancer Classification using CNN (HAM10000 Dataset)

This project implements a deep learning-based approach for classifying types of skin lesions using the **HAM10000** dataset. The model leverages **Convolutional Neural Networks (CNNs)** built with **TensorFlow** and **Keras**, aiming to assist in early detection of skin cancer.

---

## 📁 Project Structure

- **Setup & Environment**: Version checks for Python, TensorFlow, and Scikit-learn.
- **Kaggle Dataset Integration**: Automated download and extraction of the HAM10000 dataset using the Kaggle API.
- **Data Preparation**: Image preprocessing and label encoding.
- **Model Architecture**: CNN model designed and trained for skin lesion classification.
- **Evaluation**: Includes accuracy metrics, confusion matrix, and visualizations.

---

## 🔍 Dataset

- **Dataset**: [HAM10000 - Human Against Machine with 10000 training images](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Content**: High-quality dermatoscopic images of pigmented skin lesions across 7 categories.
- **License**: Open for academic use.

---

## 🛠️ Setup Instructions

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/skin-cancer-cnn.git
   cd skin-cancer-cnn
2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   pip install tensorflow keras pandas numpy matplotlib seaborn scikit-learn kaggle
3. **Set Up Kaggle API**:
   1.Go to your Kaggle account settings:
     👉 https://www.kaggle.com/account
   2.Click on "Create New API Token" – this downloads kaggle.json.
   3.Move the file to the correct location and set permissions:
   ```bash
   mkdir -p ~/.kaggle
   mv /path/to/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
   <sub>📝 If you're using Google Colab, use:</sub>
   ```bash
   from google.colab import files
   files.upload()  # Upload kaggle.json manually
4. **Download the Dataset**:
   ```bash
   kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
   unzip skin-cancer-mnist-ham10000.zip
5. **Run the Notebook**:
   1. Open Project_Notebook.ipynb in Jupyter Notebook or Google Colab.

   2. Execute each cell in sequence:
      - ✅ Import Libraries
      - ✅ Dataset Setup & Extraction
      - ✅ Preprocessing Images & Labels
      - ✅ Model Building (CNN with Keras)
      - ✅ Model Training & Evaluation
      - ✅ Visualizations and Metrics

## 🚀 Tech Stack
- Python 3.7+
- TensorFlow 2.x
- Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab (for GPU acceleration)

## 📊 Results
- Training Accuracy     : ~92%
- Validation Accuracy   : ~88%
- Evaluation Metrics:
    - 📉 Loss Curve (via Matplotlib)
    - 📈 Accuracy Plot
    - 🧮 Confusion Matrix
    - 🖼️ Sample Image Predictions
