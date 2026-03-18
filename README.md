# 🐝 Bee Audio Classification System

A machine learning-based system for classifying bee sounds into **worker bees** and **drone bees** using audio feature extraction and multiple classification models.

---

## 📌 Project Overview

This project focuses on detecting and classifying bee types based on their **acoustic signals**. By analyzing bee sounds using signal processing techniques, the system extracts meaningful features and applies machine learning models to perform classification.

This work is particularly useful for:

* Monitoring hive health 🐝
* Detecting **swarming behavior**
* Supporting **precision beekeeping**

---

## ⚙️ Features

* 🎧 Audio feature extraction using **Librosa**
* 📊 Multiple feature types:

  * 13 MFCCs (Mel-Frequency Cepstral Coefficients)
  * Zero Crossing Rate (ZCR)
  * Spectral Centroid
  * LPC & other derived features (from jAudio dataset)
* 🤖 Machine Learning Models:

  * Random Forest
  * Decision Tree
  * Logistic Regression
  * Gaussian Naive Bayes
  * Bernoulli Naive Bayes
* 🧠 Deep Learning:

  * Autoencoder for feature compression
* 📈 Performance Evaluation:

  * Accuracy
  * Classification Report
  * Confusion Matrix

---

## 🗂️ Project Structure

```
bee-audio-classification/
│
├── src/
│   ├── feature_extraction.py
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── autoencoder.py
│
├── main.py                 # Main pipeline script
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/bee-audio-classification.git
cd bee-audio-classification
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the Pipeline

```bash
python main.py
```

---

## 🔬 Methodology

### 1. Feature Extraction

Audio signals are processed using **Librosa** to extract:

* MFCCs (captures timbral texture)
* ZCR (measures signal noisiness)
* Spectral features (frequency distribution)

---

### 2. Data Preprocessing

* Cleaning missing values
* Encoding labels (Worker vs Drone)
* Train-test split

---

### 3. Model Training

Multiple classifiers are trained and compared:

* Tree-based models (RF, DT)
* Linear models (LR)
* Probabilistic models (Naive Bayes)

---

### 4. Evaluation

Each model is evaluated using:

* Accuracy score
* Precision, Recall, F1-score
* Confusion matrix

---

### 5. Autoencoder (Advanced)

An autoencoder is used to:

* Reduce feature dimensionality
* Improve model performance on compressed representations

---

## 📊 Results Summary

* High classification accuracy achieved across multiple models
* Random Forest and Decision Tree performed exceptionally well
* Autoencoder-based feature reduction maintained strong performance

---

## 🔮 Future Work

* 🌐 Deploy as a **Flask web application**
* 📱 Real-time bee monitoring system
* 🔀 Implement **multimodal learning (audio + image)**
* 🔬 Explore **early fusion and late fusion techniques**
* 📡 Integrate IoT-based hive sensors

---

## 📚 Technologies Used

* Python 🐍
* Librosa 🎧
* Scikit-learn 🤖
* TensorFlow / Keras 🧠
* Pandas & NumPy 📊
* Matplotlib & Seaborn 📈

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a pull request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👤 Author

**KM Rwarinda**

* 🎓 BSc Computer Science
* 🔬 Research focus: AI in bioacoustics & multimodal learning

---

## ⭐ Acknowledgements

* Librosa for audio processing
* Scikit-learn for machine learning tools
* Bee research datasets and open-source contributions

---

## 💡 Note

This project is part of ongoing research into **AI-driven detection of biological patterns**, with a focus on improving agricultural and environmental monitoring systems.
