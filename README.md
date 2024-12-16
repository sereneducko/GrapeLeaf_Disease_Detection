![__results___6_14](https://github.com/user-attachments/assets/bba7c470-5af2-460f-aa27-3c8f67603cbf)# 🍇 Grape Leaf Disease Detection Project

Welcome to the **Grape Leaf Disease Detection** project! Our goal is to assist farmers and agricultural practitioners with early and accurate detection to improve precision farming practices.

---

## 🛠️ Technologies Used

<div align="center">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
    <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white">
    <img src="https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white">
    <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white">
    <img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white">
    <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
    <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
    <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white">
    <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black">
</div>

---

## 👥 Project Team

- **Surgiwe** (Design Researcher)
- **Fernandi Lucky Putra** (Data Engineer)
- **Stefan Yeo** (Machine Learning Engineer)
- **Viriya Marhan Cunis** (Machine Learning Engineer)
- **Kevin Chandra Wijaya** (ML Ops)
- **Yudhistira Andilie** (ML Ops)

---

## 🌟 Idea Background

### **Theme**  
Agriculture - **Precision Farming**

### **Problem**  
Farmers often lack the expertise or knowledge required to identify grapevine leaf diseases, leading to reduced crop yields and quality.

### **Solution**  
Develop a deep learning classification model capable of identifying grapevine leaf diseases with high accuracy. The model will classify images of grapevine leaves into four categories to assist farmers with timely and accurate diagnoses.

---

## 📂 Dataset and Preprocessing

### **Dataset**
The dataset used for this project comprises grapevine leaf images sourced from Kaggle:
- [Grape Disease Dataset (Original)](https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original)
- [Grape Disease](https://www.kaggle.com/datasets/pushpalama/grape-disease)

To ensure robust model training, data augmentation techniques were applied to expand the dataset.  
Final dataset split:
- **12,000** training samples  
- **1,805** validation samples  
- **1,805** testing samples  

Find the final processed dataset [here](https://www.kaggle.com/datasets/stefanyeo/grape-vine-leaf-disease).

### **Data Augmentation**  
The following augmentations were applied to improve model performance:
- Horizontal and vertical flipping
- Zooming
- Rotation
- Shearing
- Brightness adjustments

#### Example Augmented Images:
- **Black Rot Disease**  
![black_rot](https://github.com/user-attachments/assets/16b84fdd-d2f2-4cc9-8632-6c2e13ed9343)
- **Esca Disease**  
![esca](https://github.com/user-attachments/assets/5d83371c-29a2-41ce-9c4c-eb21a40d1932)
- **Healthy Leaves**  
![healthy](https://github.com/user-attachments/assets/99afa1af-2fa2-478a-ad2d-7096b867ba34)
- **Leaf Blight Disease**  
![leaf_blight](https://github.com/user-attachments/assets/608eeec6-9bc7-492a-8c6b-921b49442fe0)

---

## 🧠 Algorithms and Frameworks

### **Frameworks**
We leveraged **TensorFlow** and **Keras** for building the deep learning model due to their:
- Ease of use
- Extensive documentation and community support
- Integrated tools for data preprocessing and augmentation  

Training was conducted using Kaggle's cloud environment for streamlined workflows.

### **Model Architecture**
After testing multiple architectures, including CNN, EfficientNetB7, and MobileNetV2, we selected **ResNet50**:
- Well-suited for medium to large datasets
- High accuracy with moderate computational requirements

#### **Training Configuration:**
- Epochs: **70**
- Batch Size : **16**
- Optimizer: **Adam** with a learning rate scheduler  
- Loss Function: **Categorical Crossentropy**  

### **Model Evaluation**
Model performance was evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC Curve

---

## 🚀 Results

The final model achieved:
- **Training Accuracy:** 97.8%  
- **Validation Accuracy:** 97.3%  
- **Test Accuracy:** 97.5%  

- **Training Accuracy Graph**
![__results___6_11](https://github.com/user-attachments/assets/5a9cb822-6e78-4172-8edd-2a001d20bfd3)

- **Training Loss Graph**
![__results___6_12](https://github.com/user-attachments/assets/5c0074f4-92e9-4007-8b80-882d56c129ca)

- **AUC-ROC Curve**
![__results___6_14](https://github.com/user-attachments/assets/3f514eb7-2673-404a-98b9-35976f2a44d8)

- **Correct Prediction Sample**
![__results___6_16](https://github.com/user-attachments/assets/f4a88946-ed4b-43aa-88ad-a7a3b5827652)

- **Incorrect Prediction Sample**
![__results___6_18](https://github.com/user-attachments/assets/5a0bb560-d6d5-436c-b403-8abf91cff5a5)

- **Confusion Matrix**
![__results___9_0](https://github.com/user-attachments/assets/e1f2917d-5143-4089-b485-529062aceb95)

---

## 💻 Deployment

To make the model accessible to end-users, a image docker is created and pushed into **IBM Cloud** environment so that it can be used as an API link.

---

## 📊 Prototype

A detailed notebook containing the full implementation and step-by-step process is available on Kaggle:  
[Notebook Link](https://www.kaggle.com/code/stefanyeo/notebookada775ee24?scriptVersionId=213280957)

---

## 📚 Conclusion

This project demonstrates how deep learning can be applied to address real-world agricultural challenges. By providing a scalable, accurate, and user-friendly solution, we aim to empower farmers and contribute to sustainable farming practices.

---

### 💡 Future Work
- Expand the dataset with additional disease categories
- Integrate IoT for real-time field analysis
- Deploy the solution as a mobile application for broader reach

---

## 🤝 Contributions

Contributions are welcome! Please feel free to fork this repository, submit issues, or create pull requests to improve the project.
