## 🍇 Grape Leaf Disease Detection Project

Welcome to the **Grape Leaf Disease Detection** project! Our goal is to create model able to differentiate between a few classes of disease that afflicts grape plants. 

This model assist farmers and agricultural practitioners with early and accurate detection to improve precision farming practices.

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
- **10,200** training samples  
- **1,805** validation samples  
- **1,805** testing samples

```python
preprocess_input = tf.keras.applications.resnet50.preprocess_input

black_rot_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       zoom_range=0.33,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       rotation_range=360,
                                       brightness_range=[0.5,1.5],
                                       shear_range=7.5)

esca_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                  zoom_range=0.33,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  rotation_range=360,
                                  brightness_range=[0.5,1.5],
                                  shear_range=7.5)

leaf_blight_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                         zoom_range=0.33,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         rotation_range=360,
                                         brightness_range=[0.5,1.5],
                                         shear_range=7.5)

healthy_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                     zoom_range=0.33,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     rotation_range=360,
                                     brightness_range=[0.5,1.5],
                                     shear_range=7.5)
```

We also used data from this augmented dataset of grape leaves:
- https://www.kaggle.com/code/rm1000/grapevine-disease-detection-data-preprocessing

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

## Model Building Function
```python
def build_resnet50_model(input_shape, num_classes):
    # Load the ResNet50 base model with pretrained weights (ImageNet) excluding top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model layers (initial training only for top layers)
    base_model.trainable = False
    
    # Create a new model on top of the pretrained base
    model = Sequential([
        base_model,  # Pretrained base model
        GlobalAveragePooling2D(),  # Global average pooling layer
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),  # Fully connected layer
        BatchNormalization(),
        Dropout(0.6),  # Dropout for regularization
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # Fully connected layer
        BatchNormalization(),
        Dropout(0.6),  # Dropout for regularization
        Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))  # Output layer
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

    return model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
## Model Training Code
```python


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Model parameters
input_shape = (224, 224, 3)  # Image dimensions
num_classes = 4  # Total number of classes

# Build the ResNet50-based model
resnet50_model = build_resnet50_model(input_shape=input_shape, num_classes=num_classes)

# Print model summary
resnet50_model.summary()

# Define callbacks
resnet50_callbacks = [
    # Dynamic learning rate adjustment
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    
    # Early stopping
    EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True),
    
    # Model checkpoint to save best model based on validation accuracy
    ModelCheckpoint('/kaggle/working/resnet50_best_weights.keras', save_best_only=True, monitor='val_accuracy', mode='max')
]

# Train the ResNet50 model
history_resnet50 = resnet50_model.fit(
    train_datagen,
    validation_data=validation_datagen,
    epochs=100,
    callbacks=resnet50_callbacks
)

# Plot accuracy and loss curves for training and validation
plot_training_history(history_resnet50, model_name="ResNet50")

# Evaluate and report on the test data
pred_labels_resnet50, true_labels_resnet50 = evaluate_and_plot(resnet50_model, test_datagen, list_of_classes)

# Save the model
resnet50_model.save("resnet50_model.h5")
```

The notebook containing the full training code and step-by-step process is available on Kaggle:  
[Notebook Link](https://www.kaggle.com/code/stefanyeo/notebookada775ee24?scriptVersionId=213280957)

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

**Training Accuracy Graph** <br  >
![__results___6_11](https://github.com/user-attachments/assets/5a9cb822-6e78-4172-8edd-2a001d20bfd3)

**Training Loss Graph** <br  >
![__results___6_12](https://github.com/user-attachments/assets/5c0074f4-92e9-4007-8b80-882d56c129ca)

**AUC-ROC Curve Chart** <br  >
![__results___6_14](https://github.com/user-attachments/assets/3f514eb7-2673-404a-98b9-35976f2a44d8)

**Correct Prediction Sample** <br  >
![__results___6_16](https://github.com/user-attachments/assets/f4a88946-ed4b-43aa-88ad-a7a3b5827652)

**Incorrect Prediction Sample** <br  >
![__results___6_18](https://github.com/user-attachments/assets/5a0bb560-d6d5-436c-b403-8abf91cff5a5)

**Confusion Matrix Chart** <br  >
![__results___9_0](https://github.com/user-attachments/assets/e1f2917d-5143-4089-b485-529062aceb95)

---

## 💻 Deployment

To make the model accessible to end-users, a image docker is created and pushed into **IBM Cloud** environment so that it can be used as an API link. We've also deployed a web-based aplication that will allow user to upload and receive real-time disease classification result.

---

## 📚 Conclusion

This project has achieved what it set out to do which is to develop a AI model that is capable of classifying an user inputted image into a few disease classes. The evaluation metrics shows that the model is performing well. The model has also been deployed and can be accessed by users. Although the model is accurate at classifying the availble classes, the number of classes is still very limited. Future improvement in this project would involve capturing or procuring photos of different key of disease that affects grape plants.

This project also shows how deep learning can be applied in real-life scenario that may help solve people's problem.

---

## 🤝 Credits
This project is heavily inspired and utilizes many code from this Kaggle Notebook by Rajarshi Mandal
https://www.kaggle.com/code/rm1000/grapevine-disease-detection-model-training
