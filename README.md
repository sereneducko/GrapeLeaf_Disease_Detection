<h1 align="center">  Grape Leaf Disease Detection Project </h1>

<p align="center"> 
Ini adalah repository untuk contoh struktural yang bisa dipakai untuk melakukan dokumentasi Project Massive anda
</p>

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

## Teams

- Surgiwe (Design Researcher)
- Fernandi Lucky Putra (Data Engineer)
- Stefan Yeo (Machine Learning Engineer)
- Viriya Marhan Cunis (Machine Learning Engineer)
- Kevin Chandra Wijaya (Machine Learning Ops)
- Yudhistira Andilie (Machine Learning Ops)

## Idea Background

### 1. Theme
Tema : Agriculture (Precision Farming)

### 2. Problem
Masalah : Kurangnya pengalaman atau pengetahuan pada penyakit di tanaman anggur.

### 3. Solution
Solusi : Membuat model klasifikasi deep learning yang mampu mengklasifikasi empat kategori dari tanaman anggur.

## Dataset and Algorithm

### 1. Dataset
- Data Collection <br />
Data yang diperlukan untuk projek ini adalah data gambar daun tanaman anggur, yang dapat ditemukan di Kaggle.
Beberapa dataset yang digunakan adalah sebagai berikut:
https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original
https://www.kaggle.com/datasets/pushpalama/grape-disease

![plotall(1)](https://github.com/user-attachments/assets/c58d42f0-b6e8-40f5-a39c-2aa24b8152e8)

- Data Augmentation <br />
Data yang telah dikoleksi dilakukan augmentasi agar membuat model menjadi lebih robust dan banyak training datanya. 
Augmentasi yang dilakukan meliputi:
1. Horizontal dan Vertical Flipping
2. Zooming
3. Rotation
4. Shearing
5. Brightness Change

![black_rot_original_vs_augmented](https://github.com/user-attachments/assets/16b84fdd-d2f2-4cc9-8632-6c2e13ed9343)
![esca_original_vs_augmented](https://github.com/user-attachments/assets/5d83371c-29a2-41ce-9c4c-eb21a40d1932)
![healthy_original_vs_augmented](https://github.com/user-attachments/assets/99afa1af-2fa2-478a-ad2d-7096b867ba34)
![leaf_blight_original_vs_augmented](https://github.com/user-attachments/assets/608eeec6-9bc7-492a-8c6b-921b49442fe0)

Setelah dilakukan semua augmentasi dan koleksi data dikumpul pada dataset Kaggle berikut:
https://www.kaggle.com/datasets/stefanyeo/grape-vine-leaf-disease

Split data akhir untuk empat kelas adalah sebagai berikut:
12.000 training
1.805 validasi
1.805 training 

### 2. Algorithm

- Framework <br />
Kami menggunakan TensorFlow dan Keras sebagai lib pembangun model kami. Untuk proyek ini kami memutuskan untuk menggunakan ResNet50
Alasannya adalah cocok dengan ukuran dataset yang kita miliki (medium to large dataset), serta memiliki track record performa yang bagus.

Karena dataset berada di Kaggle, maka training juga dilakukan di cloud environment Kaggle untuk mempermudah proses training.

- Pembangunan Model <br />
Masukkan kode training dan juga spesifikasi model, seperti epoch, learning rate, batch size, dan lain sebagainya.

- Model Evaluation <br />
Masukkan metrik evaluasi model seperti accuracy, precision, recall, F1-score, dan lain - lain.

## Prototype
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.

## Integration
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.

## Deployment
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.

## Result
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.

## Conclusion
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.
