# Crop Disease Detection System

A Machine Learning-based web application that detects crop diseases from images and provides predictions using a trained deep learning model.

---

## Features

- Upload crop images for disease detection  
- Predicts disease using a trained deep learning model  
- Uses pre-trained and fine-tuned models (.h5 / .keras)  
- Fast and easy-to-use interface  
- Helps in early detection of plant diseases  

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- Flask (if used in `app.py`)  
- NumPy & JSON  

---

## Project Structure


crop_disease/
│── app.py # Main application
│── train.py # Model training script
│── requirements.txt # Dependencies
│── class_names.json # Class labels
│── .gitignore
│── models/ (recommended)

---

## Installation & Setup

### 1. Clone the repository


git clone https://github.com/LabhanshiBhatia/crop_disease.git

cd crop_disease


---

### 2. Create virtual environment


python -m venv cropenv
source cropenv/Scripts/activate # Windows


---

### 3. Install dependencies


pip install -r requirements.txt


---

### 4. Run the application


python app.py
