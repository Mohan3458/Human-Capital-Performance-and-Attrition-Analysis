# HR Employee Attrition Prediction & Analytics Dashboard

A Full-Stack HR Analytics Web Application that predicts employee attrition using Machine Learning and provides interactive workforce insights through a dashboard.

---

##  Project Overview

This project is designed to help HR teams analyze employee data and predict attrition risk using a trained Machine Learning model.  

The system combines:
- Machine Learning model for prediction
- Flask backend for API handling
- Interactive frontend dashboard
- Authentication system
- Real-time prediction integration

##  Features

- Employee Attrition Prediction using ML model
- Data preprocessing and feature engineering
- Interactive HR analytics dashboard
- User login authentication system
- Model serialization using Pickle (.pkl)
- REST API integration between frontend and backend
- CSV-based employee data management

---

## Tech Stack

### Backend
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy

### Frontend
- HTML
- CSS
- JavaScript

### Machine Learning
- Data Preprocessing
- Feature Encoding
- Model Training
- Model Evaluation
- Pickle for model saving

##  Machine Learning Workflow

1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Selection & Encoding
4. Model Training
5. Model Evaluation
6. Model Deployment using Flask


## 📂 Project Structure

project/
│
├── model/
│   ├── model.pkl
│
├── static/
│   ├── css/
│   ├── js/
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── login.html
│
├── app.py
├── requirements.txt
└── README.md


## ⚙️ Installation & Setup

1️ Clone the repository

```
git clone https://github.com/your-username/hr-attrition-project.git
cd hr-attrition-project
```

2️ Create virtual environment

```
python -m venv venv
```

3️ Activate virtual environment

Windows:
```
venv\Scripts\activate
```

Mac/Linux:
```
source venv/bin/activate
```

4️ Install dependencies

```
pip install -r requirements.txt
```

5️ Run the application

```
python app.py
```

6️ Open in browser

```
http://127.0.0.1:5000/
```

---

## 📈 Use Case

This system can help:
- HR teams identify high attrition risk employees
- Organizations improve retention strategies
- Data teams analyze workforce trends

---

##  Authentication

The system includes:
- User login functionality
- Session handling
- Secure routing

---

##  Future Improvements

- Deploy on AWS / Render / Railway
- Add database integration (MySQL / PostgreSQL)
- Improve UI with React
- Add more ML models for comparison
- Add real-time analytics charts

---

## Author

Developed by [Mohan K]

---

## 📌 License

This project is open-source and available for learning and demonstration purposes.
