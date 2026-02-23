# HR Enterprise Data Science System

## Overview
Full-stack application for Employee Analytics & Attrition Prediction.
- **Backend:** FastAPI (Python)
- **ML Engine:** Scikit-Learn (Random Forest)
- **Frontend:** Vanilla JS + Tailwind CSS (Glassmorphism UI)

## Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Generation (Automated)**
   The `setup_env.py` script has already generated synthetic data in `data/`.
   If you need to regenerate:
   ```bash
   python ../setup_env.py
   ```

3. **Train AI Model**
   Before running the API, train the prediction model:
   ```bash
   python backend/train_model.py
   ```

4. **Run Backend API**
   Star the FastAPI server:
   ```bash
   uvicorn backend.main:app --reload
   ```
   Server will run at: `http://localhost:8000`

5. **Launch Frontend**
   Open `frontend/index.html` in your web browser.

## Features
- **Dashboard:** Real-time metrics on employees, attrition, and performance.
- **Prediction:** AI-powered form to assess employee flight risk.
- **Architecture:** Clean separation of concerns (Data layer -> ML Model -> API -> UI).
