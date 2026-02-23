from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="HR Analytics Enterprise API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, "models", "attrition_model.pkl")
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")

print("BASE_DIR =", BASE_DIR)
print("FRONTEND_DIR =", FRONTEND_DIR)

# Mount Static Files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

EMPLOYEE_CSV = os.path.join(BASE_DIR, "employees.csv")
PERFORMANCE_CSV = os.path.join(BASE_DIR, "performance.csv")
USERS_CSV = os.path.join(BASE_DIR, "users.csv")

print("EMPLOYEE_CSV =", EMPLOYEE_CSV)
print("EMPLOYEE_CSV exists =", os.path.exists(EMPLOYEE_CSV))
print("PERFORMANCE_CSV exists =", os.path.exists(PERFORMANCE_CSV))

# Initialize Users CSV if not exists
if not os.path.exists(USERS_CSV):
    pd.DataFrame(columns=["Name", "Email", "PasswordHash"]).to_csv(USERS_CSV, index=False)


# Load Model
model = None
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model not found at {MODEL_PATH}. Predictions will fail until trained. Error: {e}")

### Models
class NewEmployee(BaseModel):
    Name: str
    Age: int
    Department: str
    Role: str
    Salary: int
    JoiningYear: int
    Gender: str

class NewPerformance(BaseModel):
    EmployeeID: int
    Rating: int
    ProjectsCompleted: int
    AvgDailyHours: int
    Attrition: int = 0
    Reason: Optional[str] = ""

class EmployeeInput(BaseModel):
    Age: int
    Department: str
    Role: str
    Salary: int
    JoiningYear: int
    Gender: str
    Rating: int
    ProjectsCompleted: int
    AvgDailyHours: float

class UserRegister(BaseModel):
    Name: str
    Email: str
    Password: str

class UserLogin(BaseModel):
    Email: str
    Password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_name: str

# Security Configuration
SECRET_KEY = "hr-nexus-enterprise-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.get("/")
def login():
    return FileResponse(os.path.join(FRONTEND_DIR, "login.html"))

@app.get("/dashboard")
def dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.post("/api/employees")
def add_employee(emp: NewEmployee):
    try:
        df = pd.read_csv(EMPLOYEE_CSV)
        new_id = df['EmployeeID'].max() + 1 if not df.empty else 1001
        
        new_row = {
            "EmployeeID": new_id,
            "Name": emp.Name,
            "Age": emp.Age,
            "Department": emp.Department,
            "Role": emp.Role,
            "Salary": emp.Salary,
            "JoiningYear": emp.JoiningYear,
            "Gender": emp.Gender
        }
        
        # Append to CSV
        new_df = pd.DataFrame([new_row])
        new_df.to_csv(EMPLOYEE_CSV, mode='a', header=False, index=False)
        
        return {"message": "Employee added successfully", "employee_id": int(new_row['EmployeeID'])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/performance")
def add_performance(perf: NewPerformance):
    try:
        # Check if employee exists
        emp_df = pd.read_csv(EMPLOYEE_CSV)
        if perf.EmployeeID not in emp_df['EmployeeID'].values:
            raise HTTPException(status_code=404, detail="Employee ID not found")
            
        new_row = {
            "EmployeeID": perf.EmployeeID,
            "Rating": perf.Rating,
            "ProjectsCompleted": perf.ProjectsCompleted,
            "AvgDailyHours": perf.AvgDailyHours,
            "Attrition": perf.Attrition,
            "Reason": perf.Reason if perf.Attrition == 1 else ""
        }
        
        pd.DataFrame([new_row]).to_csv(PERFORMANCE_CSV, mode='a', header=False, index=False)
        
        return {"message": "Performance record added successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics")
def get_analytics():
    try:
        emp = pd.read_csv(os.path.join(DATA_DIR, "employees.csv"))
        perf = pd.read_csv(os.path.join(DATA_DIR, "performance.csv"))
        df = pd.merge(emp, perf, on="EmployeeID")
        
        return {
            "total_employees": len(df),
            "avg_rating": round(df['Rating'].mean(), 2),
            "attrition_rate": round(df['Attrition'].mean() * 100, 1),
            "dept_counts": df['Department'].value_counts().to_dict(),
            "attrition_by_dept": df[df['Attrition'] == 1]['Department'].value_counts().to_dict(),
            "role_counts": df['Role'].value_counts().to_dict(),
            "salary_rating_data": df[['Salary', 'Rating']].sample(n=min(100, len(df))).to_dict(orient='records'),
            
            # New Analytics
            "salary_by_dept": df.groupby("Department")['Salary'].mean().round(0).to_dict(),
            "rating_counts": df['Rating'].value_counts().sort_index().to_dict(),
            "joining_year_counts": df['JoiningYear'].value_counts().sort_index().to_dict(),
            "attrition_reasons": df[df['Attrition'] == 1]['Reason'].value_counts().to_dict(),
            "departments": sorted(df['Department'].unique().tolist())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/department/{department}")
def get_department_analytics(department: str):
    """Get analytics filtered by specific department"""
    try:
        emp = pd.read_csv(os.path.join(DATA_DIR, "employees.csv"))
        perf = pd.read_csv(os.path.join(DATA_DIR, "performance.csv"))
        df = pd.merge(emp, perf, on="EmployeeID")
        
        # Filter by department
        dept_df = df[df['Department'] == department]
        
        if dept_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for department: {department}")
        
        return {
            "department": department,
            "total_employees": len(dept_df),
            "avg_rating": round(dept_df['Rating'].mean(), 2),
            "attrition_rate": round(dept_df['Attrition'].mean() * 100, 1),
            "avg_salary": round(dept_df['Salary'].mean(), 0),
            "role_counts": dept_df['Role'].value_counts().to_dict(),
            "rating_counts": dept_df['Rating'].value_counts().sort_index().to_dict(),
            "joining_year_counts": dept_df['JoiningYear'].value_counts().sort_index().to_dict(),
            "salary_rating_data": dept_df[['Salary', 'Rating']].to_dict(orient='records'),
            "gender_distribution": dept_df['Gender'].value_counts().to_dict(),
            "attrition_count": int(dept_df['Attrition'].sum()),
            "attrition_reasons": dept_df[dept_df['Attrition'] == 1]['Reason'].value_counts().to_dict(),
            "avg_projects": round(dept_df['ProjectsCompleted'].mean(), 1),
            "avg_daily_hours": round(dept_df['AvgDailyHours'].mean(), 1)
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict-attrition")
def predict(data: EmployeeInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([data.dict()])
    try:
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
        
        return {
            "prediction": "Risk" if pred == 1 else "Safe",
            "probability": prob,
            "risk_level": "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/register")
def register(user: UserRegister):
    try:
        users_df = pd.read_csv(USERS_CSV)
        if user.Email in users_df['Email'].values:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Secure Hash
        hashed_pw = get_password_hash(user.Password)
        
        new_user = {
            "Name": user.Name,
            "Email": user.Email,
            "PasswordHash": hashed_pw
        }
        
        pd.DataFrame([new_user]).to_csv(USERS_CSV, mode='a', header=False, index=False)
        return {"message": "User registered successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/login", response_model=Token)
def login_api(user: UserLogin):
    try:
        users_df = pd.read_csv(USERS_CSV)
        user_record = users_df[users_df['Email'] == user.Email]
        
        if user_record.empty:
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
        stored_hash = user_record.iloc[0]['PasswordHash']
        
        # Verify Password
        if not verify_password(user.Password, stored_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create JWT
        access_token = create_access_token(data={"sub": user.Email})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_name": user_record.iloc[0]['Name']
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
