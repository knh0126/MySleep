from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse
import io
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, BackgroundTasks
from app.services.data_collector import collect_data, save_data_to_csv, is_collecting
from app.services.data_analyzer import analyze_sleep_data, get_csv_file_path
import matplotlib.pyplot as plt

app = FastAPI()
security = HTTPBearer()


# 간단한 사용자 데이터베이스 시뮬레이션
users = {}
tokens = {}

class User(BaseModel):
    email: str
    password: str

class SleepAction(BaseModel):
    action: str

@app.post("/signup")
def signup(user: User):
    if user.email in users:
        raise HTTPException(status_code=400, detail="User already exists")
    users[user.email] = user.password
    return {"message": "Signup successful"}

@app.post("/login")
def login(user: User):
    if users.get(user.email) != user.password:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = f"token-{user.email}"
    tokens[token] = user.email
    return {"token": token}

# 전역 변수
analysis_results = None

@app.post("/start/")
def start_data_collection(background_tasks: BackgroundTasks):
    global is_collecting
    if is_collecting:
        return {"status": "Data collection is already running."}
    is_collecting = True
    background_tasks.add_task(collect_data)
    return {"status": "Data collection started."}

@app.post("/stop/")
def stop_data_collection():
    global is_collecting
    if not is_collecting:
        return {"status": "Data collection is not running."}
    is_collecting = False
    save_data_to_csv()
    return {"status": "Data collection stopped and saved.", "file": "./sensor_data_test1.csv"}

@app.post("/analyze/")
def analyze_data():
    global analysis_results
    try:
        analysis_results = analyze_sleep_data()
        return {"status": "Analysis completed."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/summary/")
def get_summary():
    if analysis_results is None:
        raise HTTPException(status_code=400, detail="No analysis data found. Run /analyze/ first.")
    return {
        "sleep_efficiency": f"{analysis_results['sleep_efficiency']:.2f}%",
        "deep_nrem_ratio": f"{analysis_results['deep_nrem_ratio']:.2f}%",
        "rem_ratio": f"{analysis_results['rem_ratio']:.2f}%",
        "total_sleep_time": f"{analysis_results['total_sleep_time']:.2f} seconds"
    }
