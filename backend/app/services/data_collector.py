import serial
import time
import pandas as pd

# 전역 변수
ser = None
is_collecting = False
pressure_data = []
tilt_data = []

# 설정 변수
arduino_port = "COM3"
baud_rate = 115200
output_file = "./sensor_data_test1.csv"

def collect_data():
    global ser, is_collecting, pressure_data, tilt_data
    pressure_data.clear()
    tilt_data.clear()

    # 시리얼 포트 열기
    try:
        ser = serial.Serial(arduino_port, baud_rate, timeout=1)
        print(f"Connected to Arduino on port {arduino_port}")
    except serial.SerialException as e:
        print(f"Error: {e}")
        return

    start_time = time.time()
    current_turn = 0

    try:
        while is_collecting:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                current_time = round(time.time() - start_time, 2)

                if line.startswith("Pressure Sensor Value:"):
                    pressure = int(line.split(":")[1].strip())
                    pressure_data.append({"Elapsed Time (s)": current_time, "Pressure": pressure})
                    print(f"Time: {current_time}s, Pressure: {pressure}")

                elif line.startswith("Turn detected at:"):
                    parts = line.split()
                    turn_time = float(parts[3])
                    current_turn = int(parts[-1])
                    tilt_data.append({"Elapsed Time (s)": turn_time, "Tilt": current_turn})
                    print(f"Time: {turn_time}s, Tilt: {current_turn}")

    except Exception as e:
        print(f"Error during data collection: {e}")
    finally:
        ser.close()

def save_data_to_csv():
    global pressure_data, tilt_data

    # 데이터프레임 생성
    pressure_df = pd.DataFrame(pressure_data)
    tilt_df = pd.DataFrame(tilt_data)

    # 데이터 병합
    df = pd.merge_asof(
        pressure_df.sort_values("Elapsed Time (s)"),
        tilt_df.sort_values("Elapsed Time (s)"),
        on="Elapsed Time (s)",
        direction="backward"
    )
    df["Tilt"] = df["Tilt"].fillna(method="ffill").fillna(0)

    # CSV 저장
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
