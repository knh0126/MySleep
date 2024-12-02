import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch.nn as nn
from typing import Any, Dict

# CSV_FILE = "./sensor_data_test1.csv"
CSV_FILE = "app/data_6hr_1.csv"

def get_csv_file_path() -> str:
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV file '{CSV_FILE}' does not exist. Please start data collection first.")
    return CSV_FILE

def analyze_sleep_data() -> Dict[str, Any]:
    csv_path = get_csv_file_path()
    data_6hr = pd.read_csv(csv_path)

    data_ae = data_6hr.copy()

    def classify_sleep_stage_with_anomalies(row, threshold_low, threshold_mid, threshold_high):
        if row['Reconstruction Error'] < threshold_low:
            return 'Deep NREM' if not row['Anomaly'] else 'REM'
        elif threshold_low <= row['Reconstruction Error'] < threshold_mid:
            return 'REM'
        elif threshold_mid <= row['Reconstruction Error'] < threshold_high:
            return 'Light NREM'
        else:
            return 'Awake'
    # 지표 계산 함수
    def calculate_sleep_metrics(data):
        total_turns = data["Tilt"].iloc[-1] - data["Tilt"].iloc[0]
        pressure_changes = data["Pressure"].diff().abs().sum()
        tilt_changes = data["Tilt"].diff().abs().sum()
        stability_index = 1 - ((tilt_changes + pressure_changes) / len(data))
        return total_turns, stability_index

    # 지표 계산
    total_turns, stability_index = calculate_sleep_metrics(data_ae)
    data_ae["Total Turns"] = total_turns
    data_ae["Stability Index"] = stability_index
    data_ae['Current Tilt Change'] = data_ae['Tilt'].diff().fillna(0)
    data_ae["Elapsed Time (s)"] = np.floor(data_ae["Elapsed Time (s)"])

    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_ae)
    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)

    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, encoding_dim), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(encoding_dim, input_dim), nn.Sigmoid())

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    input_dim = data_tensor.shape[1]
    encoding_dim = 3
    model = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    batch_size = 64
    data_loader = torch.utils.data.DataLoader(data_tensor, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in data_loader:
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Reconstruction error 계산
    with torch.no_grad():
        reconstructed_data = model(data_tensor)
        reconstruction_error = torch.mean((reconstructed_data - data_tensor) ** 2, dim=1).numpy()

    # Reconstruction error를 데이터프레임에 추가
    data_ae['Reconstruction Error'] = reconstruction_error

    # 이상치 여부 추가 (Threshold 초과 여부)
    threshold = np.percentile(reconstruction_error, 95)  # 상위 5%를 이상치로 간주
    data_ae['Anomaly'] = data_ae['Reconstruction Error'] > threshold
    # KMeans 클러스터링 (4개의 클러스터: 깊은, 렘, 얕은, 깨어있는)

    # 2. KMeans를 사용하여 Reconstruction Error를 4개의 클러스터로 나누기
    n_clusters = 4  # 수면 단계를 나타내는 클러스터 수 (Deep NREM, REM, Light NREM, Awake)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data_ae['Cluster'] = kmeans.fit_predict(data_ae[['Reconstruction Error', "Current Tilt Change","Pressure"]])  # 클러스터 할당

    # 3. 클러스터 중심값 확인 및 Threshold 계산
    cluster_centers = kmeans.cluster_centers_[:, 0]  # Reconstruction Error 중심값 추출
    cluster_centers.sort()  # 중심값을 오름차순으로 정렬

    # Threshold 계산: 각 클러스터 중심값 사이의 중간값을 기준으로 설정
    thresholds = [(cluster_centers[i] + cluster_centers[i + 1]) / 2 for i in range(len(cluster_centers) - 1)]

    # Threshold 범위 확장 (0부터 최대 Reconstruction Error까지 포함)
    thresholds = [0] + thresholds + [data_ae['Reconstruction Error'].max()]

    # 4. 클러스터를 수면 단계에 매핑
    cluster_labels = ['Deep NREM', 'REM', 'Light NREM', 'Awake']  # 수면 단계 이름
    data_ae['Sleep Stage'] = pd.cut(
        data_ae['Reconstruction Error'],
        bins=thresholds,  # Threshold를 범위로 사용
        labels=cluster_labels,  # 각 클러스터를 수면 단계로 매핑
        include_lowest=True  # 최저값 포함
    )
    threshold_low=thresholds[1]
    threshold_mid=thresholds[2]
    threshold_high=thresholds[3]
    def classify_sleep_stage_with_anomalies(row, threshold_low, threshold_mid, threshold_high):
        if row['Reconstruction Error'] < threshold_low:
            return 'Deep NREM' if not row['Anomaly'] else 'REM'
        elif threshold_low <= row['Reconstruction Error'] < threshold_mid:
            return 'REM'
        elif threshold_mid <= row['Reconstruction Error'] < threshold_high:
            return 'Light NREM'
        else:
            return 'Awake'

    # 수면 단계 분류
    data_ae['Sleep Stage'] = data_ae.apply(
        lambda row: classify_sleep_stage_with_anomalies(row, threshold_low, threshold_mid, threshold_high),
        axis=1
    )

    # 결과 요약
    sleep_stage_summary = data_ae['Sleep Stage'].value_counts(normalize=True) * 100

    stage_durations = data_ae.groupby('Sleep Stage')['Elapsed Time (s)'].count()
    total_sleep_time = data_ae['Elapsed Time (s)'].max() - data_ae['Elapsed Time (s)'].min()

    awake_time = stage_durations.loc['Awake'] if 'Awake' in stage_durations.index else 0



    sleep_efficiency = ((total_sleep_time - awake_time) / total_sleep_time) * 100

    sleep_stage_summary = data_ae['Sleep Stage'].value_counts(normalize=True) * 100

    print(data_ae['Sleep Stage'].unique())  # Sleep Stage에 어떤 상태가 있는지 확인
    print(stage_durations)  # stage_durations 내용 확인
    print(f"수면 효율: {sleep_efficiency:.2f}%")
    print(f"awake_time: {awake_time:.2f}s")
    print(f"total_sleep: {total_sleep_time:.2f}s")

    return {
        "data_ae": data_ae,
        "total_sleep_time": total_sleep_time,
        "sleep_efficiency": max(0, min(sleep_efficiency, 100)),
        "deep_nrem_ratio": sleep_stage_summary.get('Deep NREM', 0),
        "rem_ratio": sleep_stage_summary.get('REM', 0),
        "stage_ratios": sleep_stage_summary
    }
