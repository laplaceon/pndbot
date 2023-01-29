import matplotlib.pyplot as plt
import pandas as pd

landmarks = pd.read_csv("../data/chart_landmarks_auto_overlapped.csv")
landmarks['start'] = pd.to_datetime(landmarks['start'])
landmarks['end'] = pd.to_datetime(landmarks['end'])

for i, row in landmarks.iterrows():
    file = pd.read_csv(f"../data/full/binance/{row['id']}")

    third = int(len(file["timestamp"]) / 5)
    
    bounds = (int(row['start'].timestamp()) * 1000, int(row['end'].timestamp()) * 1000)

    plt.xlabel("timestamp")
    plt.ylabel("price")
    plt.title(row["id"])
    plt.axvspan(bounds[0], bounds[1], alpha=0.5, lw=0)
    plt.grid()
    plt.plot(file["timestamp"], file["price"])
    plt.show()