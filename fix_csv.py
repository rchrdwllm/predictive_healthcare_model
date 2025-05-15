import pandas as pd
import os


columns = [
    'Date', 'Name', 'Age', 'Gender', 'Location', 'Disease'
] + [f'Symptom_{i}' for i in range(1, 18)]

csv_path = 'dataset/sample_user_data.csv'
fixed_csv_path = 'dataset/sample_user_data_fixed.csv'

if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    exit(1)


try:
    df = pd.read_csv(csv_path, names=columns, header=0, on_bad_lines='skip')
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)


df = df[columns]


df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y", errors='coerce').dt.strftime('%Y-%m-%d')


df.to_csv(fixed_csv_path, index=False)
print(f"Cleaned CSV written to {fixed_csv_path}")


os.replace(fixed_csv_path, csv_path)
print("Original CSV replaced with cleaned version.")