import pandas as pd
import json
from tqdm import tqdm

def timestamp_to_json(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # 转换为 ISO 格式的字符串
    raise TypeError(f"Type {type(obj)} not serializable")

def get_epoch(days,hours, minutes):
    return ((days-22) * 24 * 60 + hours * 60 + minutes) // 5

# 读取 Parquet 文件
df = pd.read_parquet('green_tripdata_2016-02.parquet')

# 确保日期列是 datetime 类型，进行转换
df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
pickup_day = df['lpep_pickup_datetime'].dt.day.astype(int)
dropoff_day = df['lpep_dropoff_datetime'].dt.day.astype(int)
# 设置日期范围
start_date = 22
end_date = 28

# 过滤日期范围，假设你想筛选pickup和dropoff时间都在2016-02-22到2016-02-28之间
filtered_df = df[
    (pickup_day >= start_date) &
    (pickup_day <= end_date) &
    (dropoff_day >= start_date) &
    (dropoff_day <= end_date)
]

# 去掉PULocationID和DOLocationID相同的行
filtered_df = filtered_df[filtered_df['PULocationID'] != filtered_df['DOLocationID']]
filtered_df['pickup_epoch'] = 0
filtered_df['dropoff_epoch'] = 0
data = {}
for i in tqdm(range(len(filtered_df))):
    record = filtered_df.iloc[i]
    days = record['lpep_pickup_datetime'].day
    hours = record['lpep_pickup_datetime'].hour
    minutes = record['lpep_pickup_datetime'].minute
    start_epoch = get_epoch(days, hours, minutes)
    record['pickup_epoch'] = start_epoch
    days = record['lpep_dropoff_datetime'].day
    hours = record['lpep_dropoff_datetime'].hour
    minutes = record['lpep_dropoff_datetime'].minute
    end_epoch = get_epoch(days, hours, minutes)
    record['dropoff_epoch'] = end_epoch
    if start_epoch not in data.keys():
        data[start_epoch] = []
    record = record.copy()
    data[start_epoch].append(record.to_dict())

with open("data.json", "w") as f:
    json.dump(data, f,default=timestamp_to_json)