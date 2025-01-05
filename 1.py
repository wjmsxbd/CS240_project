import pandas as pd
import h3
import json
from tqdm import tqdm
import copy

def timestamp_to_json(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # 转换为 ISO 格式的字符串
    raise TypeError(f"Type {type(obj)} not serializable")

def get_epoch(days, hours, minutes):
    return ((days-21) * 24 * 60 + hours * 60 + minutes) // 5

# 读取 Parquet 文件
df = pd.read_csv('yellow_tripdata_2016-02.csv')
# df = pd.read_parquet('green_tripdata_2016-02.parquet')

# 确保日期列是 datetime 类型，进行转换
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
pickup_day = df['tpep_pickup_datetime'].dt.day.astype(int)
dropoff_day = df['tpep_dropoff_datetime'].dt.day.astype(int)
# 设置日期范围
start_date = 21
end_date = 28

# 过滤日期范围，假设你想筛选pickup和dropoff时间都在2016-02-22到2016-02-28之间
filtered_df = df[
    (pickup_day >= start_date) &
    (pickup_day <= end_date) &
    (dropoff_day >= start_date) &
    (dropoff_day <= end_date)
]

print(filtered_df)

nyc_center_long = -74.0060
nyc_center_lat = 40.7128
center_cell = h3.latlng_to_cell(nyc_center_lat, nyc_center_long, 8)
hexagons = h3.grid_disk(center_cell, 18)
selected_hexagons = list(hexagons)[:370]

nyc_center_long = -74.0060
nyc_center_lat = 40.7128
center_cell = h3.latlng_to_cell(nyc_center_lat, nyc_center_long, 8)
hexagons = h3.grid_disk(center_cell, 18)
selected_hexagons = list(hexagons)[:370]

def convert_to_h3_index(lng, lat, resolution):
    # 使用h3.latlng_to_cell函数直接将经纬度转换为H3索引
    h3_index = h3.latlng_to_cell(lat, lng, resolution)
    #return h3_index
    
    return h3_index if h3_index in selected_hexagons else -1

def filter_out_of_bounds(data):
    # 纽约市的边界坐标
    bounds = {
        'min_longitude': -74.3280,
        'max_longitude': -73.6317,
        'min_latitude': 40.5503,
        'max_latitude': 40.8445
    }
    
    # 将边长转换为单位
    res = 8
    
    
    # 过滤掉超出边界的行程
    data['pickup_h3_index'] = data.apply(lambda row: convert_to_h3_index(row['pickup_longitude'], row['pickup_latitude'], res), axis=1)
    data['dropoff_h3_index'] = data.apply(lambda row: convert_to_h3_index(row['dropoff_longitude'], row['dropoff_latitude'], res), axis=1)
    
    data = data[data["pickup_h3_index"] != -1]
    data = data[data["dropoff_h3_index"] != -1]
    data = data[data["pickup_h3_index"] != data["dropoff_h3_index"]]
    
    data = data[(data['pickup_longitude'] >= bounds['min_longitude']) & 
               (data['pickup_longitude'] <= bounds['max_longitude']) &
               (data['pickup_latitude'] >= bounds['min_latitude']) & 
               (data['pickup_latitude'] <= bounds['max_latitude'])]
    data = data[(data['dropoff_longitude'] >= bounds['min_longitude']) & 
               (data['dropoff_longitude'] <= bounds['max_longitude']) &
               (data['dropoff_latitude'] >= bounds['min_latitude']) & 
               (data['dropoff_latitude'] <= bounds['max_latitude'])]
    
    return data
filtered_df2 = filter_out_of_bounds(filtered_df)

print(len(filtered_df2), len(set(filtered_df2['pickup_h3_index']).union(set(filtered_df2['dropoff_h3_index']))))
all_index = set(filtered_df2['pickup_h3_index']).union(set(filtered_df2['dropoff_h3_index']))
all_index = sorted(all_index)
map2idx = {}
cnt = 0
for item in all_index:
    map2idx[item] = cnt
    cnt += 1
# 生成request
import joblib
data = {}
for i in tqdm(range(len(filtered_df2))):
    record = filtered_df2.iloc[i]
    record = record.to_dict()
    days = record['tpep_pickup_datetime'].day
    hours = record['tpep_pickup_datetime'].hour
    minutes = record['tpep_pickup_datetime'].minute
    start_epoch = get_epoch(days, hours, minutes)
    
    days = record['tpep_dropoff_datetime'].day
    hours = record['tpep_dropoff_datetime'].hour
    minutes = record['tpep_dropoff_datetime'].minute
    end_epoch = get_epoch(days, hours, minutes)
    if start_epoch not in data.keys():
        data[start_epoch] = []
    record['PULocationID'] = copy.deepcopy(map2idx[record["pickup_h3_index"]])
    record['DOLocationID'] = copy.deepcopy(map2idx[record["dropoff_h3_index"]])
    record['pickup_epoch'] = start_epoch
    record['dropoff_epoch'] = end_epoch
    data[start_epoch].append(record)
    
with open("request_day1_2.json", "w") as f:
    json.dump(data, f,default=timestamp_to_json)