import json

# 原始文件路径
input_path = 'all_data.json'

# 读取原始 JSON 数据
with open(input_path, 'r') as f:
    data = json.load(f)

# 检查数据总量
total = len(data)
if total < 100:
    raise ValueError(f"数据量不足100条，仅有 {total} 条")

# 划分数据
data_60 = data[:60]
data_30 = data[60:90]
data_10 = data[90:100]

# 保存三个子文件
with open('data_part1_60.json', 'w') as f1:
    json.dump(data_60, f1, indent=2)

with open('data_part2_30.json', 'w') as f2:
    json.dump(data_30, f2, indent=2)

with open('data_part3_10.json', 'w') as f3:
    json.dump(data_10, f3, indent=2)

print("数据划分并保存成功：60条、30条、10条")
