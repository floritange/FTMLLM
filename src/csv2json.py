import pandas as pd
import json

# 载入 CSV 文件
df = pd.read_csv('./coco-2024-dataset.csv')
data = []

# 添加对话数据
for i in range(len(df)):
    message_list = [
        {
            "content": f"<image>What is in this picture?",
            "role": "user"
        },
        {
            "content": df.iloc[i]['caption'],
            "role": "assistant"
        }
    ]
    image_list = [df.iloc[i]['image_path']]

    data_item = {
        "messages": message_list,
        "images": image_list
    }
    data.append(data_item)

# 保存为 Json
with open('data_vl.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    