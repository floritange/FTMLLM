#####========
# 推荐：modelscope找huggingface对应的模型和数据集。如：https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct
#####========
# from modelscope.hub.snapshot_download import snapshot_download

# model_dir = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", cache_dir="../models/llm_models")


#####========
#  huggingface镜像源方法
#####========
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# local_dir = "../models/llm_models"
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

# cmd = f"huggingface-cli download --resume-download {model_id} --local-dir {local_dir}/{model_id} --local-dir-use-symlinks False"
# os.system(cmd)

### huggingface方法

# from huggingface_hub import snapshot_download
# import os

# # 配置参数
# repo_id = "llamafactory/RLHF-V"
# local_dir = "./RLHF-V"  # 本地保存路径
# resume_download = True
# repo_type = "dataset"
# use_symlinks = False

# # 可选：如果需要认证，设置token（可通过环境变量或直接传入）
# # token = "hf_your_token_here"

# # 执行下载
# snapshot_download(
#     repo_id="llamafactory/RLHF-V",
#     repo_type="dataset",
#     local_dir="./RLHF-V",
#     resume_download=True,
#     local_dir_use_symlinks=use_symlinks,
#     token="xxx",
# )

        

from datasets import load_dataset
# 加载数据集
ds = load_dataset("UCSC-VLAA/MedTrinity-25M", "25M_demo", cache_dir="/data/huggingface/hub",token="xxxx")
