# 多模态大模型微调
# 这是公共开发环境，删除文件时，请确认清除别删错了
- 路径：/home/tangou/FTMLLM
- python环境：conda activate mllm

- Qwen2.5VL
- llamafactory: https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md
- git版本：https://github.com/hiyouga/LLaMA-Factory/tree/142fd7e7558fa9c169ce3d07a3fdff3eafa7b8d7
- ubuntu:22.04
- cuda:12.4.1
- python:3.10.16

![1743475622277](images/1743475622277.png)
```bash
cd /home/tangou/tangou2
conda activate tg10

# 训练
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_3.yaml

export GRADIO_SERVER_PORT=7860

# 推理
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli webchat qwen2.5vl_lora_sft_3_inference.yaml

# 评估
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_3_evaluation.yaml
```



```bash
source /data/tools/setproxy.sh
git clone https://github.com/floritange/FTMLLM.git
conda create -n mllm python=3.10.16
conda activate mllm
cd /home/tangou/tangou1/FTMLLM
pip install -r requirements_llamafactory.txt

# 评估：https://blog.csdn.net/H66778899/article/details/140525340

# 下载数据集：https://huggingface.co/datasets/openbmb/RLHF-V-Dataset/tree/main
# 精简版：https://huggingface.co/datasets/llamafactory/RLHF-V/tree/main
# openbmb/RLHF-V-Dataset
# llamafactory/RLHF-V
# 教程：https://developer.aliyun.com/article/1643200
# 数据集：https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M

# 教程最新: https://github.com/datawhalechina/self-llm/blob/master/models/Qwen2-VL/04-Qwen2-VL-2B%20Lora%20%E5%BE%AE%E8%B0%83.md
# 🔥https://baaidata.csdn.net/67bd31f13b685529b7ffd73c.html
# 🔥https://blog.csdn.net/2301_80247435/article/details/143678295

# 系列教程：https://zhuanlan.zhihu.com/p/26993872051
# https://blog.csdn.net/2301_80247435/article/details/143678295

=== python
from datasets import load_dataset
# 加载数据集 huggingface token
ds = load_dataset("UCSC-VLAA/MedTrinity-25M", "25M_demo", cache_dir="/data/huggingface/hub",token="xxxx")
===




huggingface-cli login
# token: xxxx
# huggingface-cli download --resume-download <xxx> --local-dir-use-symlinks False
export HUGGINGFACE_HUB_CACHE="/data/huggingface/hub"
huggingface-cli download --resume-download --repo-type dataset llamafactory/RLHF-V --local-dir-use-symlinks False

# 教程：https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md
# https://github.com/hiyouga/LLaMA-Factory/tree/main/examples
# model位置：/data/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/6bcf1c9155874e6961bcf82792681b4f4421d2f7
# /data/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/319ccfdc6cd974fab8373cb598dfe77ad93dedd3
# rlhf位置：/data/huggingface/hub/datasets--openbmb--RLHF-V-Dataset/snapshots/1d8e9804b59e9da64ad7b1e17d505869ab9b2ad3
# rlhf精简版：/data/huggingface/hub/datasets--llamafactory--RLHF-V/snapshots/7e91ceac7cd540381751d434c8ab40ea1138ef9f
# 1.参考配置dataset_info.json文件：https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md
# 2.配置qwen2.5vl_lora_dpo.yaml文件
export HUGGINGFACE_HUB_CACHE="/data/huggingface/hub"
export NCCL_P2P_LEVEL=NVL
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train /home/tangou/tangou1/FTMLLM/configs/qwen2.5vl_lora_sft_3.yaml
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli chat /home/tangou/tangou1/FTMLLM/configs/qwen2.5vl_lora_sft_3_inference.yaml
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli webchat /home/tangou/tangou1/FTMLLM/configs/qwen2.5vl_lora_sft_3_inference.yaml
kill -9 1966685 1968824 1968825 1968826 1968827 1968839 1968840 1968841


####### 教程
source /data/tools/setproxy.sh
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 142fd7e7558fa9c169ce3d07a3fdff3eafa7b8d7
conda create -n mllm python=3.10.16
conda activate mllm
# 新建个requirements_llamafactory.txt，配置在上面
pip install -r requirements_llamafactory.txt
wget https://github.com/floritange/FTMLLM/blob/main/EETQ-1.0.1-cp310-cp310-linux_x86_64.whl
pip install EETQ-1.0.1-cp310-cp310-linux_x86_64.whl


# 教程：https://github.com/hiyouga/LLaMA-Factory/tree/main/examples
# 参考sft脚本，如：
# llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml
# llamafactory-cli webchat examples/inference/llama3_lora_sft.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HUGGINGFACE_HUB_CACHE="/data/huggingface/hub"
# 训练
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train /home/tangou/tangou1/FTMLLM/configs/qwen2.5vl_lora_sft_3.yaml
# 推理
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli webchat /home/tangou/tangou1/FTMLLM/configs/qwen2.5vl_lora_sft_3_inference.yaml

#####  两个yaml配置如下：

# qwen2.5vl_lora_sft_3.yaml
===
### model，模型在共享目录下
model_name_or_path: /data/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/319ccfdc6cd974fab8373cb598dfe77ad93dedd3
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: mllm_demo
dataset_dir: /home/tangou/tangou1/FTMLLM/LLaMA-Factory/data  # dataset_info.json所在文件夹
template: qwen2_vl
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: /home/tangou/tangou1/1results/qwen2.5_vl/lora/sft_dev
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500

### SwanLab 配置 (这里改为自己的信息)
use_swanlab: true
swanlab_project: llamafactory
swanlab_mode: cloud
swanlab_api_key: iiuay9Y1iDp0wrrYDX4cW

# deepspeed
deepspeed: /home/tangou/tangou1/cache/ds_z3_config.json
===

# qwen2.5vl_lora_sft_3_inference.yaml
===
model_name_or_path: /data/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/319ccfdc6cd974fab8373cb598dfe77ad93dedd3
adapter_name_or_path: /home/tangou/tangou1/1results/qwen2.5_vl/lora/sft_dev
template: qwen2_vl
infer_backend: vllm  # choices: [huggingface, vllm]
trust_remote_code: true
===

# ds_z3_config.json
===
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1000000000.0,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1000000000.0,
    "stage3_max_reuse_distance": 1000000000.0,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
===


  "rlhf_v": {
    "hf_hub_url": "llamafactory/RLHF-V",
    "ranking": true,
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected",
      "images": "images"
    }
  },

  "mllm_demo": {
    "file_name": "mllm_demo.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },

```
```json
,
  "rlhf_v_custom": {
    "hf_hub_url": "llamafactory/RLHF-V",
    "ranking": true,
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected",
      "images": "images"
    }
  },
  "mllm_demo_custom": {
    "file_name": "mllm_demo.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    }
  }
```




数据格式
```json
[
  {
   
    "messages": [
      {
   
        "content": "<image>他们是谁？",
        "role": "user"
      },
      {
   
        "content": "他们是拜仁慕尼黑的凯恩和格雷茨卡。",
        "role": "assistant"
      },
      {
   
        "content": "他们在做什么？",
        "role": "user"
      },
      {
   
        "content": "他们在足球场上庆祝。",
        "role": "assistant"
      }
    ],
    "images": [
      "mllm_demo_data/1.jpg"
    ]
  }
]
```

```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/319ccfdc6cd974fab8373cb598dfe77ad93dedd3 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir data \
    --dataset mllm,mllm_demo_custom \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2.5-VL-3B-Instruct/lora/train_2025-03-31-15-32-55 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --deepspeed cache/ds_z3_config.json
    --use_swanlab True \
    --swanlab_project llamafactory \
    --swanlab_api_key iiuay9Y1iDp0wrrYDX4cW \
    --swanlab_mode cloud
```

evaluation
```bash
llamafactory-cli train \
    --stage sft \
    --model_name_or_path /data/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/319ccfdc6cd974fab8373cb598dfe77ad93dedd3 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bitsandbytes \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset alpaca_en_demo \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate True \
    --max_new_tokens 512 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/Qwen2.5-VL-3B-Instruct/lora/eval_2025-04-21-00-59-20 \
    --trust_remote_code True \
    --do_predict True 
```



cp -r /home/tangou/tangou1/FTMLLM/share/* /data/3e/share/

```bash
.
├── models
│   ├── ft_models		#微调后的模型
│   └── llm_models	    #微调前基座llm
├── README.md
├── requirements.txt
└── src					#所有代码文件位置
    └── down_llm.py
```


### 微调：https://github.com/hiyouga/LLaMA-Factory/tree/main/examples
- Swanlab：https://swanlab.cn/login
- 13056500789
- tg123456
- api_key: iiuay9Y1iDp0wrrYDX4cW


```bash
export HF_ENDPOINT=https://hf-mirror.com
export USE_MODELSCOPE_HUB=1
pip install EETQ-1.0.1-cp310-cp310-linux_x86_64.whl




# 推荐可视化界面
cd /home/tangou/FTMLLM/LLaMA-Factory
conda activate mllm
llamafactory-cli webui
### ui配置见下图
运行后的配置缓存路径：/home/tangou/FTMLLM/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/lora/train_2025-03-12-17-06-35



# ========
## data配置路径：/home/tangou/FTMLLM/LLaMA-Factory/data。只需要把数据集放入配置data_info。https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md
## 参照
/home/tangou/FTMLLM/LLaMA-Factory/data/llamafactory/RLHF-V # huggingface上面下载
/home/tangou/FTMLLM/LLaMA-Factory/data/data_info.json
  "rlhf_v": {
    "hf_hub_url": "llamafactory/RLHF-V",
    "ranking": true,
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected",
      "images": "images"
    }


# 终端
# llamafactory-cli train /home/tangou/FTMLLM/configs/qwen2.5vl_lora_dpo.yaml
# llamafactory-cli chat /home/tangou/FTMLLM/configs/qwen2.5vl_lora_dpo.yaml
# llamafactory-cli export /home/tangou/FTMLLM/configs/qwen2.5vl_lora_dpo.yaml
```


可视化
llamafactory
![alt text](images/1.png)

swanlab
![alt text](images/2.png)

后台
![alt text](images/3.png)


webchat
![alt text](./images/4.png)

![alt text](./images/5.png)

![alt text](./images/6.png)



# 教程详细版
视频版地址：
链接: https://pan.baidu.com/s/1GTNBCrZ5hBw4w2CYj2jMeA?pwd=73c8 提取码: 73c8

vscode

vscode插件安装：chinese、remote、python、pylance、python debugger、Python Environment Manager

![alt text](image-12.png)

![alt text](image-13.png)



### 服务器连接
大家查看群文件自己的user和密码

vscode连接:

![alt text](./images/7.png)

1. 复制到文件里面去
```yaml
Host 36.212.4.98
  HostName 36.212.4.98
  User tangou
```

![alt text](./images/8.png)

2. 点刷新，再打开文件

![alt text](image.png)

3. 输入密码123456，回车

![alt text](image-1.png)

4. 进入

![alt text](image-2.png)

![alt text](image-7.png)

5. 回到第2步，再去开一个目录。这是共享模型数据的目录

![alt text](image-4.png)

![alt text](image-5.png)


### 微调预备
- 环境变量：conda、ollama
1. 打开终端

![alt text](image-14.png)

2. 命令行运行，复制命令过去回车，运行。下面截图我之前运行过了，没运行。

```bash
cat /data/tools/setenv.sh >> ~/.bashrc
source ~/.bashrc
```
![alt text](image-15.png)

![alt text](image-16.png)

检查是否运行成功
```bash
conda info --envs #查看conda环境
ollama list # 查看ollama有哪些模型
ollama run bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh # 运行ollama交互式，ctrl d 取消
```

![alt text](image-17.png)


- vpn：先不用管，知道这个就行
```bash
# http://127.0.0.1:18099
source /data/tools/setproxy.sh  #启动vpn
source /data/tools/unsetproxy.sh  #关闭vpn
```

- python环境

1. 打开终端，进入文件夹
```bash
cd /home/tangou/tangou2 #你自己的路径
```

![alt text](image-18.png)

2. copy文件到目录

```bash
cp -r /data/3e/share/* /home/tangou/tangou2/
```

copy之后

![alt text](image-19.png)

3. 创建环境并安装包

```bash
# 开vpn
source /data/tools/setproxy.sh
# tg10 换成自己的名字
conda create -n tg10 python=3.10.16
# 切换环境
conda activate tg10
```

![alt text](image-20.png)

4. pip安装包

```bash
# 如果重新打开终端，没启动。请启动下，开vpn。
source /data/tools/setproxy.sh
# 安装包，第一次跑没缓存，运行时间会很久，在下数据包
pip install -r requirements.txt
# 额外安装这个包，pip源没有
pip install EETQ-1.0.1-cp310-cp310-linux_x86_64.whl
```

![alt text](image-21.png)

![alt text](image-23.png)

### 微调，这里用llamafactory提供的数据

1. 数据解读

![alt text](image-24.png)

2. 微调加载数据，首先将自定义数据配置到dataset_info.json

![alt text](image-25.png)

3. 配置模型路径

首先回到连接data共享目录，连vscode

![alt text](image-26.png)

复制路径，我们这里微调32B

![alt text](image-28.png)

3. 配置微调的配置文件qwen2.5vl_lora_sft_3.yaml

回到原来的vscode，将上面复制的model路径放进来

![alt text](image-30.png)

4. 运行微调

打开终端

![alt text](image-31.png)

```bash
# 如果重新打开终端，没启动。请启动下，开vpn。
source /data/tools/setproxy.sh
# 切换你的python环境
conda activate tg10
# 训练
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_3.yaml



# 查看运行记录，swanlog是相对路径，如果端口被占用，则--port xxx
conda activate tg10
swanlab watch swanlog --port 5092
```

![alt text](image-32.png)

![alt text](image-34.png)

![alt text](image-35.png)

![alt text](image-36.png)

本地浏览器访问：http://127.0.0.1:5092

![alt text](image-37.png)


- 实测：32B
- 训练：在per_device_train_batch_size=1的情况下，32B显存空余如下，如果调72B需要乘2，72B勉强够用。根据显存空余，可调大per_device_train_batch_size=2, 4, 6, 8不等。
- 训练：6组数据（每组2-3轮对话），一个epoch需要：50s-80s。
- 推理：差点爆显存
- 评估：直接爆显存

![alt text](image-38.png)

![alt text](image-50.png)

5. 推理
```bash
source /data/tools/setproxy.sh
conda activate tg10
# 如果端口占用，请换个端口
export GRADIO_SERVER_PORT=7860
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli webchat qwen2.5vl_lora_sft_3_inference.yaml
```

![alt text](image-41.png)

![alt text](image-42.png)

本地浏览器访问：http://0.0.0.0:7860

![alt text](image-43.png)

拿刚刚训练的数据来测

![alt text](image-44.png)

下载到本地

![alt text](image-45.png)

推理（这个图片在原本的模型上就一个训练过）

![alt text](image-46.png)


6. 评估（实测：32B评估时会爆显存）

```bash
source /data/tools/setproxy.sh
conda activate tg10
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_3_evaluation.yaml
```

![alt text](image-47.png)


### 额外
webui运行

```bash
source /data/tools/setproxy.sh
conda activate tg10
# 如果端口占用请换个端口
export GRADIO_SERVER_PORT=7860
llamafactory-cli webui
```

![alt text](image-48.png)

下载模型、数据集

```bash
source /data/tools/setproxy.sh
conda activate tg10
huggingface-cli login # token教程：https://blog.csdn.net/m0_52625549/article/details/134255660
----
export HUGGINGFACE_HUB_CACHE="/data/huggingface/hub"  #设置缓存路径，就是之前的共享目录
# 数据集
huggingface-cli download --resume-download --repo-type dataset llamafactory/RLHF-V --local-dir-use-symlinks False
# 模型
huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir-use-symlinks False
```

![alt text](image-49.png)

### export/merge model，合并然后评估（显存爆，和上面评估情况一样）
```bash
llamafactory-cli export qwen2.5vl_lora_sft_3_export.yaml
```
![1745168057142](image/README_tg/1745168057142.png)

导出完成

![1745168755998](image/README_tg/1745168755998.png)

```bash
ollama create qwen2_5_vl_custom -f /home/tangou/tangou2/results/export/qwen2.5_vl_lora_sft_dev/Modelfile
ollama run qwen2_5_vl_custom
```

评估

```bash
source /data/tools/setproxy.sh
conda activate tg10
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_3_evaluation_no_lora.yaml
```

爆显存

![1745169235780](image/README_tg/1745169235780.png)


### 量化评估
32B模型微调后，8bit量化爆显存，8卡每张卡占40G左右，4bit量化每张卡占20G左右，如下结果

修改qwen2.5vl_lora_sft_3_evaluation.yaml的量化，bit，为4和8，如果不需要就把其全部注释
```bash
source /data/tools/setproxy.sh
conda activate tg10
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_3_evaluation.yaml
```
![1745169986012](image/README_tg/1745169986012.png)

![1745169707648](image/README_tg/1745169707648.png)

4bit评估结果
![1745170121915](image/README_tg/1745170121915.png)



### 测评，32B-4bit量化，7B全量。
数据集coco_2014_caption，部分

60，30，10

10 epoch

![1745171465045](image/README_tg/1745171465045.png)

#### 7B效果
```bash
conda activate tg10
# 7B，微调，per_device_train_batch_size: 8, 8卡每张显存占用 28-35G, 1 epoch/30s
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_7B_train.yaml
# 7B微调前，评估，per_device_train_batch_size: 4, 8卡每张显存占用 26G
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_7B_evaluation_before.yaml
# 7B微调后，评估，per_device_train_batch_size: 4, 8卡每张显存占用 26G
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_7B_evaluation_after.yaml
```

7B微调loss（收敛不明显）

![1745172260168](image/README_tg/1745172260168.png)

结果
```json
// 7B微调前
{
    "predict_bleu-4": 8.043468749999999,
    "predict_model_preparation_time": 0.0075,
    "predict_rouge-1": 18.08796875,
    "predict_rouge-2": 4.3805499999999995,
    "predict_rouge-l": 11.1585375,
    "predict_runtime": 8.4024,
    "predict_samples_per_second": 3.57,
    "predict_steps_per_second": 0.119
}
// 7B微调后
{
    "predict_bleu-4": 8.093843750000001,
    "predict_model_preparation_time": 0.0101,
    "predict_rouge-1": 18.75373125,
    "predict_rouge-2": 4.851125,
    "predict_rouge-l": 11.525771875,
    "predict_runtime": 12.6196,
    "predict_samples_per_second": 2.377,
    "predict_steps_per_second": 0.079
}
```

#### 32B效果

```bash
conda activate tg10
# 32B，微调，per_device_train_batch_size: 6, 8卡每张显存占用 40G, 1 epoch/120s
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_32B_train.yaml

# 32B微调前，评估，8bit量化，per_device_train_batch_size: 2, 8卡每张显存占用 38G
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_32B_evaluation_before_8bit.yaml
# 32B微调前，评估，4bit量化，per_device_train_batch_size: 2, 8卡每张显存占用 24G
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_32B_evaluation_before_4bit.yaml

# 32B微调后，评估，8bit量化，爆显存
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_32B_evaluation_after_8bit.yaml
# 32B微调后，评估，4bit量化，per_device_train_batch_size: 2, 8卡每张显存占用 24G
NCCL_P2P_LEVEL=NVL HUGGINGFACE_HUB_CACHE="/data/huggingface/hub" FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train qwen2.5vl_lora_sft_32B_evaluation_after_4bit.yaml
```

32B微调loss（收敛不明显）

![1745174712804](image/README_tg/1745174712804.png)

结果
```json
// 32B微调前，8bit
{
    "predict_bleu-4": 4.050425000000001,
    "predict_model_preparation_time": 0.0026,
    "predict_rouge-1": 11.830003125,
    "predict_rouge-2": 2.671021875,
    "predict_rouge-l": 5.869315625,
    "predict_runtime": 236.453,
    "predict_samples_per_second": 0.127,
    "predict_steps_per_second": 0.008
}
// 32B微调前，4bit
{
    "predict_bleu-4": 3.9843874999999995,
    "predict_model_preparation_time": 0.0026,
    "predict_rouge-1": 11.7092875,
    "predict_rouge-2": 2.0235125,
    "predict_rouge-l": 5.531334375,
    "predict_runtime": 193.3049,
    "predict_samples_per_second": 0.155,
    "predict_steps_per_second": 0.01
}

// 32B微调后，4bit
{
    "predict_bleu-4": 4.27783125,
    "predict_model_preparation_time": 0.0129,
    "predict_rouge-1": 12.3642375,
    "predict_rouge-2": 2.23608125,
    "predict_rouge-l": 6.093603125,
    "predict_runtime": 220.0491,
    "predict_samples_per_second": 0.136,
    "predict_steps_per_second": 0.009
}
```