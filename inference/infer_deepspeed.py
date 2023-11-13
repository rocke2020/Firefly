import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0, help="")
args = parser.parse_args()

# 加载 Llama 2 7B 模型和分词器
tokenizer = AutoTokenizer.from_pretrained("/mnt/sda/yuzhao/code/llm/Firefly/checkpoints/Llama-2-13b-chat-hf-qlora-sft-merge")
model = AutoModelForCausalLM.from_pretrained("/mnt/sda/yuzhao/code/llm/Firefly/checkpoints/Llama-2-13b-chat-hf-qlora-sft-merge")


# 设置 DeepSpeed 配置参数
deepspeed_config = {
  "gradient_accumulation_steps": 3,
  "gradient_clipping": 0,
  "steps_per_print": 200,
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 32,
  "wall_clock_breakdown": False,

  "optimizer": {
        "type": "Adam",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
  "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": True,
    "contiguous_gradients": True,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": True
  },
  "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
}

# 使用 DeepSpeed 初始化模型和优化器
model_engine, optimizer, _, _  = deepspeed.initialize(model=model, optimizer=None, args=None, config_params=deepspeed_config)

# 定义一个函数来生成文本
def generate_text(prompt, max_length=50):
    # 将输入文本编码为 token ids
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # 将 token ids 复制到模型所在的设备上
    input_ids = input_ids.to(model_engine.device)
    # 使用 DeepSpeed 运行推理，并获取输出 token ids
    output_ids = model_engine.generate(input_ids, max_length=max_length)
    # 将输出 token ids 解码为文本
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# 测试一下生成文本的函数
prompt = "Hello, world!"
output_text = generate_text(prompt)
print(output_text)
