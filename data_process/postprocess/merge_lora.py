from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

"""
使用该脚本，将lora的权重合并大base model中
"""


def merge_lora_to_base_model(
        model_name_or_path,
        adapter_name_or_path,
        save_path,
    ):
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )    
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name_or_path,
    #     trust_remote_code=True,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    #     # device_map='auto',
    #     device_map={'': 'cpu'}
    # )
    # model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    # model = model.merge_and_unload()

    # tokenizer.save_pretrained(save_path)
    # model.save_pretrained(save_path)


def merge_multi_lora():
    """  """
    model_name_or_path='/mnt/nas1/models/llama/pretrained_weights/llama2-7b-chat-hf'
    quantized_root = Path('/mnt/nas1/models/llama/quantized_models')
    save_root = Path('/mnt/nas1/models/llama/merged_models')
    adapter_names = [
        'llama2-7b-ner-chem_gene-e3s6',
    ]
    for adapter_name in adapter_names:
        adapter_path = quantized_root / adapter_name / 'final'
        save_path = save_root / adapter_name
        print(Path(model_name_or_path).exists())
        merge_lora_to_base_model(model_name_or_path, adapter_path, save_path)


if __name__ == '__main__':
    merge_multi_lora()
