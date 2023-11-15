from peft import AutoPeftModelForCausalLM

# No need base model?
path_to_adapter = '/mnt/nas1/models/llama/quantized_models/llama2-7b-ner-chem_gene-e3s10/final'
model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()