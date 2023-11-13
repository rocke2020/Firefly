from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
import os
import deepspeed
from pathlib import Path


model_name_or_path = "/mnt/nas1/models/llama/merged_models/llama2-7b-ner-chem_gene-e3s6"
device_map = "auto"
# if we are in a distributed setting, we need to set the device map and max memory per device
if os.environ.get('LOCAL_RANK') is not None:
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        ),
    )

def chat_ner(x):
    input_pattern = '<s>{}</s>'
    text = x.strip()
    text = input_pattern.format(text)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=50, do_sample=False,
            top_p=1, temperature=0.01, repetition_penalty=1,
            eos_token_id=tokenizer.eos_token_id
        )
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs)
    response = response.strip().replace(text, "").replace('</s>', "").replace('<s>', "").strip()
    return response

file = Path('/mnt/nas1/corpus-bio-nlp/NER/PGx_CTD_chem_x_gene.csv')
df_pgx_ctd = pd.read_csv(file, dtype=str)
question = (
    "{sentence}"
    "\n---------------\n"
    "please extract all Chemical and Gene in the above text, "
    "Gene includes gene or protein, excluding Limited variation, Genomic variation, Genomic factor, Haplotype. "
    "Chemical includes chemical, drug and amino acid, excluding disease."
    "The output format should be '<starting index in sentence, ending index in sentence, entity name, entity type>' .")
df_pgx_ctd = df_pgx_ctd.drop_duplicates(subset=["sentence"])
df_pgx_ctd["prompt"] = df_pgx_ctd["sentence"].apply(lambda x: question.format(sentence=x))

input1 = df_pgx_ctd["prompt"][0]
print(input1)
r = chat_ner(input1)
print(r)


# NF4
# from tqdm import tqdm
# dict_sent = {}
# for i, (sentnece, prompt) in tqdm(enumerate(zip(df_pgx_ctd["sentence"], df_pgx_ctd["prompt"])), desc="pred", total=len(df_pgx_ctd)):
#     response = chat_ner(prompt)
#     dict_sent[sentnece] = response
# df = pd.DataFrame.from_dict(dict_sent, orient="index").reset_index().rename(columns={"index": "sentence", 0:"ner"})
# df_cga_res = df_pgx_ctd.merge(df)
# df_cga_res.to_json("/mnt/sda/yuzhao/code/llm/Firefly/data/pgx_ctd_res_nf4.json", orient="records")
