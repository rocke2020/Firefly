# 
file=data_process/postprocess/merge_lora.py
file=data_process/preprocess/ner/create_sft_data.py
nohup python $file \
    > $file.log 2>&1 &