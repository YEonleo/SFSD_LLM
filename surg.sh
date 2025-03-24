python3 surgical.py --layers o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj \
       --dataset cnn_dailymail \
       --log_path surgical_logs.txt \
       --start_layer 28 \
       --base_model decompose_llama2_cnndm.pt \
       --model meta-llama/Llama-2-7b-hf \
       --delta 0.0 \
       