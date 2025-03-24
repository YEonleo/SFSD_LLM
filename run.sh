python decomposer.py --layers o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj \
       --dataset boolq --batch_size 64 \
       --seq_len 128 \
       --log_path surgical_logs.txt \
       --algo eigen \
       --weights_name decompose_llama2_cnndm.pt \
       --model meta-llama/Llama-2-7b-hf \
       --cache_dir "../SpeculativeDecoding/cache_dir" \

python3 surgical.py --layers o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj \
       --dataset boolq \
       --log_path surgical_logs.txt \
       --start_layer 28 \
       --base_model decompose_llama2_cnndm.pt \
       --model meta-llama/Llama-2-7b-hf \
       --delta 0.0 \
       