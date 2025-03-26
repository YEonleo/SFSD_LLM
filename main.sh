wandb login 20f894088a42a42e5eef02b48b1e6cce6805fdfe

#export CUDA_VISIBLE_DEVICES=0,1
wandb login 20f894088a42a42e5eef02b48b1e6cce6805fdfe


python main.py --layers o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj \
       --dataset cnn_dailymail --batch_size 32  \
       --seq_len 128 \
       --log_path surgical_logs.txt \
       --max_ratio 0.5 \
       --algo eigen \
       --weights_name decompose_llama2_cnndm \
       --model meta-llama/Llama-2-7b-hf \
       --cache_dir "../cache_dir" \
       --start_layer 28 \
       --delta 0.0 \
       --mode decompose \

python main.py --layers o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj \
       --dataset cnn_dailymail --batch_size 32  \
       --seq_len 128 \
       --log_path surgical_logs.txt \
       --max_ratio 0.5 \
       --algo eigen \
       --weights_name decompose_llama2_cnndm \
       --model meta-llama/Llama-2-7b-hf \
       --cache_dir "../cache_dir" \
       --start_layer 28 \
       --delta 0.0 \
       --mode prune \

python main.py --layers o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj \
       --dataset xsum --batch_size 32  \
       --seq_len 128 \
       --log_path surgical_logs.txt \
       --max_ratio 0.5 \
       --algo eigen \
       --weights_name decompose_llama2_xsum \
       --model meta-llama/Llama-2-7b-hf \
       --cache_dir "../cache_dir" \
       --start_layer 28 \
       --delta 0.0 \
       --mode decompose \

python main.py --layers o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj \
       --dataset cnn_dailymail --batch_size 32  \
       --seq_len 128 \
       --log_path surgical_logs.txt \
       --max_ratio 0.5 \
       --algo eigen \
       --weights_name decompose_llama2_cnndm \
       --model meta-llama/Llama-2-7b-hf \
       --cache_dir "../cache_dir" \
       --start_layer 28 \
       --delta 0.0 \
       --mode prune \