import os 

os.system('python train.py --dataset input_data/S5.txt --model_name 124M --top_k 50 --top_p 1.0 --save_every 10000 --sample_every 1000000 --max_to_keep 60 --val_batch_count 40')
