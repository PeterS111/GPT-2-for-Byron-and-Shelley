import os
import shutil

seed_start = 0
seed_end = 3
ckpt = 100
steps_increment = 100
sample_length = 1000

## model_descr can be anything, it is not used to build directory paths, 
## it is only used to name the generated files

# model_name = "B5_XL"
# model_size = "1558M"
# directory_path = "/cluster/home/cug/ps564/"

model_descr = "S5_SM"
model_size = "124M"
directory_path = "C:/Users/MASTER/GPT_2_XL_TEST/"
input_name = "input_data/S5.txt"

for i in range(1,11):


    
    os.system('python train.py --dataset {} --model_name {} --top_k 50 --top_p 1.0 --max_steps {} --save_every {} --sample_every 100000 --max_to_keep 10  --val_batch_count 40'.format(input_name, model_size, ckpt, steps_increment))

    os.system('python mover.py --model_steps {} --model_size {} --directory_path {}'.format(ckpt, model_size, directory_path))       
        
    for s in range(seed_start, seed_end):
        os.system('python generate_conditional_samples_to_file.py --model_descr {} --model_name model_{} --length {} --seed {} --temperature 1.0 --top_k 50 --top_p 1.0 --nsamples 1 --raw_text "The eternal sky "'.format(model_descr, ckpt, sample_length, s))

    shutil.rmtree('models/model_{}'.format(str(ckpt)))
    
    c = "model-" + str(ckpt - steps_increment)
    
    for z in os.listdir("checkpoint/run1"):
        if c in z:
            d  = "checkpoint/run1/" + z
            os.remove(d)
        

    ckpt += steps_increment
