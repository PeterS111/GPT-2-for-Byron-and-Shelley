import os
import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_steps", type=str, default="0")
parser.add_argument("--model_size", type=str, default="ABC")
parser.add_argument("--directory_path", type=str, default="ABC")

args = parser.parse_args()
model_steps = args.model_steps
m_size = args.model_size
dir_path = args.directory_path

base_path = dir_path + "/checkpoint/run1"
source = dir_path + "/models/" +  m_size + "/"

source_1 = source + "encoder.json"
source_2 = source + "hparams.json"
source_3 = source + "vocab.bpe"

path = dir_path + "/models/model_{}".format(model_steps)

os.mkdir(path)

for j in os.listdir(base_path):

    if ("model-" + model_steps + ".") in j:
        print(j)

        fb = base_path + "/" + j
        ft = path + "/" + j

        shutil.copy(fb, ft)
        print("fb: ", fb)
        print("ft: ", ft)

target_1 = path + "/" + "encoder.json"
target_2 = path + "/" + "hparams.json"
target_3 = path + "/" + "vocab.bpe"

shutil.copy(source_1, target_1)
shutil.copy(source_2, target_2)
shutil.copy(source_3, target_3)

checkpoint_path = path + "/" + "checkpoint"
f = open(checkpoint_path, "w", encoding="utf-8")
write_str=('model_checkpoint_path: "model-{}"\nall_model_checkpoint_paths: "model-{}"'.format(model_steps, model_steps))

f.write(write_str)
f.close()
