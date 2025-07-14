import json
import os

def load_mapping(mapping_file):
    new_mapping = {}
    with open(mapping_file, 'r') as file:
        data = json.load(file)
        if "tiny" in mapping_file:
            for index, line in enumerate(file):
                key = line.split()[0]
                new_mapping[key] = index 
        else:
            new_mapping = {item["wnid"]: item["index"] for item in data.values()}
    return new_mapping


wnid_to_index = load_mapping("ds_inf/imagenet_1k_mapping.json")
class_dirs = sorted(list(wnid_to_index.keys()))
path_list = []
for class_dir in class_dirs:
    for i in range(20):
        path_list.append(os.path.join('/scratch/zhao.lin1/imagenet1k_256_4.0classfree_start_step_18_ddim_inversion_20_min_images_2/', class_dir, str(i)+'.png'))
output_file = "/scratch/zhao.lin1/imagenet1k_256_4.0classfree_start_step_18_ddim_inversion_20_min_images_2/train.txt"
with open(output_file, "w") as file:
    for path in path_list:
        file.write(path + "\n") 