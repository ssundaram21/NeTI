import yaml
import os
import pickle
import json


SD1_4 = "CompVis/stable-diffusion-v1-4"
SD1_5 = "runwayml/stable-diffusion-v1-5"
SD2 = "stabilityai/stable-diffusion-2"

def train():
    train_root = "/data/scratch/shobhita/data/datasets/imagenet_sd_training"
    classes = os.listdir(train_root)
    assert len(classes) == 50
    
    with open("/data/vision/phillipi/perception/code/personalized-synthetic/imagenet_classes.json", "r") as f:
        info = json.load(f)

    root = "/data/vision/phillipi/perception/code/personalized-synthetic/personalize_model/NeTI/input_configs"
    with open(os.path.join(root, "train.yaml"), 'r') as file:
        t = yaml.safe_load(file)

    start_idx = 0
    for i, cls in enumerate(classes):
        new_t = t.copy()
    
        new_t['data']['train_data_dir'] = os.path.join(train_root, cls)
        new_t['data']['center_crop'] = True
        new_t['log']['exp_dir'] = f"/data/vision/phillipi/perception/code/personalized-synthetic/data/outputs/finetuned_models/neti/imagenet_50_cc"
        new_t['log']['exp_name'] = f"neti_stable-diffusion-v1-5_{cls}"
        new_t['model']["pretrained_model_name_or_path"] = SD1_5
        new_t['data']['super_category_token'] = info['class_to_superclass'][cls]
        new_t['data']['placeholder_token'] = "<new1>"
        new_t['optim']['max_train_steps'] = 500
        with open(os.path.join(root, f'train{i+start_idx}.yaml'), 'w') as file:
            yaml.dump(new_t, file)
    
#     start_idx = 50
#     for i, cls in enumerate(info['sd_train_paths'].keys()):
#         paths = info['sd_train_paths'][cls]
#         new_t = t.copy()
    
#         cat = paths[0].split("/")[-2]
    
#         new_t['data']['train_data_dir'] = "/".join(paths[0].split("/")[:-1])
#         new_t['log']['exp_dir'] = f"/data/vision/phillipi/perception/code/personalized-synthetic/data/outputs/finetuned_models/neti/imagenet_50"
#         new_t['log']['exp_name'] = f"neti_stable-diffusion-v1-4_{cls}"
#         new_t['model']["pretrained_model_name_or_path"] = SD1_4
#         new_t['data']['super_category_token'] = info['synset_to_superclass'][cls]
#         new_t['data']['placeholder_token'] = "<new1>"
#         new_t['optim']['max_train_steps'] = 500
#         with open(os.path.join(root, f'train{i+start_idx}.yaml'), 'w') as file:
#             yaml.dump(new_t, file)


def infer():
    start_idx=100
    root = "/data/vision/phillipi/perception/code/personalized-synthetic/personalize_model/NeTI/input_configs"
    with open(os.path.join(root, "inference.yaml"), 'r') as file:
        t = yaml.safe_load(file)

    train_root = "/data/scratch/shobhita/data/datasets/imagenet_sd_training"
    classes = os.listdir(train_root)
    assert len(classes) == 50

    for i, cls in enumerate(classes):
        new_t = t.copy()

        new_t['input_dir'] = f"/data/vision/phillipi/perception/code/personalized-synthetic/data/outputs/finetuned_models/neti/imagenet_50/neti_stable-diffusion-v1-5_{cls}"
        new_t["inference_dir"] = f"/data/vision/phillipi/perception/code/personalized-synthetic/data/outputs/generated_data/neti/imagenet_50/cfg_1.5_llm/{cls}"
        new_t["num_images_per_prompt"] = 10
        new_t["n"] = 1000
        new_t["gs"] = 1.5
        new_t["diverse"] = True
        new_t["cls"] = cls

        with open(os.path.join(root, f'inference{i+start_idx}.yaml'), 'w') as file:
            yaml.dump(new_t, file)

    start_idx=150
    root = "/data/vision/phillipi/perception/code/personalized-synthetic/personalize_model/NeTI/input_configs"
    with open(os.path.join(root, "inference.yaml"), 'r') as file:
        t = yaml.safe_load(file)

    for i, cls in enumerate(classes):
        new_t = t.copy()

        new_t['input_dir'] = f"/data/vision/phillipi/perception/code/personalized-synthetic/data/outputs/finetuned_models/neti/imagenet_50/neti_stable-diffusion-v1-5_{cls}"
        new_t["inference_dir"] = f"/data/vision/phillipi/perception/code/personalized-synthetic/data/outputs/generated_data/neti/imagenet_50/cfg_7.5_llm/{cls}"
        new_t["num_images_per_prompt"] = 10
        new_t["n"] = 1000
        new_t["gs"] = 7.5
        new_t["diverse"] = True
        new_t["cls"] = cls

        with open(os.path.join(root, f'inference{i+start_idx}.yaml'), 'w') as file:
            yaml.dump(new_t, file)
            
train()
