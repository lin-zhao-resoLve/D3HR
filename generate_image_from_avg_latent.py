import torch
import os
from dit_inversion import parse_args, sample1, invert
from dataset import LocalDataset
from diffusers import DiTPipeline, DDIMScheduler


def save_average_latents(input_dir, save_dir):
    latents_by_class = {}

    # Iterate over each class folder in input_dir
    for class_name in sorted(os.listdir(input_dir)):
        class_dir = os.path.join(input_dir, class_name)
        
        if os.path.isdir(class_dir):
            latents = []
            for file_name in sorted(os.listdir(class_dir)):
                if file_name.endswith('.pt'):
                    file_path = os.path.join(class_dir, file_name)
                    latent = torch.load(file_path)  # Load the latent .pth file
                    latents.append(latent)
            
            # Store the latent vectors for this class
            latents_tensor = torch.stack(latents)
            average_latent = torch.mean(latents_tensor, dim=0)

            # latents_by_class[class_name] = latents

            save_file_path = os.path.join(save_dir, f'{class_name}_average_latent.pth')
            torch.save(average_latent, save_file_path)
            print(f"Saved average latent for class '{class_name}' to {save_file_path}")
    return 

def load_mapping(mapping_file):
    wnid_to_index = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            wnid, index = line.strip().split('\t')
            wnid_to_index[wnid] = int(index)
    return wnid_to_index

def main():
    gpu = 1
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.set_device(gpu)
    # Load the ImageNet class names
    latents_path = '../dataset/tiny-imagenet-200/key50_far_latents_train_vit_image_feature_CLS'
    load_dir = '../dataset/tiny-imagenet-200/train_vae_encode'
    # os.makedirs(avg_latents_path, exist_ok=True)
    # save_average_latents(input_path, avg_latents_path)
    save_images_path = '../dataset/tiny-imagenet-200/key50_far_images_inversion_train_vit_image_feature_CLS'
    os.makedirs(save_images_path, exist_ok=True)
    pipe = DiTPipeline.from_pretrained("/home/user1/workspace/leilu/linzhao/dit_imagenet", torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) 
    pipe = pipe.to(device)
    wnid_to_index = load_mapping(os.path.join('../dataset/tiny-imagenet-200', 'tiny-imagenet-mapping.txt'))
    for class_name in os.listdir(latents_path):
        class_dir = os.path.join(latents_path, class_name)
        for file_name in sorted(os.listdir(class_dir)):
            path = os.path.join(load_dir, class_dir.split('/')[-1], file_name)
            # latent = torch.load(os.path.join(class_dir,file_name)).to(device)
            latent = torch.load(path).to(device)
            start_step = 20
            inverted_latent = invert(pipe,start_latents = latent.unsqueeze(0).to(torch.float16), class_labels=torch.tensor(wnid_to_index[class_dir.split('/')[-1]]).unsqueeze(0), num_inference_steps = 50, device=device)
            
            image = sample1(
                pipe,
                class_labels=torch.tensor(wnid_to_index[class_dir.split('/')[-1]]).unsqueeze(0),
                start_latents=inverted_latent,
                # start_latents = latent.unsqueeze(0).to(torch.float16),
                start_step=start_step,
                num_inference_steps=50,
                device=device
            )
            os.makedirs(os.path.join(save_images_path,class_dir.split('/')[-1]), exist_ok=True)
            # pipe.numpy_to_pil(image)[0].save(os.path.join(save_images_path,class_dir.split('/')[-1], file_name.split('_')[0]+'_image_'+file_name.split('_')[2]+'_'+file_name.split('_')[-1].split('.')[0]+'.png'))
            pipe.numpy_to_pil(image)[0].save(os.path.join(save_images_path,class_dir.split('/')[-1], file_name.split('.')[0]+'.png'))
            


if __name__ == "__main__":
    main()

