import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler, DiTPipeline
import argparse
import os
from scipy import io
from diffusers import DiTPipeline
from diffusers.utils.torch_utils import randn_tensor
from concurrent.futures import ThreadPoolExecutor, as_completed
import ipdb
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from itertools import islice
import json

# sample
@torch.no_grad()
def sample(
    pipe,
    class_labels,
    start_step=0,
    start_latents=None,
    guidance_scale=4.0,
    num_inference_steps=30,
    do_classifier_free_guidance=True,
    device=None,
):
    
    batch_size = len(class_labels)
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels
    if start_latents == None:
        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=pipe._execution_device,
            dtype=pipe.transformer.dtype,
        )
    else:
        latents = start_latents.clone()
    
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

    class_labels = torch.tensor(class_labels, device=device).reshape(-1)
    class_null = torch.tensor([1000] * batch_size, device=device)
    class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels
    class_labels_input = class_labels_input.to(device)

    # set step values
    pipe.scheduler.set_timesteps(num_inference_steps)
    
    
    for i in tqdm(range(start_step, num_inference_steps)):
    
        t = pipe.scheduler.timesteps[i]
        
        if do_classifier_free_guidance:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)
            
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        timesteps = t
        if not torch.is_tensor(timesteps):
            is_mps = latent_model_input.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latent_model_input.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(latent_model_input.shape[0])
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        # predict noise model_output
        noise_pred = pipe.transformer(
            latent_model_input, timestep=timesteps, class_labels=class_labels_input
        ).sample

         # Perform guidance
        if do_classifier_free_guidance:
            # perform guidance
            # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if pipe.transformer.config.out_channels // 2 == latent_channels:
        
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # calculate ddim:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latent_model_input - (1 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * model_output
        latent_model_input = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
        # latent_model_input = pipe.scheduler.step(model_output, t, latent_model_input).prev_sample

    if guidance_scale > 1:
        latents, _ = latent_model_input.chunk(2, dim=0)
    else:
        latents = latent_model_input
            
    latents = 1 / pipe.vae.config.scaling_factor * latents
    samples = pipe.vae.decode(latents).sample

    samples = (samples / 2 + 0.5).clamp(0, 1)

    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

    return samples

    




## Inversion
@torch.no_grad()
def invert(
    pipe,
    start_latents,
    class_labels,
    guidance_scale=4.0,
    num_inference_steps=80,
    do_classifier_free_guidance=True,
    device=None,
):

    batch_size = len(class_labels)
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels
    
    latents = start_latents.clone()
    
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

    class_labels = torch.tensor(class_labels, device=device).reshape(-1)
    class_null = torch.tensor([1000] * batch_size, device=device)
    class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels
    class_labels_input = class_labels_input.to(device)

    # set step values
    pipe.scheduler.set_timesteps(num_inference_steps)
    intermediate_latents = []
    
    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps_all = reversed(pipe.scheduler.timesteps)
    
    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):
        
        # We'll skip the final iteration
        if i >= num_inference_steps - 1 -10:
            continue

        t = timesteps_all[i]    
        
        if do_classifier_free_guidance:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)
            
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        timesteps = t
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latent_model_input.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latent_model_input.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(latent_model_input.shape[0])
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        # predict noise model_output
        noise_pred = pipe.transformer(
            latent_model_input, timestep=timesteps, class_labels=class_labels_input
        ).sample

         # Perform guidance
        if do_classifier_free_guidance:
            # perform guidance
            # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if pipe.transformer.config.out_channels // 2 == latent_channels:
        
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latent_model_input = (latent_model_input - (1 - alpha_t).sqrt() * model_output) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * model_output

        if guidance_scale > 1:
            latents_out, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents_out = latent_model_input

        # Store i=[3, 8, 13, 18, 23, 28, 33, 38, 43, 48]
        # if (i+2)%5 == 0:
        if i>23 and i<39:
            intermediate_latents.append(latents_out)
    return torch.stack(intermediate_latents, dim=0)

    # return torch.cat(intermediate_latents)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/scratch/zhao.lin1/ddim_inversion_statistic",
        help="statistic save path",
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        default="ds_inf/imagenet_1k_mapping.json",
    )
    parser.add_argument("--txt_file", default='ds_inf/imagenet1k_train.txt', type=str)
    parser.add_argument("--pretrained_path", default='/scratch/zhao.lin1/DiT-XL-2-256', type=str)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    
    return args

def view_latents(pipe = None, inverted_latents = None):
    with torch.no_grad():
        im = pipe.decode_latents(inverted_latents[-1].unsqueeze(0))
    pipe.numpy_to_pil(im)[0]
    
    
def collate_fn(batch):
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    images = torch.stack([item['images'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    idx = torch.tensor([item['idx'] for item in batch])
    paths = [item['paths'] for item in batch]

    return {
        'images': images,
        'labels': labels,
        'idx':idx,
        'paths': paths
    }


def save_latent(latent, save_path):
    torch.save(latent, save_path)



class ImageNetDataset(Dataset):
    def __init__(self, txt_file='', mapping_file=None, class_dir=None):
        self.images = []
        self.img_labels = []
        self.class_dir = class_dir
        self.transform = self.get_transforms()
        
        # Load class mapping and json file
        self.wnid_to_index = load_mapping(mapping_file)
        self._load_from_txt(txt_file)
        

    def _load_from_txt(self, txt_file):
        with open(txt_file, "r") as file:
            image_paths = file.readlines()
        image_paths = [path.strip() for path in image_paths if path.split('/')[-2]==self.class_dir]
        for path in image_paths:
            self.images.append(path)
            class_index = self.wnid_to_index[path.split('/')[-2]]
            self.img_labels.append(class_index)


    def get_transforms(self):
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])
        
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            # normalize
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in case of error
            image = Image.new('RGB', (256, 256))
            
        img_label = self.img_labels[idx]
        
        if self.transform:
            image = self.transform(image)

        sample = {
            'images': image,
            'paths': img_path,
            'labels': img_label,
            'idx': idx
        }

        return sample


def load_mapping(mapping_file):
    new_mapping = {}
    with open(mapping_file, 'r') as file:
        data = json.load(file)
        if "tiny" in mapping_file:
            for index, line in enumerate(file):
                # Extract wnid (eg. n01443537) for each line and -1
                key = line.split()[0]
                new_mapping[key] = index 
        else:
            new_mapping = {item["wnid"]: item["index"] for item in data.values()}
    return new_mapping

def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.set_device(args.gpu)
    
    wnid_to_index = load_mapping(args.mapping_file)
    class_dirs = sorted(list(wnid_to_index.keys()))[args.start:args.end]
 
    pipe = DiTPipeline.from_pretrained(args.pretrained_path, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) 

    os.makedirs(args.save_dir, exist_ok=True)
    

    pipe = pipe.to(device)

    for class_dir in tqdm(class_dirs):
        imgnet1k_dataset = ImageNetDataset(args.txt_file, args.mapping_file, class_dir)
        trainloader = torch.utils.data.DataLoader(imgnet1k_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers, drop_last=False, collate_fn=collate_fn)
        latents = []
        for sample in tqdm(trainloader):
            with torch.no_grad():
                images = sample['images'].to(device)
                latent = pipe.vae.encode(images.to(device, dtype=torch.float16) * 2 - 1)
                ls = 0.18215 * latent.latent_dist.sample()
                
                inverted_latents = invert(pipe,start_latents = ls, class_labels=sample['labels'], num_inference_steps = 50, device=device).cpu()
                latents.append(torch.flatten(inverted_latents.permute(1,0,2,3,4), start_dim=2))

        latents = torch.cat(latents, dim=0).cpu()
        mean = latents.mean(dim=0)
        variance = latents.var(dim=0)
        torch.save({"mean": mean, "variance": variance}, os.path.join(args.save_dir,class_dir+'.pt'))

            



if __name__ == "__main__":
    main()