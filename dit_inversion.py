import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler 
from dataset import DiffDataset, TinyDataset, LocalDataset
import argparse
import os
from scipy import io
from diffusers import DiTPipeline
from diffusers.utils.torch_utils import randn_tensor


# sample
@torch.no_grad()
def sample1(
    pipe,
    class_labels,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
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

        # Instead, let's do it ourselves:
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
    guidance_scale=3.5,
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
        if i >= num_inference_steps - 1 -20:
            continue
        # t = pipe.scheduler.timesteps[i]

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

        # Store
        # intermediate_latents.append(latents_out)
    return latents_out

    # return torch.cat(intermediate_latents)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        # default="zh-plus/tiny-imagenet", # Load from website, but will face the index matching error in sample1() function
        default="../dataset/tiny-imagenet-200",
        help="dataset_path",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='/home/user1/workspace/leilu/linzhao/dataset/tiny-imagenet-200/vae_distribution',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
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
    # 过滤掉 batch 中为 None 的样本
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # 将图像和标签分别堆叠成批次
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


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.set_device(args.gpu)
   
    
    diffdataset = LocalDataset(args.dataset_path,args.start, args.end)
    trainloader = torch.utils.data.DataLoader(diffdataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers, drop_last=False, collate_fn=collate_fn)
    
    # pipe = ConsistencyModelPipeline.from_pretrained("/hdd2/linzhao/imagenet_diffusion")
    
    # pipe = StableDiffusionPipeline.from_pretrained("/hdd2/linzhao/sd1.5_pretrained_model").to(device)
    pipe = DiTPipeline.from_pretrained("/home/user1/workspace/leilu/linzhao/dit_imagenet", torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) 
    

    pipe = pipe.to(device)

    
    for sample in tqdm(trainloader):
        with torch.no_grad():
            images = sample['images'].to(device)
            latent = pipe.vae.encode(images.to(device, dtype=torch.float16) * 2 - 1)
            #ls = 0.18215 * latent.latent_dist.sample()
            for i in range(latent.latent_dist.mean.shape[0]):
                # latent = ls[i] 
                # os.makedirs(os.path.join(args.save_dir, sample['paths'][i].split('train')[1].split('/')[1]), exist_ok=True)
                # save_path = os.path.join(args.save_dir, sample['paths'][i].split('train')[1].split('/')[1], sample['paths'][i].split('train')[1].split('/')[3].split('.')[0]+'.pt') 
                
                os.makedirs(os.path.join(args.save_dir, 'mean', sample['paths'][i].split('train')[1].split('/')[1]), exist_ok=True)
                mean_save_path = os.path.join(args.save_dir, 'mean', sample['paths'][i].split('train')[1].split('/')[1], sample['paths'][i].split('train')[1].split('/')[3].split('.')[0]+'.pt') 
                torch.save(latent.latent_dist.mean, mean_save_path) 

                os.makedirs(os.path.join(args.save_dir, 'std', sample['paths'][i].split('train')[1].split('/')[1]), exist_ok=True)
                std_save_path = os.path.join(args.save_dir, 'std', sample['paths'][i].split('train')[1].split('/')[1], sample['paths'][i].split('train')[1].split('/')[3].split('.')[0]+'.pt') 
                torch.save(latent.latent_dist.std, std_save_path) 

                os.makedirs(os.path.join(args.save_dir, 'var', sample['paths'][i].split('train')[1].split('/')[1]), exist_ok=True)
                var_save_path = os.path.join(args.save_dir, 'var', sample['paths'][i].split('train')[1].split('/')[1], sample['paths'][i].split('train')[1].split('/')[3].split('.')[0]+'.pt') 
                torch.save(latent.latent_dist.var, var_save_path) 

                os.makedirs(os.path.join(args.save_dir, 'logvar', sample['paths'][i].split('train')[1].split('/')[1]), exist_ok=True)
                logvar_save_path = os.path.join(args.save_dir, 'logvar', sample['paths'][i].split('train')[1].split('/')[1], sample['paths'][i].split('train')[1].split('/')[3].split('.')[0]+'.pt') 
                torch.save(latent.latent_dist.logvar, logvar_save_path) 

            
            # val_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            # ILSVRC_ID = labels_class[val_id-1]
            # # WIND = synset['synsets'][ILSVRC_ID-1][0][1][0]
            # # assert WIND == img_path.split('/')[-2]
            # input_image_prompt = synset['synsets'][ILSVRC_ID-1][0][2][0]
            # words = input_image_prompt.split(', ')[-1]
            
            # inverted_latent = invert(pipe,start_latents = ls, class_labels=sample['labels'], num_inference_steps = 50, device=device)
            
            
            # for i in range(0, inverted_latent.shape[0]):
            #     latent = inverted_latent[i]  
            #     # image_path = image_paths[i//48]
            #     # os.makedirs(os.path.join(args.save_dir, sample['paths'][i].split('train')[1].split('/')[1]), exist_ok=True)
            #     # save_path = os.path.join(args.save_dir, sample['paths'][i].split('train')[1].split('/')[1], sample['paths'][i].split('train')[1].split('/')[3].split('.')[0]+'.pt') 
            #     # torch.save(latent, save_path)  
                
            #     # # test sample image
            #     start_step = 20
            #     image = sample1(
            #         pipe,
            #         class_labels=(sample['labels'][i]).unsqueeze(0),
            #         # start_latents=inverted_latent[(-(start_step + 2))*args.batch_size: (-(start_step+1))*args.batch_size],
            #         start_latents = latent.unsqueeze(0),
            #         start_step=start_step,
            #         num_inference_steps=50,
            #         device=device
            #     )
                
                
            #     # tfms.ToPILImage()(images[i]).save('111.png')
            #     # pipe.numpy_to_pil(image)[0].save('112.png')
            #     # print('1111')
            
            
          
            



if __name__ == "__main__":
    main()