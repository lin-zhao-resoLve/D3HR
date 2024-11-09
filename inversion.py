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
from dataset import DiffDataset
import argparse
import os
from scipy import io
from diffusers import ConsistencyModelPipeline


# sample
@torch.no_grad()
def sample(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=None,
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images




## Inversion
@torch.no_grad()
def invert(
    pipe,
    start_latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=None,
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='/hdd2/linzhao/dataset/train',
        help="dataset_path",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='/hdd2/linzhao/dataset/train_latent',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    
    return args

def view_latents(pipe = None, inverted_latents = None):
    with torch.no_grad():
        im = pipe.decode_latents(inverted_latents[-1].unsqueeze(0))
    pipe.numpy_to_pil(im)[0]

def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.set_device(args.gpu)
    
    synset = io.loadmat('/hdd2/linzhao/dataset/ILSVRC2012_devkit_t12/data/meta.mat')
    
    ground_truth = open(os.path.join('/hdd2/linzhao/dataset/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
    lines = ground_truth.readlines()
    labels_class = [int(line[:-1]) for line in lines]
   
    
    diffdataset = DiffDataset(args.dataset_path)
    trainloader = torch.utils.data.DataLoader(diffdataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, drop_last=False)
    
    # pipe = ConsistencyModelPipeline.from_pretrained("/hdd2/linzhao/imagenet_diffusion")
    
    pipe = StableDiffusionPipeline.from_pretrained("/hdd2/linzhao/sd1.5_pretrained_model").to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) 

    
    for images, img_paths, labels in trainloader:
        with torch.no_grad():
            images = images.to(device)
            latent = pipe.vae.encode(images.to(device) * 2 - 1)
        ls = 0.18215 * latent.latent_dist.sample()
        for i in range(args.batch_size):
            img_path = img_paths[i]
            l = ls[i]
            
            val_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            ILSVRC_ID = labels_class[val_id-1]
            WIND = synset['synsets'][ILSVRC_ID-1][0][1][0]
            assert WIND == img_path.split('/')[-2]
            input_image_prompt = synset['synsets'][ILSVRC_ID-1][0][2][0]
            
            inverted_latent = invert(pipe,l.unsqueeze(0), str(input_image_prompt), num_inference_steps = 50, device=device)
            
            # test sample image
            start_step = 0
            image = sample(
                pipe,
                str(input_image_prompt),
                start_latents=inverted_latent[-(start_step + 1)][None],
                start_step=start_step,
                num_inference_steps=50,
                device=device
            )[0]
            
            tfms.ToPILImage()(images[i]).save('111.png')
            image.save('112.png')
            
            
            os.makedirs(args.save_dir, exist_ok=True)
            # torch.save(latent, os.path.join(args.save_dir, ))
            

# # Example: Assuming `latents` is a dictionary or a list of your latent representations
# # For demonstration, I'll create dummy latents
# latents = {
#     'class_1': torch.randn(256),  # Replace with your actual latent representation
#     'class_2': torch.randn(256),
#     # Add more classes as needed
# }

# # Save each latent representation
# for class_name, latent in latents.items():
#     file_path = os.path.join(save_dir, f'{class_name}.pth')
#     torch.save(latent, file_path)
#     print(f'Saved {class_name} latent to {file_path}')
  


if __name__ == "__main__":
    main()