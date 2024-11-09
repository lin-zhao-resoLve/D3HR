import torch

from diffusers import ConsistencyModelPipeline

device = "cuda"
# Load the cd_imagenet64_l2 checkpoint.
model_id_or_path = "/hdd2/linzhao/imagenet_diffusion"
pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# Onestep Sampling
image = pipe(num_inference_steps=1).images[0]
image.save("cd_imagenet64_l2_onestep_sample.png")

# Onestep sampling, class-conditional image generation
# ImageNet-64 class label 145 corresponds to king penguins
image = pipe(num_inference_steps=1, class_labels=145).images[0]
image.save("cd_imagenet64_l2_onestep_sample_penguin.png")

# Multistep sampling, class-conditional image generation
# Timesteps can be explicitly specified; the particular timesteps below are from the original Github repo:
# https://github.com/openai/consistency_models/blob/main/scripts/launch.sh#L77
image = pipe(num_inference_steps=None, timesteps=[22, 0], class_labels=145).images[0]
image.save("cd_imagenet64_l2_multistep_sample_penguin.png")