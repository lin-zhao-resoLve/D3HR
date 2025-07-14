import json
import random
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet18
from transformers import CLIPModel, AutoTokenizer

from .download import find_model
from diffusion import create_diffusion
from models.dit_models import DiT_models
from diffusers.models import AutoencoderKL


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.05, base_temperatue=0.05):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperatue
    
    def forward(self, image_features, text_features, text_labels):
        logits = (image_features @ text_features.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * text_labels
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = ((1 - text_labels) * log_prob).sum(1) / (1 - text_labels).sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class ImageSynthesizer(object):
    def __init__(self, args):
        self.vae = AutoencoderKL.from_pretrained(args.vae_path).to('cuda')
        self.clip_model = CLIPModel.from_pretrained('laion/CLIP-ViT-L-14-laion2B-s32B-b82K').to('cuda')
        self.clip_tokenizer = AutoTokenizer.from_pretrained('laion/CLIP-ViT-L-14-laion2B-s32B-b82K')

        # DiT model
        assert args.dit_image_size % 8 == 0, 'Image size must be divisible by 8'
        latent_size = args.dit_image_size // 8
        self.latent_size = latent_size
        self.dit = DiT_models[args.dit_model](
            input_size=latent_size,
            num_classes=args.num_dit_classes
        ).to('cuda')
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        self.dit.load_state_dict(state_dict, strict=False)

        # Diffusion
        self.diffusion = create_diffusion(str(args.diffusion_steps))

        # Class description
        self.description_file = args.description_path
        self.load_class_description()

        self.cfg_scale = args.cfg_scale
        self.clip_alpha = args.clip_alpha
        self.cls_alpha = args.cls_alpha
        self.num_pos_samples = 5
        self.num_neg_samples = args.num_neg_samples
        self.clip_normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.contrast_criterion = SupConLoss()
        self.neg_policy = args.neg_policy

    def load_class_description(self):
        with open(self.description_file, 'r') as fp:
            descriptions = json.load(fp)
        self.class_names = {}
        self.descriptions = {}

        for class_index, (class_name, description) in descriptions.items():
            self.class_names[class_index] = class_name
            self.descriptions[class_index] = description
        
        self.class_indices = list(self.class_names.keys())
        self.class_name_list = list(self.class_names.values())
        self.neighbors = {}
        with torch.no_grad():
            class_name_feat = self.extract_text_feature(self.class_name_list)
            name_sims = (class_name_feat @ class_name_feat.T).cpu()
            name_sims -= torch.eye(len(name_sims))
            name_sims = name_sims.numpy()
        for class_index, sim_indices in zip(self.class_indices, name_sims):
            self.neighbors[class_index] = list(sim_indices)

    def extract_text_feature(self, descriptions):
        input_text = self.clip_tokenizer(descriptions, padding=True, return_tensors='pt').to('cuda')
        text_feature = self.clip_model.get_text_features(**input_text)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return text_feature

    def cond_fn(self, x, t, y=None, text_features=None, contrastive=True, **kwargs):
        with torch.enable_grad():
            x = nn.Parameter(x).requires_grad_()
            real_x, _ = x.chunk(2, dim=0)
            pseudo_image = self.vae.decode(real_x / 0.18215, return_dict=False)[0]
            pseudo_image = T.Resize((224, 224))(pseudo_image) * 0.5 + 0.5
            pseudo_image = self.clip_normalize(pseudo_image)

            # Extract image embedding
            clip_feat_image = self.clip_model.get_image_features(pseudo_image)
            clip_feat_image = clip_feat_image / clip_feat_image.norm(dim=-1, keepdim=True)

            # Extract text embedding
            clip_feat_text_pos, clip_feat_text_neg = torch.split(
                text_features, [self.num_pos_samples, self.num_neg_samples]
            )

            if contrastive:
                clip_loss = self.contrast_criterion(
                    clip_feat_image, torch.cat((clip_feat_text_pos, clip_feat_text_neg), dim=0),
                    torch.cat((torch.zeros(self.num_pos_samples), torch.ones(self.num_neg_samples))).unsqueeze(0).cuda()
                )
            else:
                clip_loss = 1. - (clip_feat_image @ clip_feat_text_pos.T).mean()

            loss = self.clip_alpha * clip_loss

        return -torch.autograd.grad(loss, x, allow_unused=True)[0]

    def sample(self, original_label, class_index, batch_size=1, device=None):
        z = torch.randn(batch_size, 4, self.latent_size, self.latent_size, device=device)
        y = torch.tensor([original_label] * batch_size, device=device)

        # classifier-free guidance
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * batch_size, device=device)
        y = torch.cat([y, y_null], 0)

        pos_descriptions = self.descriptions[class_index]
        pos_descriptions = [self.class_names[class_index]+' with '+description for description in pos_descriptions]
        neg_descriptions = []
        if self.neg_policy == 'random':
            neg_classes = random.choices(self.class_indices, k=self.num_neg_samples)
        elif self.neg_policy == 'similar':
            max_indices = np.argsort(self.neighbors[class_index])[-self.num_neg_samples:]
            neg_classes = [self.class_indices[max_index] for max_index in max_indices]
        else:
            neg_classes = random.choices(self.class_indices, self.neighbors[class_index], k=self.num_neg_samples)
        for rand_index in neg_classes:
            neg_descriptions.append(self.class_names[rand_index] + ' with ' + self.descriptions[rand_index][np.random.randint(0, 4)])
        all_descriptions = pos_descriptions + neg_descriptions
        text_features = self.extract_text_feature(all_descriptions)

        model_kwargs = dict(
            y=y, cfg_scale=self.cfg_scale,
            text_features=text_features, contrastive=True
        )

        def get_samples(z):
            samples = self.diffusion.ddim_sample_loop(
                self.dit.forward_with_cfg, z.shape, z, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device,
                cond_fn=self.cond_fn
            )
            samples, _ = samples.chunk(2, dim=0)
            samples = self.vae.decode(samples / 0.18215).sample
            samples = T.Resize((224, 224))(samples)

            return samples

        samples = get_samples(z)

        return samples
