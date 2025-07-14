import json
import torch
import random
import numpy as np
from torchvision import transforms as T
from transformers import CLIPModel, AutoTokenizer

from misc import prompts
from models.scheduling_ddim import DDIMScheduler


class SupConLoss(torch.nn.Module):
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
        self.init_clip()
        self.description_file = args.description_path
        self.load_class_description()
        self.contrast_criterion = SupConLoss()

        self.prompts = prompts.prompt_templates
        self.diffusion_steps = args.diffusion_steps
        self.clip_alpha = args.clip_alpha
        self.num_neg_samples = args.num_neg_samples
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
            class_name_feat = self.extract_clip_text_embed(self.class_name_list)
            name_sims = (class_name_feat @ class_name_feat.T).cpu()
            name_sims -= torch.eye(len(name_sims))
            name_sims = name_sims.numpy()
        for class_index, sim_indices in zip(self.class_indices, name_sims):
            self.neighbors[class_index] = list(sim_indices)

    def init_clip(self):
        self.clip_model = CLIPModel.from_pretrained('laion/CLIP-ViT-L-14-laion2B-s32B-b82K').to('cuda')
        self.clip_tokenizer = AutoTokenizer.from_pretrained('laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
        self.clip_normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )

    def extract_clip_image_embed(self, image):
        image = self.clip_transform(image).unsqueeze(0)
        clip_feat = self.clip_model.encode_image(image)
        return clip_feat

    def extract_clip_text_embed(self, descriptions):
        input_text = self.clip_tokenizer(descriptions, padding=True, return_tensors='pt').to('cuda')
        text_features = self.clip_model.get_text_features(**input_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def cond_fn(self, sample, **kwargs):
        with torch.enable_grad():
            sample = torch.nn.Parameter(sample).requires_grad_()
            pseudo_image = self.pipe.vae.decode(sample / 0.18215, return_dict=False)[0]
            pseudo_image = T.Resize((224, 224))(pseudo_image) * 0.5 + 0.5
            pseudo_image = self.clip_normalize(pseudo_image)

            # Extract image embedding
            clip_feat_image = self.clip_model.get_image_features(pseudo_image)
            clip_feat_image = clip_feat_image / clip_feat_image.norm(dim=-1, keepdim=True)

            clip_loss = self.contrast_criterion(
                clip_feat_image, self.current_desc_embeddings,
                torch.cat((torch.zeros(5), torch.ones(len(self.current_desc_embeddings) - 5))).unsqueeze(0).cuda()
            )

            loss = self.clip_alpha * clip_loss

        return -torch.autograd.grad(loss, sample, allow_unused=True)[0]

    def init_img2img(self):
        from models.pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
        self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            'radames/stable-diffusion-2-1-unclip-img2img'
        )
        self.pipe.scheduler = DDIMScheduler.from_pretrained('radames/stable-diffusion-2-1-unclip-img2img', subfolder='scheduler')
        self.pipe = self.pipe.to('cuda')

    def sample_img2img(self, image, class_index):
        class_name = self.class_names[class_index]
        class_name = class_name.split(',')[0]
        pos_descriptions = self.descriptions[class_index]
        prompt = random.choice(self.prompts).format(class_name, '')

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
            neg_descriptions.append(self.class_names[rand_index] + 'with ' + self.descriptions[rand_index][np.random.randint(0, 4)])
        self.current_desc_embeddings = self.extract_clip_text_embed(pos_descriptions + neg_descriptions)
        new_image = self.pipe(image=image, prompt=prompt, cond_fn=self.cond_fn, num_inference_steps=self.diffusion_steps).images[0]

        new_image = new_image.resize((224, 224))

        return new_image
