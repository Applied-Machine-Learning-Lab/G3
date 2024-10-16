import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import itertools
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .rff.layers import GaussianEncoding
from pyproj import Proj, Transformer

class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x

class CustomLocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8]):
        super(CustomLocationEncoder, self).__init__()

        self.sigma = sigma
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))

        proj_wgs84 = Proj('epsg:4326')
        proj_mercator = Proj('epsg:3857')
        self.transformer = Transformer.from_proj(proj_wgs84, proj_mercator, always_xy=True)

    def forward(self, input):
        lat = input[:, 0].float().detach().cpu().numpy()
        lon = input[:, 1].float().detach().cpu().numpy()
        projected_lon_lat = self.transformer.transform(lon, lat)
        location = []
        for coord in zip(*projected_lon_lat):
            location.append([coord[1],coord[0]])
        location = torch.Tensor(location).to('cuda')
        location = location / 20037508.3427892

        location_features = torch.zeros(location.shape[0], 512).to('cuda')

        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)

        return location_features


class G3(torch.nn.Module):
    def __init__(self, device):
        super(G3, self).__init__()
        self.device = device

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_model = clip_model.vision_model
        self.text_model = clip_model.text_model
        self.vision_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.text_processor = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_projection = clip_model.visual_projection
        self.text_projection = clip_model.text_projection

        self.logit_scale1 = nn.Parameter(torch.tensor(3.99))
        self.logit_scale2 = nn.Parameter(torch.tensor(3.99))
        self.logit_scale3 = nn.Parameter(torch.tensor(3.99))

        self.location_encoder = CustomLocationEncoder() # output batch_size, 3, 512
        #self.location_encoder = LocationEncoder(sigma=[2**0, 2**4, 2**8])
        self.vision_projection_else_1 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
        self.text_projection_else = nn.Sequential(nn.Linear(768,768), nn.ReLU(), nn.Linear(768, 768))

        self.vision_projection_else_2 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
        self.location_projection_else = nn.Sequential(nn.Linear(512,512), nn.ReLU(), nn.Linear(512, 768))

        # freeze CLIP
        self.vision_model.requires_grad_(False)
        self.vision_projection.requires_grad_(False)
        self.text_model.requires_grad_(False)
        self.text_projection.requires_grad_(False)

    def forward(self, images, texts, longitude, latitude, return_loss=True):

        vision_output = self.vision_model(images)[1]
        text_output = self.text_model(**texts)[1]
        image_embeds = self.vision_projection(vision_output)
        text_embeds = self.text_projection(text_output) # batch_size, 512
        this_batch_locations = torch.stack((latitude, longitude), dim=1)
        location_embeds = self.location_encoder(this_batch_locations)

        # phase _1
        image_embeds_1 = self.vision_projection_else_1(image_embeds)
        text_embeds_1 = self.text_projection_else(text_embeds.reshape(text_embeds.shape[0], -1))
        
        # normalized features
        image_embeds_1 = image_embeds_1 / image_embeds_1.norm(p=2, dim=-1, keepdim=True)
        text_embeds_1 = text_embeds_1 / text_embeds_1.norm(p=2, dim=-1, keepdim=True)

        # image with texts
        logit_scale = self.logit_scale1.exp()
        logits_per_texts_with_images = torch.matmul(text_embeds_1, image_embeds_1.t()) * logit_scale
        logits_per_images_with_texts = logits_per_texts_with_images.t()
        if return_loss: loss1 = self.clip_loss(logits_per_texts_with_images)

        loss_phase_1 = None
        if return_loss:
            loss_phase_1 = loss1

        # phase _2
        image_embeds_2 = self.vision_projection_else_2(image_embeds)
        location_embeds_2 = self.location_projection_else(location_embeds.reshape(location_embeds.shape[0], -1))

        # normalized features
        image_embeds_2 = image_embeds_2 / image_embeds_2.norm(p=2, dim=-1, keepdim=True)
        location_embeds_2 = location_embeds_2 / location_embeds_2.norm(p=2, dim=-1, keepdim=True)

        # image with location
        logit_scale = self.logit_scale2.exp()
        logits_per_locations_with_images = torch.matmul(location_embeds_2, image_embeds_2.t()) * logit_scale
        logits_per_images_with_locations = logits_per_locations_with_images.t()
        loss_phase_2 = None
        if return_loss: loss_phase_2 = self.clip_loss(logits_per_locations_with_images)

        loss = loss_phase_1 + loss_phase_2

        return {
            'logits_per_texts_with_images': logits_per_texts_with_images,
            'logits_per_images_with_texts': logits_per_images_with_texts,
            'logits_per_locations_with_images': logits_per_locations_with_images,
            'logits_per_images_with_locations': logits_per_images_with_locations,
            'logits_per_locations_with_texts': None,
            'logits_per_texts_with_locations': None,
            'loss': loss,
            'vision_output': vision_output,
            'text_output': text_output,
            'image_embeds': image_embeds,
            'text_embeds': text_embeds
        }


    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0
