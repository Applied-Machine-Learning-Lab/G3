This is the code repository for paper "G3: An Effective and Adaptive Framework for Worldwide Geolocalization Using Large Multi-Modality Models"

# MP16-Pro

You can download the images and metadata of MP16-Pro from huggingface: [Jia-py/MP16-Pro](https://huggingface.co/datasets/Jia-py/MP16-Pro/tree/main)

# Data

IM2GPS3K: [images](http://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip) | [metadata](https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/meta/im2gps3k_places365.csv)

YFCC4K: [images](http://www.mediafire.com/file/3og8y3o6c9de3ye/yfcc4k.zip) | [metadata](https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_places365.csv)

# Checkpoint

You can download the checkpoints and retrieval index from [Jia-py/G3-checkpoint](https://huggingface.co/Jia-py/G3-checkpoint)

# Environment Setting

```bash
# test on cuda12.0
conda create -n g3 python=3.9
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate huggingface_hub pandas
```

If there are any issues with transformers, you may try `transformers==4.42.0`.

# Quick Use

## Similarity between GPS and Images

```python
import torch
import numpy as np
from PIL import Image
from utils.G3 import G3

model = G3('cuda')
model.load_state_dict(torch.load('./checkpoints/g3.pth'))
image = Image.open(your_img_path).convert('RGB')
image = model.vision_processor(images=image, return_tensors='pt')['pixel_values'].reshape(3,224,224)

images = image.reshape(1,3,224,224) # pretend as a batch

images = images.to(args.device) # b,3,224,224
image_embeds = model.vision_projection_else_2(model.vision_projection(model.vision_model(images)[1]))
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True) # b, 768

gps_batch = torch.tensor([[0,0],[0,0]]).to('cuda').reshape(1,2,2) # (latitude, longitude)
gps_batch = gps_batch.to(args.device) # b,n,2; n is the number of candidates
gps_input = gps_batch.clone().detach()
b, c, _ = gps_input.shape
gps_input = gps_input.reshape(b*c, 2)
location_embeds = model.location_encoder(gps_input)
location_embeds = model.location_projection_else(location_embeds.reshape(b*c, -1))
location_embeds = location_embeds / location_embeds.norm(p=2, dim=-1, keepdim=True)
location_embeds = location_embeds.reshape(b, c, -1) #  b, c, 768

similarity = torch.matmul(image_embeds.unsqueeze(1), location_embeds.permute(0, 2, 1)) # b, 1, c
similarity = similarity.squeeze(1).cpu().detach().numpy()
max_idxs = np.argmax(similarity, axis=1)
```

This code can be easily adapted to calculate the similarity between text and images. Please check the source code in G3. The `vision_projection_else_2` layer should be modified accordingly.

# Running samples

1. Geo-alignment

You can run `python run_G3.py` to train the model.

2. Geo-diversification

First, you need to build the index file using `python IndexSearch.py`. 

Parameters in IndexSearch.py
- index name --> which model you want to use for embedding
- dataset --> im2gps3k or yfcc4k
- database --> default mp16

Then, you also need to construct index for negative samples by modifying images_embeds to -1 * images_embeds

Then, you can run `llm_predict_hf.py` or `llm_predict.py` to generate llm predictions.

After that, `running aggregate_llm_predictions.py` to aggregate the predictions.

3. Geo-verification

`python IndexSearch.py --index=g3 --dataset=im2gps3k or yfcc4k` to verificate predictions and evaluate.

# Citation

```bib
@article{jia2024g3,
  title={G3: an effective and adaptive framework for worldwide geolocalization using large multi-modality models},
  author={Jia, Pengyue and Liu, Yiding and Li, Xiaopeng and Zhao, Xiangyu and Wang, Yuhao and Du, Yantong and Han, Xiao and Wei, Xuetao and Wang, Shuaiqiang and Yin, Dawei},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={53198--53221},
  year={2024}
}
```
