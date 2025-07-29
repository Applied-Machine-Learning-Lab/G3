import torch
import numpy as np
from PIL import Image
from utils.G3 import G3

image_path = 'xxx'
device = 'cuda'

model = G3(device).to(device)
model.load_state_dict(torch.load('/checkpoints/g3.pth'))
image = Image.open(image_path).convert('RGB')
image = model.vision_processor(images=image, return_tensors='pt')['pixel_values'].reshape(3,224,224)

images = image.reshape(1,3,224,224) # pretend as a batch

images = images.to(device) # b,3,224,224
image_embeds = model.vision_projection_else_2(model.vision_projection(model.vision_model(images)[1]))
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True) # b, 768

gps_batch = torch.tensor([[10,20],[0,0]]).reshape(1,2,2) # [[latitude1, longitude1],[latitude2, longitude2]]
gps_batch = gps_batch.to(device) # b,n,2; n is the number of candidates
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
print('similarity:', similarity)
# similarity: [[0.05875633 0.10544068]]