import torch
import os
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import MP16Dataset
from utils.G3 import G3
from accelerate import Accelerator, DistributedDataParallelKwargs
import warnings

warnings.filterwarnings('ignore')

def train_1epoch(dataloader, eval_dataloader, earlystopper, model, vision_processor, text_processor, optimizer, scheduler, device, accelerator=None):
    model.train()
    t = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    for i, (images, texts, longitude, latitude) in enumerate(t):
        texts = text_processor(text=texts, padding='max_length', truncation=True, return_tensors='pt', max_length=77)
        images = images.to(device)
        texts = texts.to(device)
        longitude = longitude.to(device).float()
        latitude = latitude.to(device).float()
        optimizer.zero_grad()

        output = model(images, texts, longitude, latitude, return_loss=True)
        loss = output['loss']

        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        if i % 1 == 0:
            t.set_description('step {}, loss {}, lr {}'.format(i, loss.item(), scheduler.get_last_lr()[0]))
    scheduler.step()


def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # fine-tune
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    model = G3(device).to(device)
    location_encoder_dict = torch.load('location_encoder.pth') # from geoclip
    model.location_encoder.load_state_dict(location_encoder_dict)

    dataset = MP16Dataset(vision_processor = model.vision_processor, text_processor = model.text_processor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=5)


    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
            params.append(param)

    optimizer = torch.optim.AdamW([param for name,param in model.named_parameters() if param.requires_grad], lr=3e-5, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.87)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    eval_dataloader = None
    earlystopper = None
    for epoch in range(10):
        train_1epoch(dataloader, eval_dataloader, earlystopper, model, model.vision_processor, model.text_processor, optimizer, scheduler, device, accelerator)
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model, 'checkpoints/g3_{}_.pth'.format(epoch))

if __name__ == '__main__':
    main()
