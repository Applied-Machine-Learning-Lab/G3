import faiss
import torch
import numpy as np
import os
import argparse
import pandas as pd
import ast
import itertools
from PIL import Image
from geopy.distance import geodesic
from transformers import CLIPImageProcessor, CLIPModel
from utils.utils import MP16Dataset, im2gps3kDataset, yfcc4kDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

def build_index(args):
    if args.index == 'g3':
        model = torch.load('./checkpoints/g3.pth', map_location='cuda:0')
        model.requires_grad_(False)
        vision_processor = model.vision_processor
        dataset = MP16Dataset(vision_processor = model.vision_processor, text_processor = None)
        index_flat = faiss.IndexFlatIP(768*3)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=3)
        model.eval()
        t= tqdm(dataloader)
        for i, (images, texts, longitude, latitude) in enumerate(t):
            images = images.to(args.device)
            vision_output = model.vision_model(images)[1]
            image_embeds = model.vision_projection(vision_output)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

            image_text_embeds = model.vision_projection_else_1(model.vision_projection(vision_output))
            image_text_embeds = image_text_embeds / image_text_embeds.norm(p=2, dim=-1, keepdim=True)

            image_location_embeds = model.vision_projection_else_2(model.vision_projection(vision_output))
            image_location_embeds = image_location_embeds / image_location_embeds.norm(p=2, dim=-1, keepdim=True)

            image_embeds = torch.cat([image_embeds, image_text_embeds, image_location_embeds], dim=1)
            index_flat.add(image_embeds.cpu().detach().numpy())

        faiss.write_index(index_flat, f'./index/{args.index}.index')

def search_index(args, index, topk):
    print('start searching...')
    if args.dataset == 'im2gps3k':
        if args.index == 'g3':
            model = torch.load('./checkpoints/g3.pth', map_location='cuda:0')
            model.requires_grad_(False)
            vision_processor = model.vision_processor
            dataset = im2gps3kDataset(vision_processor = vision_processor, text_processor = None)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=5)
            test_images_embeds = np.empty((0, 768*3))
            model.eval()
            print('generating embeds...')
            t = tqdm(dataloader)
            for i, (images, texts, longitude, latitude) in enumerate(t):
                images = images.to(args.device)
                vision_output = model.vision_model(images)[1]
                image_embeds = model.vision_projection(vision_output)
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

                image_text_embeds = model.vision_projection_else_1(model.vision_projection(vision_output))
                image_text_embeds = image_text_embeds / image_text_embeds.norm(p=2, dim=-1, keepdim=True)

                image_location_embeds = model.vision_projection_else_2(model.vision_projection(vision_output))
                image_location_embeds = image_location_embeds / image_location_embeds.norm(p=2, dim=-1, keepdim=True)

                image_embeds = torch.cat([image_embeds, image_text_embeds, image_location_embeds], dim=1)
                test_images_embeds = np.concatenate([test_images_embeds, image_embeds.cpu().detach().numpy()], axis=0)
            print(test_images_embeds.shape)
            test_images_embeds = test_images_embeds.reshape(-1, 768*3)
            print('start searching NN...')
            D, I = index.search(test_images_embeds, topk)
            print(I)
            return D, I
    elif args.dataset == 'yfcc4k':
        if args.index == 'g3':
            model = torch.load('./checkpoints/g3.pth', map_location='cuda:0')
            model.requires_grad_(False)
            vision_processor = model.vision_processor
            dataset = yfcc4kDataset(vision_processor = vision_processor, text_processor = None)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=5)
            test_images_embeds = np.empty((0, 768*3))
            model.eval()
            print('generating embeds...')
            t = tqdm(dataloader)
            for i, (images, texts, longitude, latitude) in enumerate(t):
                images = images.to(args.device)
                vision_output = model.vision_model(images)[1]
                image_embeds = model.vision_projection(vision_output)
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

                image_text_embeds = model.vision_projection_else_1(model.vision_projection(vision_output))
                image_text_embeds = image_text_embeds / image_text_embeds.norm(p=2, dim=-1, keepdim=True)

                image_location_embeds = model.vision_projection_else_2(model.vision_projection(vision_output))
                image_location_embeds = image_location_embeds / image_location_embeds.norm(p=2, dim=-1, keepdim=True)

                image_embeds = torch.cat([image_embeds, image_text_embeds, image_location_embeds], dim=1)
                test_images_embeds = np.concatenate([test_images_embeds, image_embeds.cpu().detach().numpy()], axis=0)
            print(test_images_embeds.shape)
            test_images_embeds = test_images_embeds.reshape(-1, 768*3)
            print('start searching NN...')
            D, I = index.search(test_images_embeds, topk)
            return D, I

class GeoImageDataset(Dataset):
    def __init__(self, dataframe, img_folder, topn, vision_processor, database_df, I):
        self.dataframe = dataframe
        self.img_folder = img_folder
        self.topn = topn
        self.vision_processor = vision_processor
        self.database_df = database_df
        self.I = I

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = f'{self.img_folder}/{self.dataframe.loc[idx, "IMG_ID"]}'
        image = Image.open(img_path).convert('RGB')
        image = self.vision_processor(images=image, return_tensors='pt')['pixel_values'].reshape(3,224,224)
        
        gps_data = []
        search_top1_latitude, search_top1_longitude = self.database_df.loc[self.I[idx][0], ['LAT', 'LON']].values
        rag_5, rag_10, rag_15, zs = [],[],[],[]
        for j in range(self.topn):
            gps_data.extend([
                float(self.dataframe.loc[idx, f'5_rag_{j}_latitude']),
                float(self.dataframe.loc[idx, f'5_rag_{j}_longitude']),
                float(self.dataframe.loc[idx, f'10_rag_{j}_latitude']),
                float(self.dataframe.loc[idx, f'10_rag_{j}_longitude']),
                float(self.dataframe.loc[idx, f'15_rag_{j}_latitude']),
                float(self.dataframe.loc[idx, f'15_rag_{j}_longitude']),
                float(self.dataframe.loc[idx, f'zs_{j}_latitude']),
                float(self.dataframe.loc[idx, f'zs_{j}_longitude']),
                search_top1_latitude,
                search_top1_longitude
            ])
        
        gps_data = np.array(gps_data).reshape(-1, 2)
        return image, gps_data, idx

def evaluate(args, I):
    print('start evaluation')
    if args.database == 'mp16':
        database = args.database_df
        df = args.dataset_df
        df['NN_idx'] = I[:, 0]
        df['LAT_pred'] = df.apply(lambda x: database.loc[x['NN_idx'],'LAT'], axis=1)
        df['LON_pred'] = df.apply(lambda x: database.loc[x['NN_idx'],'LON'], axis=1)

        df_llm = pd.read_csv(f'./data/{args.dataset}/{args.dataset}_prediction.csv')
        model = torch.load('./checkpoints/g3.pth', map_location='cuda:0')
        topn = 5 # number of candidates

        dataset = GeoImageDataset(df_llm, f'./data/{args.dataset}/images', topn, vision_processor=model.vision_processor, database_df=database, I=I)
        data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)

        for images, gps_batch, indices in tqdm(data_loader):
            images = images.to(args.device)
            image_embeds = model.vision_projection_else_2(model.vision_projection(model.vision_model(images)[1]))
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True) # b, 768

            gps_batch = gps_batch.to(args.device)
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
            
            # update DataFrame
            for i, max_idx in enumerate(max_idxs):
                final_idx = indices[i]
                final_idx = final_idx.item()
                final_latitude, final_longitude = gps_batch[i][max_idx]
                final_latitude, final_longitude = final_latitude.item(), final_longitude.item()
                if final_latitude < -90 or final_latitude > 90:
                    final_latitude = 0
                if final_longitude < -180 or final_longitude > 180:
                    final_longitude = 0
                df.loc[final_idx, 'LAT_pred'] = final_latitude
                df.loc[final_idx, 'LON_pred'] = final_longitude

        df['geodesic'] = df.apply(lambda x: geodesic((x['LAT'], x['LON']), (x['LAT_pred'], x['LON_pred'])).km, axis=1)
        print(df.head())
        df.to_csv(f'./data/{args.dataset}_{args.index}_results.csv', index=False)

        # 1, 25, 200, 750, 2500 km level
        print('2500km level: ', df[df['geodesic'] < 2500].shape[0] / df.shape[0])
        print('750km level: ', df[df['geodesic'] < 750].shape[0] / df.shape[0])
        print('200km level: ', df[df['geodesic'] < 200].shape[0] / df.shape[0])
        print('25km level: ', df[df['geodesic'] < 25].shape[0] / df.shape[0])
        print('1km level: ', df[df['geodesic'] < 1].shape[0] / df.shape[0])

if __name__ == '__main__':

    res = faiss.StandardGpuResources()

    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default='g3')
    parser.add_argument('--dataset', type=str, default='im2gps3k')
    parser.add_argument('--database', type=str, default='mp16')
    args = parser.parse_args()
    if args.dataset == 'im2gps3k':
        args.dataset_df = pd.read_csv('./data/im2gps3k/im2gps3k_places365.csv')
    elif args.dataset == 'yfcc4k':
        args.dataset_df = pd.read_csv('./data/yfcc4k/yfcc4k_places365.csv')

    if args.database == 'mp16':
        args.database_df = pd.read_csv('./data/MP16_Pro_filtered.csv')

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(f'./index'): os.makedirs(f'./index')
    if not os.path.exists(f'./index/{args.index}.index'):
        build_index(args)
    else:
        # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        if not os.path.exists(f'./index/I_{args.index}_{args.dataset}.npy'):
            index = faiss.read_index(f'./index/{args.index}.index')
            print('read index success')
            D,I = search_index(args, index, 20)
            np.save(f'./index/D_{args.index}_{args.dataset}.npy', D)
            np.save(f'./index/I_{args.index}_{args.dataset}.npy', I)
        else:
            D = np.load(f'./index/D_{args.index}_{args.dataset}.npy')
            I = np.load(f'./index/I_{args.index}_{args.dataset}.npy')
        evaluate(args, I)

