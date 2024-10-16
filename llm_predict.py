import requests
import base64
import os
import re
import pandas as pd
import numpy as np
import ast
from pandarallel import pandarallel
from tqdm import tqdm
import argparse

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_response(image_path, base_url, api_key, model_name, detail="low", max_tokens=200, temperature=1.2, n=10):
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model_name,
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": """Suppose you are an expert in geo-localization, you have the ability to give two number GPS coordination given an image.
                Please give me the location of the given image.
                Remember, you must have an answer, just output your best guess, don't answer me that you can't give a location.
                Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}.
                Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}."""
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": detail
                }
                }
            ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n
    }

    response = requests.post(base_url, headers=headers, json=payload, timeout=(30,60))
    ans = []
    for choice in response.json()['choices']:
        try:
            ans.append(choice['message']['content'])
        except:
            ans.append('{"latitude": 0.0,"longitude": 0.0}')
    return ans

def get_response_rag(image_path, base_url, api_key, model_name, candidates_gps, reverse_gps, detail="low", max_tokens=200, temperature=1.2, n=10):
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model_name,
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"""Suppose you are an expert in geo-localization, Please analyze this image and give me a guess of the location.
                Your answer must be to the coordinates level in (latitude, longitude) format.
                For your reference, these are coordinates of some similar images: {candidates_gps}, and these are coordinates of some dissimilar images: {reverse_gps}.
                Remember, you must have an answer, just output your best guess, don't answer me that you can't give an location.
                Your answer should be in the following JSON format without any other information: {{"latitude": float,"longitude": float}}.
                Your answer should be in the following JSON format without any other information: {{"latitude": float,"longitude": float}}.
                """
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": detail
                }
                }
            ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n
    }

    response = requests.post(base_url, headers=headers, json=payload, timeout=(30,60))
    ans = []
    for choice in response.json()['choices']:
        try:
            ans.append(choice['message']['content'])
        except:
            ans.append('{"latitude": 0.0,"longitude": 0.0}')
    return ans

def process_row(row, base_url, api_key, model_name, root_path, image_path):
    image_path = os.path.join(root_path, image_path, row["IMG_ID"])
    try:
        response = get_response(image_path, base_url, api_key, model_name)
    except Exception as e:
        response = "None"
        print(e)
    row['response'] = response
    return row

def process_row_rag(row, base_url, api_key, model_name, root_path, image_path, rag_sample_num):
    image_path = os.path.join(root_path, image_path, row["IMG_ID"])
    try:
        #candidates_gps = [eval(row[f'candidate_{i}_gps']) for i in range(rag_sample_num)]
        candidates_gps = [row[f'candidate_{i}_gps'] for i in range(rag_sample_num)]
        candidates_gps = str(candidates_gps)
        #reverse_gps = [eval(row[f'reverse_{i}_gps']) for i in range(rag_sample_num)]
        reverse_gps = [row[f'reverse_{i}_gps'] for i in range(rag_sample_num)]
        reverse_gps = str(reverse_gps)
        response = get_response_rag(image_path, base_url, api_key, model_name, candidates_gps, reverse_gps)
    except Exception as e:
        response = "None"
        print(e)
    row['rag_response'] = response
    return row

def check_conditions(coord_str):
    if coord_str.startswith('[]') or coord_str.startswith('None'):
        return True
    try:
        coordinates = ast.literal_eval(coord_str)
        return float(coordinates[0]) == 0.0
    except:
        return False
def run(args):
    api_key = args.api_key
    model_name = args.model_name
    base_url = args.base_url
    root_path = args.root_path
    text_path = args.text_path
    image_path = args.image_path
    result_path = args.result_path
    rag_path = args.rag_path
    process = args.process
    rag_sample_num = args.rag_sample_num
    searching_file_name = args.searching_file_name

    if process == 'predict':
        if os.path.exists(os.path.join(root_path, result_path)):
            df = pd.read_csv(os.path.join(root_path, result_path))
            df_rerun = df[df['response'].isna()]
            print('Need Rerun:', df_rerun.shape[0])
            df_rerun = df_rerun.parallel_apply(lambda row: process_row(row, base_url, api_key, model_name, root_path, image_path), axis=1)
            df.update(df_rerun)
            df.to_csv(os.path.join(root_path, result_path), index=False)
        else:
            df = pd.read_csv(os.path.join(root_path, text_path))
            df = df.parallel_apply(lambda row: process_row(row, base_url, api_key, model_name, root_path, image_path), axis=1)
            df.to_csv(os.path.join(root_path, result_path), index=False)

    if process == 'extract':
        df = pd.read_csv(os.path.join(root_path, result_path))
        pattern = r'[-+]?\d+\.\d+'
        df['coordinates'] = df['response'].apply(lambda x: re.findall(pattern, x))
        df.to_csv(os.path.join(root_path, result_path), index=False)

    if process == 'rag':
        database_df = pd.read_csv('./data/MP16_Pro_filtered.csv')
        if not os.path.exists(os.path.join(root_path, str(rag_sample_num) + '_' + rag_path)):
            df = pd.read_csv(os.path.join(root_path, text_path))
            I = np.load('./index/{}.npy'.format(searching_file_name))
            reverse_I = np.load('./index/{}_reverse.npy'.format(searching_file_name))
            for i in tqdm(range(df.shape[0])):
                candidate_idx_lis = I[i]
                candidate_gps = database_df.loc[candidate_idx_lis, ['LAT', 'LON', 'city', 'state', 'country']].values
                for idx, (latitude, longitude, city, state, country) in enumerate(candidate_gps):
                    df.loc[i, f'candidate_{idx}_gps'] = f'[{latitude}, {longitude}]'
                reverse_idx_lis = reverse_I[i]
                reverse_gps = database_df.loc[reverse_idx_lis, ['LAT', 'LON', 'city', 'state', 'country']].values
                for idx, (latitude, longitude, city, state, country) in enumerate(reverse_gps):
                    df.loc[i, f'reverse_{idx}_gps'] = f'[{latitude}, {longitude}]'
            df.to_csv(os.path.join(root_path, str(rag_sample_num) + '_' + rag_path), index=False)
            df = df.parallel_apply(lambda row: process_row_rag(row, base_url, api_key, model_name, root_path, image_path, rag_sample_num), axis=1)
            df.to_csv(os.path.join(root_path, str(rag_sample_num) + '_' + rag_path), index=False)
        else:
            df = pd.read_csv(os.path.join(root_path, str(rag_sample_num) + '_' + rag_path))
            # df_rerun = df[df['rag_coordinates'].apply(check_conditions)]
            df_rerun = df[df['rag_response'].isna()]
            print('Need Rerun:', df_rerun.shape[0])
            df_rerun = df_rerun.parallel_apply(lambda row: process_row_rag(row, base_url, api_key, model_name, root_path, image_path, rag_sample_num), axis=1)
            df.update(df_rerun)
            df.to_csv(os.path.join(root_path,  str(rag_sample_num) + '_' + rag_path), index=False)
            
    if process == 'rag_extract':
        df = pd.read_csv(os.path.join(root_path, rag_path)).fillna("None")
        pattern = r'[-+]?\d+\.\d+'
        df['rag_coordinates'] = df['rag_response'].apply(lambda x: re.findall(pattern, x))
        df.to_csv(os.path.join(root_path, rag_path), index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    api_key = "sk-xxx"
    model_name = "gpt-xxx" # gpt-4-vision-preview, gpt-4-turbo-2024-04-09
    base_url = "https://xxx"

    root_path = "./data/im2gps3k"
    text_path = "im2gps3k_places365.csv"
    image_path = "images"
    result_path = "llm_predict_results_zs.csv"
    rag_path = "llm_predict_results_rag.csv"
    process = 'rag' # predict, extract, rag, rag_extract
    rag_sample_num = 15
    searching_file_name = 'I_g3_im2gps3k'

    pandarallel.initialize(progress_bar=True, nb_workers=16)
    args.add_argument('--api_key', type=str, default=api_key)
    args.add_argument('--model_name', type=str, default=model_name)
    args.add_argument('--base_url', type=str, default=base_url)
    args.add_argument('--root_path', type=str, default=root_path)
    args.add_argument('--text_path', type=str, default=text_path)
    args.add_argument('--image_path', type=str, default=image_path)
    args.add_argument('--result_path', type=str, default=result_path)
    args.add_argument('--rag_path', type=str, default=rag_path)
    args.add_argument('--process', type=str, default=process)
    args.add_argument('--rag_sample_num', type=int, default=rag_sample_num)
    args.add_argument('--searching_file_name', type=str, default=searching_file_name)
    args = args.parse_args()
    print(args)

    run(args)


