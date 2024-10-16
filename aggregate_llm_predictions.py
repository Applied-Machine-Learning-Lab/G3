import pandas as pd
import re
import ast
from tqdm import tqdm

df_raw = pd.read_csv('./data/im2gps3k/im2gps3k_places365.csv')
zs_df = pd.read_csv('./data/im2gps3k/llm_predict_results_zs.csv')
rag_5_df = pd.read_csv('./data/im2gps3k/5_llm_predict_results_rag.csv')
rag_10_df = pd.read_csv('./data/im2gps3k/10_llm_predict_results_rag.csv')
rag_15_df = pd.read_csv('./data/im2gps3k/15_llm_predict_results_rag.csv')

pattern = r'[-+]?\d+\.\d+'

for i in tqdm(range(zs_df.shape[0])):
    response = zs_df.loc[i, 'response']
    response = ast.literal_eval(response)
    for idx, content in enumerate(response):
        try:
            match = re.findall(pattern, content)
            latitude = match[0]
            longitude = match[1]
            df_raw.loc[i, f'zs_{idx}_latitude'] = latitude
            df_raw.loc[i, f'zs_{idx}_longitude'] = longitude
        except:
            df_raw.loc[i, f'zs_{idx}_latitude'] = '0.0'
            df_raw.loc[i, f'zs_{idx}_longitude'] = '0.0'

for i in tqdm(range(df_raw.shape[0])):
    response = rag_5_df.loc[i, 'rag_response']
    response = ast.literal_eval(response)
    for idx, content in enumerate(response):
        try:
            match = re.findall(pattern, content)
            latitude = match[0]
            longitude = match[1]
            df_raw.loc[i, f'5_rag_{idx}_latitude'] = latitude
            df_raw.loc[i, f'5_rag_{idx}_longitude'] = longitude
        except:
            df_raw.loc[i, f'5_rag_{idx}_latitude'] = '0.0'
            df_raw.loc[i, f'5_rag_{idx}_longitude'] = '0.0'

for i in tqdm(range(df_raw.shape[0])):
    response = rag_10_df.loc[i, 'rag_response']
    response = ast.literal_eval(response)
    for idx, content in enumerate(response):
        try:
            match = re.findall(pattern, content)
            latitude = match[0]
            longitude = match[1]
            df_raw.loc[i, f'10_rag_{idx}_latitude'] = latitude
            df_raw.loc[i, f'10_rag_{idx}_longitude'] = longitude
        except:
            df_raw.loc[i, f'10_rag_{idx}_latitude'] = '0.0'
            df_raw.loc[i, f'10_rag_{idx}_longitude'] = '0.0'

for i in tqdm(range(df_raw.shape[0])):
    response = rag_15_df.loc[i, 'rag_response']
    response = ast.literal_eval(response)
    for idx, content in enumerate(response):
        try:
            match = re.findall(pattern, content)
            latitude = match[0]
            longitude = match[1]
            df_raw.loc[i, f'15_rag_{idx}_latitude'] = latitude
            df_raw.loc[i, f'15_rag_{idx}_longitude'] = longitude
        except:
            df_raw.loc[i, f'15_rag_{idx}_latitude'] = '0.0'
            df_raw.loc[i, f'15_rag_{idx}_longitude'] = '0.0'

df_raw.to_csv('./data/im2gps3k/im2gps3k_prediction.csv', index=False)
