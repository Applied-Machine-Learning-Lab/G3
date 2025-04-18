This is the code repository for paper "G3: An Effective and Adaptive Framework for Worldwide Geolocalization Using Large Multi-Modality Models"

# MP16-Pro

You can download the images and metadata of MP16-Pro from huggingface: [Jia-py/MP16-Pro](https://huggingface.co/datasets/Jia-py/MP16-Pro/tree/main)

# Data

IM2GPS3K: [images](http://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip) | [metadata](https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/meta/im2gps3k_places365.csv)

YFCC4K: [images](http://www.mediafire.com/file/3og8y3o6c9de3ye/yfcc4k.zip) | [metadata](https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_places365.csv)

# Environment Setting

```bash
# test on cuda12.0
conda create -n g3 python=3.9
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate huggingface_hub pandas
```

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
