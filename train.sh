# UpDn model train on cpv2 dataset
CUDA_VISIBLE_DEVICES=0 python main.py -dataset cpv2 -mode base -output base_cp

# GLD w/regularization debiasing method train on VQAcp dataset
CUDA_VISIBLE_DEVICES=0 python main.py -dataset cpv2 -mode gld_reg -scale sin -output gld_cp

# UpDn model train on VQAv2 dataset
CUDA_VISIBLE_DEVICES=0 python main.py -dataset v2 -mode base -output base

# GLD debiasing method train on VQAv2 dataset
CUDA_VISIBLE_DEVICES=0 python main.py -dataset v2 -mode gld_iter -output gld_v2