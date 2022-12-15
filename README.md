# Gradient-based Label Correction for Debiasing in Visual Question Answering


## Data Setup
- Download UpDn features from [google drive](https://drive.google.com/drive/folders/1IXTsTudZtYLqmKzsXxIZbXfCnys_Izxr) into `/data/detection_features` folder
- Download questions/answers for VQAv2 and VQA-CPv2 by executing `bash tools/download.sh`
- Download visual cues/hints provided in [A negative case analysis of visual grounding methods for VQA](https://drive.google.com/drive/folders/1fkydOF-_LRpXK1ecgst5XujhyQdE6It7?usp=sharing) into `data/hints`. Note that we use caption based hints for grounding-based method reproduction, CGR and CGW.
- Preprocess process the data with `bash tools/process.sh`

## Training
Run
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cpv2 --mode MODE --debias gradient --topq 1 --topv -1 --qvp 5 --output [] 
```
to train a model.  In `main.py`, `import base_model` for UpDn baseline; `import base_model_ban as base_model` for BAN baseline; `import base_model_block as base_model` for S-MRL baseline.

Set `MODE` as `gge_iter` and `gge_tog` for our best performance model; `gge_d_bias` and `gge_q_bias` for single bias ablation; `base` for baseline model.

## Training ablations
For models in Sec. 3, execute `from train_ab import train` and `import base_model_ab as base_model` in `main.py`. Run
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cpv2 --mode MODE --debias METHODS --topq 1 --topv -1 --qvp 5 --output [] 
```

## Visualization
We provide visualization in `visualization.ipynb`. If you want to see other visualization by yourself, download MS-COCO 2014 to `data/images`.
