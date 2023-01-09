# Gradient-based Label Correction for Debiasing in Visual Question Answering


## Install
```
pip install -r requirements.txt
```
## Data Setup
- Download UpDn features from [google drive](https://drive.google.com/drive/folders/1IXTsTudZtYLqmKzsXxIZbXfCnys_Izxr) into `/data/detection_features` folder
- Download questions/answers for VQAv2 and VQA-CPv2 by executing `bash tools/download.sh`
<!-- - Download visual cues/hints provided in [A negative case analysis of visual grounding methods for VQA](https://drive.google.com/drive/folders/1fkydOF-_LRpXK1ecgst5XujhyQdE6It7?usp=sharing) into `data/hints`. -->
- Preprocess process the data with `bash tools/process.sh`

## Training
Run
```
CUDA_VISIBLE_DEVICES=0 python main.py -dataset cpv2 -mode base -scale sin -output base
```
Set `mode` as `gld_iter` and `gld_joint` for our model in iterative and joint training; `base` for baseline model;`gld_reg`.for w/ regularization term version 
Set `dataset` as `v2` for the general VQA task; `cpv2` for the VQA task which enhance the language prior
```
CUDA_VISIBLE_DEVICES=0 python gld_iter_ce.py
CUDA_VISIBLE_DEVICES=0 python gld_joint_ce.py
```
to see the difference with crossentropy as loss;

## Visualization
To see visualization, set `visual` as `True`
```
CUDA_VISIBLE_DEVICES=0 python main.py -dataset cpv2 -mode gld_reg -scale sin -visual True -qid 140 -output vis
```
change `qid` to see the different question and image pairs
and change mode to see the visualization result to see in different setting
