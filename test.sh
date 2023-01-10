#demo including attention visualization
CUDA_VISIBLE_DEVICES=0 python test.py -load_checkpoint_path logs/gldreg -visual True -qid 140 -output analysis/gld