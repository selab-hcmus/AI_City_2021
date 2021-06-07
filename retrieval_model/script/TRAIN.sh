# Train model 
echo 'TRAIN'
python tools/train.py -c 'config/retrieval/paper2020/aic21_6148_atn1.yaml' 

# Uptrain a few epoch on val set
echo 'UPTRAIN'
python tools/train.py -c 'config/retrieval/paper2020/aic21_6148_atn1.yaml' --load_model 'experiments/aic21_6148_atn1_run1/models/model_23_r1.pth' --uptrain
