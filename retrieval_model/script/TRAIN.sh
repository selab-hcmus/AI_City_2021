# Train model 
echo 'TRAIN'
python tools/train_action.py -c 'config/v0_Jun10.yaml' \
--data_path 'data/aic21' 
