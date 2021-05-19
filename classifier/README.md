# Classifier Module

This module handles the following tasks:
- Train the Color, Vehicle Type Classifier.
- Extract feature vector for retrieval model.
- Predict labels on visual test data for refinement process.

## Module organization 
- `data`: stores required files to train the model + vehicle boxes extracted from video tracks.
- `results`: stores best model weight and prediction results.

## Prepare
Install dependencies
```
cd 'EfficientNet-PyTorch'
pip install -e . 
cd '../'
```
The necessary data to train classifiers is provided from previous step ([srl handler](../srl_handler)) or can be downloaded at [gdrive](https://drive.google.com/drive/folders/11BwLV-UigyJOrKm3604syyTkcgjIemSW?usp=sharing).

## Train Classifier
```
python train.py
```
Best model weights will be saved in `./results`
Our trained model weight can be downloaded at [gdrive](https://drive.google.com/drive/folders/11DA5Zuc8kH537fqpbjlhHYQpBq4aXnHo?usp=sharing)

## Extract Feature
```
python extract_feat.py
```
The extracted feature will be saved in `../retrieval_model/data`

## Predict test labels
```
python label_prediction.py
```
The label outputs will be saved in `./results`


TODO
### Instruction for extract feat of tracked object