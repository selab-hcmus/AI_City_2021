This module handles the following tasks:
- Train the retrieval model
- Inference on test set to get candidate results, which is then refined to produce the final submission.

## Module organization 
- `data`: stores required files to train the model
- `config`: contains our best configurations
- `experiemnts`: contains experiment results such as training log, model checkpoints, etc.
- `results`: stores the output of inference step.

## Prepare
Install dependencies
```
pip install -r requirements.txt
```
The necessary data to train is produced from previous step ( [classifier](../classifier)) or can be downloaded at [gdrive](https://drive.google.com/drive/folders/11AziLNiIZvyb10AkyodOMv_za64wSB-a?usp=sharing).

## Train
```
sh script/PREPARE.sh
sh script/TRAIN.sh
```
When training from scratch, please change the model checkpoint path in *uptrain step* in `TRAIN.sh`. Currently we hardcode our best model (download from [gdrive](https://drive.google.com/drive/folders/117cxzdS6JWNW3KuJwOrQcoW8Qm91CN-H?usp=sharing))

## Inference
```
sh script/INFERENCE.sh
```
The final result will be saved in `results/result.json`