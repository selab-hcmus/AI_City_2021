# Traffic Video Event Retrieval via Text Query using Vehicle Appearance and Motion Attributes
CVPR AI City Challenge 2021 (HCMUS Team)

This is the code for our work at AI City Challenge, CVPR 2021.
![system overview](assets/system_overview.png)


## Project organization
Our system contains 4 main modules: 
- **Textual attribute extraction**: Apply SRL toolkit on the input text query to extract color, vehicle type and action of the target object.
- **Visual attribute extraction**: given the target object boxes, Classifier aims to classify color, vehicle type and extract feature vector. Detector uses the object tracklets to identify the vehicle turn or stop.
- **Retrieval model**: Representation learning based model to handle the retreieval task
- **Refinement process**: Refine and produce final results 

## Data preparation
1. Download the challenge dataset and place in folder `dataset`
2. Run each module to produce input data for next steps or directly download them from our [gdrive]() (*Uploading*)

## Train
To train the whole system from scratch, run each module in the following order:
1. Textual attribute extraction: [srl_extraction](./srl_extraction), [srl_handler](./srl_handler)
1. Visual attribute extraction: [classifier](./classifier), [detector](./detector)
1. Retrieval model: [retrieval_model](./retrieval_model)
1. Refinement process: [refinement](./refinement)

In each folder, we also provide a notebook for easy setup and usage.

## Acknowledgement
The implementation of the Retrieval Model is customized from the great work [COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning](https://github.com/gingsi/coot-videotext).

The Classifier is modified from the well-organized repo [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch).

The toolkit used for SRL Extraction step is taken from the [AllenNLP library](https://github.com/allenai/allennlp).

## Citations
Please consider citing this project in your publications if it helps your research: *Uploading*

------------------
The code is used for academic purpose only.

