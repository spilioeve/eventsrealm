# EvEntS ReaLM
This repo contains the code for our paper [EvEntS ReaLM: Event Reasoning of Entity States via Language Models](https://arxiv.org/abs/2211.05392), to-appear in EMNLP 2022.
We provide code to reproduce the experiments described in the paper. We also provide the preprocessed data used in the experiments to facilitate reproducibility of our experiments.

## Requirements
We use the `PyTorch` library in all our experiments. Our zero-prompt experiments are based on the [Simple Transforemrs](https://github.com/ThilinaRajapakse/simpletransformers) librarry. We adapt the [Huggingface Transformers library](https://github.com/huggingface/transformers) for single, all, and k attribute prompt experiments. 

For the single-Attribute prompt experiments with Roberta, we had to modify the original Transformer library. We therefore provide a full copy of the library with our changes. To reproduce these experiments, it is necessary to install this library. Move to the `transformers-single-all-attribute-prompt-experiments` directory and then execute `pip install -e .`

## Data, Outputs, and Checkpoints
We provide the preprocessed data for most experiments in the `data` directory. This data was adapted from the original [PiGLET](https://github.com/rowanz/piglet) and [OpenPI](https://github.com/allenai/openpi-dataset) datasets. If you use the data make sure to cite the original work. 

In addition, data, outputs, and checkpoint for the k-attribute prompt model available at the following drive link:
https://drive.google.com/drive/folders/1TVxD0biBa04OKZTugV6iBZtYrNFPYIaM?usp=sharing

## Experiments
The `/scripts` directory contains the commands to run the experiments described in the paper, separated based on the dataset (PiGLET vs OpenPI).


<!-- 
### PiGLET Experiments

1. N-Gram Classifier
The code for the n-gram classifier can be found at 

2. Zero-prompt classifier (Roberta)

3. All-Attribute classifier (T5)

### OpenPI Experiments

#### Zero-prompt classifier (Roberta)

#### Single-Attribute prompt (Roberta)

#### T5 Experiments

1. Single-Attribute prompt (T5)

2. All-Attribute prompt (T5)

3. k-Attribute prompt (T5)

### GPT-3 Experiments
1. Single-Attribute prompt few-shot (GPT-3 Babbage)

2. All-Attribute prompt few-shot (GPT-3 Davinci)
-->


## Citation
`
@article{lively2019analyzing,
  title={EVENTS REALM: Event Reasoning of Entity States via Language Models},
  author={Spiliopoulou, Evangelia and Pagnoni, Artidoro and Bisk, Yonatan and Hovy, Eduard},
  journal={arXiv preprint arXiv:2211.05392},
  year={2022}
}

`
