# TabMDA: Tabular Manifold Data Augmentation for Any Classifier using Transformers with In-context Subsetting

[![Arxiv-Paper](https://img.shields.io/badge/Arxiv-Paper-yellow)](https://arxiv.org/abs/2406.01805)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

Official code for the paper [**TabMDA: Tabular Manifold Data Augmentation for Any Classifier using Transformers with In-context Subsetting**](https://arxiv.org/abs/2406.01805) accepted at [**ICML 1st Workshop on In-Context Learning**](https://iclworkshop.github.io/), 2024. 

by [Andrei Margeloiu](https://www.linkedin.com/in/andreimargeloiu/),
[Adrian Bazaga](https://bazaga.ai/),
[Nikola Simidjievski](https://simidjievskin.github.io/),
[Pietro Lio](https://www.cl.cam.ac.uk/~pl219/),
[Mateja Jamnik](https://www.cl.cam.ac.uk/~mj201/)


**TLDR**: TabMDA is a novel tabular data augmentation method that jointly embeds and augments the data using pre-trained in-context models. TabMDA is training-free, can be applied to any classifier, and our results show that it improves performance on small datasets.

![alt text](<images/TabMDA_method_with_caption.png>)

![alt text](<images/TabMDA_poster.png>)

**Abstract:** Tabular data is prevalent in many critical domains, yet it is often challenging to acquire in large quantities. This scarcity usually results in poor performance of machine learning models on such data. Data augmentation, a common strategy for performance improvement in vision and language tasks, typically underperforms for tabular data due to the lack of explicit symmetries in the input space. To overcome this challenge, we introduce TabMDA, a novel method for manifold data augmentation on tabular data. This method utilises a pre-trained in-context model, such as TabPFN, to map the data into an embedding space. TabMDA performs label-invariant transformations by encoding the data multiple times with varied contexts. This process explores the learned embedding space of the underlying in-context models, thereby enlarging the training dataset. TabMDA is a training-free method, making it applicable to any classifier. We evaluate TabMDA on five standard classifiers and observe significant performance improvements across various tabular datasets. Our results demonstrate that TabMDA provides an effective way to leverage information from pre-trained in-context models to enhance the performance of downstream classifiers.

# Citation
For attribution in academic contexts, please cite this work as:
```
@inproceedings{margeloiu2024tabmda,
  title={TabMDA: Tabular Manifold Data Augmentation for Any Classifier using Transformers with In-context Subsetting},
  author={Margeloiu, Andrei and Bazaga, Adri{\'a}n and Simidjievski, Nikola and Li{\`o}, Pietro and Jamnik, Mateja},
  booktitle={ICML 1st Workshop on In-Context Learning},
  year={2024}
}
```

## Installing the project 
1. You must have **conda** installed locally. Create a conda environment.

```
cd REPOSITORY
conda create python=3.10 --name tabmda
conda activate tabmda
```

2. **Install dependencies**. We use pytorch 1.13.1 because some packages we'll definetly use aren't compatible with pytorch 2.0.

```
Note: I have CUDA 11.7 so the pytorch version in `requirements.txt` is for CUDA 11.7

pip install -r requirements.txt
```

3. **Install TabPFN locally in editable mode**. This allows us to edit the code in the project folder and have the changes reflected in the installed package (e.g., for retrieving the embeddings at different layers of the encoder).
```
cd TabPFN
pip install -e .
```
