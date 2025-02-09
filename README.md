# Relevance-aware Individual Item Fairness Measures in Recommender Systems ⚖

This repository contains the code for the _extra_ experiments and analyses in our work on "Relevance-aware Individual Item Fairness Measures in Recommender Systems", which is currently under review (single-blind). 
This work extends the SIGIR'24 full paper "Can We Trust Recommender System Fairness Evaluation? The Role of Fairness and Relevance" by Theresia Veronika Rampisela, Tuukka Ruotsalo, Maria Maistro, and Christina Lioma. 
The code for the original experiments is available [here](https://github.com/theresiavr/can-we-trust-recsys-fairness-evaluation).

Links to the SIGIR'24 paper: 
[[ACM]](https://doi.org/10.1145/3626772.3657832) [[arXiv]](https://arxiv.org/abs/2405.18276) 


# License and Terms of Usage
The code is usable under the MIT License. Please note that RecBole may have different terms of usage (see [their page](https://github.com/RUCAIBox/RecBole) for updated information).

# Citation

--TBA--

If you use the code for the relevance-aware (joint) fairness measures in `metrics.py`, please cite the following work:

```BibTeX
@inproceedings{10.1145/3626772.3657832,
author = {Rampisela, Theresia Veronika and Ruotsalo, Tuukka and Maistro, Maria and Lioma, Christina},
title = {Can We Trust Recommender System Fairness Evaluation? The Role of Fairness and Relevance},
year = {2024},
isbn = {9798400704314},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3626772.3657832},
doi = {10.1145/3626772.3657832},,
pages = {271–281},
numpages = {11},
keywords = {fairness and relevance evaluation, recommender systems},
location = {Washington DC, USA},
series = {SIGIR '24}
}
```

If you use the code for the exposure-based fairness measures in `metrics.py` (FairWORel), please cite our the following work and the original papers proposing the measures.

```BibTeX
@article{10.1145/3631943,
author = {Rampisela, Theresia Veronika and Maistro, Maria and Ruotsalo, Tuukka and Lioma, Christina},
title = {Evaluation Measures of Individual Item Fairness for Recommender Systems: A Critical Study},
year = {2024},
issue_date = {June 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {3},
number = {2},
url = {https://doi.org/10.1145/3631943},
doi = {10.1145/3631943},
journal = {ACM Trans. Recomm. Syst.},
month = nov,
articleno = {18},
numpages = {52},
keywords = {Item fairness, individual fairness, fairness measures, evaluation measures, recommender systems}
}
```

# Datasets, Model Training, and Experiments
Please refer to the [code repository of the SIGIR'24 paper](https://github.com/theresiavr/can-we-trust-recsys-fairness-evaluation) to find information on dataset downloads, model training, and the experiment code for the conference paper.

# Hyperparameter Search Space & Best Configuration
The hyperparameter search space per model can be found in  `Recbole/hyperchoice`, and the file `cluster/bestparam.txt` contains the best hyperparameter configurations.
