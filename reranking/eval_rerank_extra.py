import builtins
from recbole.config import Config
from recbole.evaluator.evaluator import Evaluator

import pickle


import warnings 
warnings.filterwarnings('ignore')

import os
import time

import datetime


def print(*args, **kwargs):
    with open(f'reranking/log/log_evalextra_{dataset}_{model_name}_{rerank}.txt', 'a+') as f:
        return builtins.print(*args, file=f, **kwargs)
    

list_dataset = [
                "Amazon-lb", 
                "Lastfm", 
                "QK-video",
                "ML-10M", 
                ]
list_model = [
    "BPR",
    "MultiVAE",
    "ItemKNN",
    "NCL"
]

list_rerank = ["combmnz"]

for dataset in list_dataset:

    for model_name in list_model:
        for rerank in list_rerank:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(now)

            print(f"Doing {dataset} - {model_name} - {rerank}")


            config = Config(
                    model=model_name, 
                    dataset="new_"+dataset, 
                    config_file_list=["RecBole/recbole/properties/overall.yaml"],

                    config_dict={"topk": [10], 
                                "metrics":[
                                        "FixedIAArerank",
                                        "FixedIIF",
                                        "FixedIFDrerank",
                                    ]})

            evaluator = Evaluator(config)

            list_filename = [f for f in os.listdir("reranking/rerank_struct") if dataset in f and model_name in f and rerank in f and "rerank25" in f and "at10" in f]

            assert len(list_filename) == 1

            with open(f"reranking/rerank_struct/{list_filename[0]}","rb") as f:
                struct = pickle.load(f)
                #add dataset info to the struct, so we can retrieve the unfairest rec configuration for IFD_mul
                struct.set("data.name", dataset)
                start_time = time.time()
                result = evaluator.evaluate(struct)
                print("total time taken: ", time.time() - start_time)
                print(result)

                with open(f"reranking/result_extra/{dataset}_{model_name}_{rerank}_at10_rerank25.pickle","wb") as f:
                    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)