import builtins
from recbole.config import Config
from recbole.evaluator.evaluator import Evaluator

import pickle
import warnings 
warnings.filterwarnings('ignore')

import os
import time


import datetime

path = "top_k_insensitivity"

list_dataset = [
                "Amazon-lb", 
                "Lastfm", 
                "QK-video",
                "ML-10M", 
                ]

model_name = "NCL"
list_k = [1, 3, 5, 10, 20]

def print(*args, **kwargs):
    with open(f'experiments/{path}/log_{dataset}.txt', 'a+') as f:
        return builtins.print(*args, file=f, **kwargs)
    

for dataset in list_dataset:
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(now)

    config = Config(
            model=model_name, 
            dataset="new_"+dataset, 
            config_file_list=["RecBole/recbole/properties/overall.yaml"],

            config_dict={"topk": list_k, 
                        "metrics":[
                                    "FixedIFDdiv", #IFD div ori and IFD div our
                            ]})

    evaluator = Evaluator(config)

    list_filename = [f for f in os.listdir("cluster/best_struct") if dataset in f and model_name in f]

    assert len(list_filename) == 1

    with open(f"cluster/best_struct/{list_filename[0]}","rb") as f:
        struct = pickle.load(f)

    #checked metric need: only pos items need to be updated, and full_rec_mat to be inserted

    pred_rel = struct.get('rec.score')
    num_item = struct.get("data.num_items") -1
    num_user = pred_rel.shape[0]

    full_rec_mat = pred_rel[:,1:]\
                                .sort(descending=True, stable=True)\
                                .indices + 1 #first column is dummy

    assert full_rec_mat.shape == (num_user, num_item)
    struct.set("rec.all_items", full_rec_mat)

    start_time = time.time()
    result = evaluator.evaluate(struct)
    print("total time taken: ", time.time() - start_time)
    print(result)

    with open(f"experiments/{path}/result_{dataset}.pickle","wb") as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)