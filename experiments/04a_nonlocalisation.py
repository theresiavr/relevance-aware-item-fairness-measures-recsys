import builtins
from recbole.config import Config
from recbole.evaluator.evaluator import Evaluator

import pickle
import warnings 
warnings.filterwarnings('ignore')

import os
import time

import numpy as np

import datetime

path = "nonlocal"

list_exp_type = [
            "front", 
            "back"
            ]

list_dataset = [
                "Amazon-lb", 
                "Lastfm", 
                "QK-video",
                "ML-10M", 
                ]

model_name = "NCL"
k = 10
max_missing_rel = 10

def print(*args, **kwargs):
    with open(f'experiments/{path}/log_{dataset}.txt', 'a+') as f:
        return builtins.print(*args, file=f, **kwargs)
    



for exp_type in list_exp_type:
    for dataset in list_dataset:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(now)

        print(f"Doing {dataset}")
        if exp_type == "back":
            print("Placing the missing relevant item all the way at the back...")

        config = Config(
                model=model_name, 
                dataset="new_"+dataset, 
                config_file_list=["RecBole/recbole/properties/overall.yaml"],

                config_dict={"topk": k, 
                            "metrics":[
                                        "IBO_IWO", #ori and our
                                        "IAArerank",
                                        "FixedIAA", 
                                        "FixedIFDdiv", #IFD div ori and IFD div our
                                        "HD",
                                        "IIF", 
                                        "FixedIIF",
                                        "AIF"
                                ]})

        evaluator = Evaluator(config)

        list_filename = [f for f in os.listdir("cluster/best_struct") if dataset in f and model_name in f]

        assert len(list_filename) == 1

        with open(f"cluster/best_struct/{list_filename[0]}","rb") as f:
            struct = pickle.load(f)

        #already checked metric need, only pos items need to be updated, and full_rec_mat to be inserted

        pred_rel = struct.get('rec.score')
        pos_items = struct.get("data.pos_items")
        num_item = struct.get("data.num_items") -1
        num_user = pred_rel.shape[0]

        full_rec_mat = pred_rel[:,1:]\
                                    .sort(descending=True, stable=True)\
                                    .indices + 1 #first column is dummy

        assert full_rec_mat.shape == (num_user, num_item)
        struct.set("rec.all_items", full_rec_mat)

        item_at_k_plus = full_rec_mat[:,k:]
        user_item_rel = np.array([np.in1d(item_at_k_plus[u], pos_items[u], assume_unique=True) for u in range(pos_items.shape[0])], dtype=int) 
        candidate_item = np.where(user_item_rel==0, item_at_k_plus, -1)
        if exp_type == "back":
            dataset = "back_" + dataset
            candidate_item = candidate_item[:, ::-1] #placing the missing rel items at the back = reversing the matrix columns with ::-1
        incoming_extra_rel_item = np.asarray([cand_item_u[cand_item_u!=-1][:max_missing_rel] for cand_item_u in candidate_item])

        for missing_rel in range(1, max_missing_rel+1):
            print(f"Doing {dataset}, missing {missing_rel} relevance label(s)")
            
            updated_pos_items = np.array([np.concatenate([pos_items[u], incoming_extra_rel_item[u][:missing_rel]]) for u in range(pos_items.shape[0])])

            #update pos_items
            struct.set("data.pos_items", updated_pos_items)
            
            start_time = time.time()
            result = evaluator.evaluate(struct)
            print("total time taken: ", time.time() - start_time)
            print(result)

            with open(f"experiments/{path}/result_{dataset}_missing_{missing_rel}.pickle","wb") as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)