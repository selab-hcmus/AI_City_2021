import os.path as osp
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import ast
from constant import *


def apply_refine(submit_dict):
    return refine(submit_dict)

def save_json(data_dict, save_path):
    with open(save_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    print(f'Save result to {save_path}')

def refine(submit):
    for key_query in tqdm(submit):
        top_keys_visual = submit[key_query]
        ans = []
        veh_pred_query = np.array(ast.literal_eval(veh_pred_text[key_query]))
        col_pred_query = np.array(ast.literal_eval(col_pred_text[key_query]))

        query_actions = query_action_map[key_query]
        list_a, list_b, list_c = get_priority_list_by_action(top_keys_visual, query_actions)
        
        list_a = sort_by_att(list_a, veh_pred_text, col_pred_text, key_query, veh_pred_visual, col_pred_visual)
        list_b = sort_by_att(list_b, veh_pred_text, col_pred_text, key_query, veh_pred_visual, col_pred_visual)
        list_c = sort_by_att(list_c, veh_pred_text, col_pred_text, key_query, veh_pred_visual, col_pred_visual)
        final_results = list_a + list_b + list_c

        freeze_list = final_results[:TOP_TO_FREEZE]
        sort_list = final_results[TOP_TO_FREEZE:]
        sort_list = sort_by_att(sort_list, veh_pred_text, col_pred_text, key_query, veh_pred_visual, col_pred_visual)  

        submit[key_query] = freeze_list + sort_list
    
    return submit

def check_label_by_predict(label, chosen_idx):
    for i in chosen_idx:
        if label[i] > 0:
            return True

    return False

def score(label, predict, mode='color'):
    MAX_POINT = 1.5
    MIN_POINT = 0.0
    if mode == 'color':
        MAX_POINT = 1.25
        MIN_POINT = -0.25
    else:
        MAX_POINT = 1.25
        MIN_POINT = -0.2
    # label: from query, predict: classifier from track
    max_label = max(label)
    is_right = False
    
    if max_label == 1:
        idx = np.transpose(np.argwhere(predict>=0.3)).tolist()[0]
        is_right = check_label_by_predict(label, idx)
        if is_right:
            return MAX_POINT
        else:
            return MIN_POINT
    
    idx = np.transpose(np.argwhere(predict>=0.3)).tolist()[0]
    is_right = check_label_by_predict(label, idx)
    if is_right:
        return MAX_POINT
        
    return MIN_POINT

def is_list_in_list(list_src, list2comp):
    for val in list2comp:
        if val not in list_src:
            return False
    return True

def sort_by_att(list_key_visual, veh_pred_text, col_pred_text, key_query, veh_pred_visual, col_pred_visual):
    if len(list_key_visual) == 0:
        return []
    veh_pred_query = np.array(ast.literal_eval(veh_pred_text[key_query]))
    col_pred_query = np.array(ast.literal_eval(col_pred_text[key_query]))

    ans = []
    for key_visual in list_key_visual:
        att_score = score(veh_pred_query, np.array(veh_pred_visual[key_visual][0]), 'vehicle')
        att_score += score(col_pred_query, np.array(col_pred_visual[key_visual][0]), 'color')
        ans.append((key_visual, att_score))
        pass
    
    ans = sorted(ans, key=lambda x: -x[1])
    top_keys_visual = [x[0] for x in ans]
    return top_keys_visual

def get_priority_list_by_action(top_keys_visual, query_actions):
    list_a, list_b, list_c = [], [], []
    
    for key_visual in top_keys_visual:
        order_visual = track2order_map[key_visual]
        
        if is_list_in_list(query_actions, [LABEL_MAP['stop'], LABEL_MAP['turn']]): 
            if (order_visual in STRICT_STOP) and (order_visual in STRICT_TURN):
                list_a.append(key_visual)
                pass
            elif (order_visual in STRICT_STOP) or (order_visual in STRICT_TURN):
                list_b.append(key_visual)
                pass
            else:
                list_c.append(key_visual)
                pass
            
        elif (is_list_in_list(query_actions, [LABEL_MAP['stop']])
            and not is_list_in_list(query_actions, [LABEL_MAP['turn']])
        ):
            if (order_visual in STRICT_STOP):
                list_a.append(key_visual)
                pass
            else:
                list_b.append(key_visual)
                pass

        elif (is_list_in_list(query_actions, [LABEL_MAP['turn']])
            and not is_list_in_list(query_actions, [LABEL_MAP['stop']])
        ):
            if (order_visual in STRICT_TURN):
                list_a.append(key_visual)
                pass
            else:
                list_b.append(key_visual)
                pass

        else:
            if (order_visual not in STRICT_TURN) and (order_visual not in STRICT_STOP):
                list_a.append(key_visual)
            else:
                list_b.append(key_visual)

    return list_a, list_b, list_c