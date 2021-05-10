import json
import pandas as pd

veh_pred_visual = json.load(open("./data/test_vehicle_predict.json"))
col_pred_visual = json.load(open("./data/test_color_predict.json"))

veh_pred_text = pd.read_csv("../srl_handler/results/veh_test_fraction.csv").drop('vehicles', 1)
col_pred_text = pd.read_csv("../srl_handler/results/col_test_fraction.csv").drop('colors', 1)
veh_pred_text = veh_pred_text.set_index('query_id').to_dict()["labels"]
col_pred_text = col_pred_text.set_index('query_id').to_dict()["labels"]

# Action constants: vector = [0, 0, 0] : [straight, turn, stop]
test_order_csv = './data/test_order.csv'
query_action_csv = '../srl_handler/results/action_test_fraction.csv'

submit_json_path = "../retrieval_model/results/result.json" 
submit_org = json.load(open(submit_json_path))
submit = json.load(open(submit_json_path))

save_path = "./results/result_refine.json" 
print(f'Refined result will be saved to {save_path}')

STRICT_TURN_V1 = json.load(open("../detector/results/turn_v1.json"))
STRICT_TURN_V2 = json.load(open("../detector/results/turn_v2.json"))

STRICT_TURN = list(set(STRICT_TURN_V1) | set(STRICT_TURN_V2))
STRICT_STOP = json.load(open("../detector/results/stop.json"))
STRICT_STOP_TURN = [i for i in STRICT_TURN if i in STRICT_STOP]

TOP_TO_FREEZE = 5

track2order_map = {}
order2track_map = {}

query_action_map = {}

test_order = pd.read_csv(test_order_csv)

LABEL_MAP = {
    'turn': 0, 'stop': 1, 'straight': 2
}


# SETUP ENVIRONMENT
########################################################

for i, row in test_order.iterrows():
    track_id = row['track_id']
    order = row['order']
    order = int(order)
    track2order_map[track_id] = order
    order2track_map[order] = track_id

########################################################
action_csv = pd.read_csv(query_action_csv)
for i, row in action_csv.iterrows():
    query_id = row['query_id']
    label_vector = eval(row['labels'])
    act_vector = []
    for i, val in enumerate(label_vector):
        if val > 0:
            act_vector.append(i)
    
    query_action_map[query_id] = list(set(act_vector))