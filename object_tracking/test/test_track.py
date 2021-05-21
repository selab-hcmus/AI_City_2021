import os, sys, pickle, json
import os.path as osp 

result_dir = './results/annotate'
sample_file = os.listdir(result_dir)[1]
track_dict = pickle.load(open(osp.join(result_dir, sample_file), 'rb'))

list_keys = list(track_dict.keys())

for k in list_keys[:8]:
    print(track_dict[k])
    
