from utils import dict_load


old_stop_json = '/home/ntphat/projects/AI_City_2021/detector/results/stop.json'
new_stop_json = '/home/ntphat/projects/AI_City_2021/results/detector/stop.json'

list_old_stop = dict_load(old_stop_json)
list_new_stop = dict_load(new_stop_json)['stop_ids']
list_new_stop = [int(i) for i in list_new_stop]

old_not_new = list(set(list_old_stop) - set(list_new_stop))
new_not_old = list(set(list_new_stop) - set(list_old_stop))

print(f'old not new, {len(old_not_new)}: {old_not_new}')
print(f'new not old, {len(new_not_old)}: {sorted(new_not_old)}')
