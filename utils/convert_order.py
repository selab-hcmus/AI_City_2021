import json
import pandas as pd

# change path
train_tracks = json.load(open("/content/train-tracks.json"))
test_tracks = json.load(open("/content/test-tracks.json"))
test_queries = json.load(open("/content/test-queries.json"))

count = 0
ans = []
for key in train_tracks:
    ans.append({
        "key": key,
        "order": count
    })
    count += 1
train_tracks_df = pd.DataFrame(data=ans)


count = 0
ans = []
for key in test_tracks:
    ans.append({
        "key": key,
        "order": count
    })
    count += 1
test_tracks_df = pd.DataFrame(data=ans)


count = 0
ans = []
for key in test_queries:
    ans.append({
        "key": key,
        "order": count
    })
    count += 1
test_queries_df = pd.DataFrame(data=ans)

# change path
train_tracks_df.to_csv("/content/AI_City_2021/utils/train_tracks_order.csv", index=False)
test_tracks_df.to_csv("/content/AI_City_2021/utils/test_tracks_order.csv", index=False)
test_queries_df.to_csv("/content/AI_City_2021/utils/test_queries_order.csv", index=False)