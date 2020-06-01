# analyzes json files

import pandas as pd
import json

'''
df = pd.read_csv('titanic.csv')
print(df)

df = df.drop(['Name', 'Siblings/Spouses Aboard', 'Parents/Children Aboard'], axis=1)
print(df.head(8))

print(df['Survived'][4])
'''

with open('test_json.json') as json_file:
    file = json.load(json_file)

realtk = file["result"]["real_topk"]
predtk = file["result"]["pred_topk"]
print(realtk)

realdf = pd.DataFrame(realtk)
print(realdf)

preddf = pd.DataFrame(predtk)
print(preddf)

print(preddf[19][0][1])
