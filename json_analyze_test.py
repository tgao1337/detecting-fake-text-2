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

print(file)

df = pd.read_json('test_json.json')
print(df)
