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


def get_top_k_count(real_topk, top1 = 10, top2 = 100, top3 = 1000):
    # takes in the json part for real_topk and returns the counts of top1,2,3,4
    # top4 is just whatever is past the last number, for example >1000
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    for item in real_topk:
        if(item[0] < top1):
            t1 = t1 + 1
        elif(item[0] < top2):
            t2 = t2 + 1
        elif(item[0] < top3):
            t3 = t3 + 1
        else:
            t4 = t4 + 1
    return [t1, t2, t3, t4]


def get_frac_p(real_topk, pred_topk):
    res = []
    for i in range(len(real_topk)):
        res.append(real_topk[i][1] / pred_topk[i][0][1])
    return res


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
#realdf.to_csv("real.csv")

print(realdf.describe())

lst = []

for item in file["result"]["pred_topk"]:
    lst.append(item[0][1])

'''for L in lst:
    f = open("top_pred.txt", "w")
    f.writelines(str(L)+'\n')'''

print(get_top_k_count(realtk))
print(get_frac_p(realtk, predtk))



