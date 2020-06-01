# analyzes json files

import pandas as pd
import json
from scipy.stats import entropy
from scipy.spatial import distance


def get_top_k_count(real_topk, top1 = 10, top2 = 100, top3 = 1000):
    # takes in the json part for real_topk and returns the counts of top1,2,3,4
    # top4 is just whatever is past the last number, for example >1000
    # returns list in order of top1 to top4 bins
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
    # takes in real_topk and pred_topk and returns list of
    # frac(p)
    res = []
    for i in range(len(real_topk)):
        res.append(real_topk[i][1] / pred_topk[i][0][1])
    return res


def fracp_bin_counter(fracp):
    # takes in the list of all frac(p) and returns list of buckets from 0-1
    # counting by 0.1
    b0 = 0
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0
    b5 = 0
    b6 = 0
    b7 = 0
    b8 = 0
    b9 = 0

    for val in fracp:
        if(val <= 0.1):
            b0 = b0 + 1
        elif(val <= 0.2):
            b1 = b1 + 1
        elif(val <= 0.3):
            b2 = b2 + 1
        elif (val <= 0.4):
            b3 = b3 + 1
        elif (val <= 0.5):
            b4 = b4 + 1
        elif (val <= 0.6):
            b5 = b5 + 1
        elif (val <= 0.7):
            b6 = b6 + 1
        elif (val <= 0.8):
            b7 = b7 + 1
        elif (val <= 0.9):
            b8 = b8 + 1
        else:
            b9 = b9 + 1
    # print([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    return [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9]


def get_kld(fp_bin1, fp_bin2):
    return entropy(fp_bin1, fp_bin2)


def get_kld_from_json_file(file1, file2):
    realtk_1 = file1["result"]["real_topk"]
    predtk_1 = file1["result"]["pred_topk"]
    realtk_2 = file2["result"]["real_topk"]
    predtk_2 = file2["result"]["pred_topk"]
    return get_kld(fracp_bin_counter(get_frac_p(realtk_1, predtk_1)), fracp_bin_counter(get_frac_p(realtk_2, predtk_2)))


def get_jsd(fp_bin1, fp_bin2):
    return distance.jensenshannon(fp_bin1, fp_bin2)


with open('test1.json') as json_file1:
    file1 = json.load(json_file1)

with open('test2.json') as json_file2:
    file2 = json.load(json_file2)

with open('test3.json') as json_file3:
    file3 = json.load(json_file3)

print(get_kld_from_json_file(file1, file2))
print(get_kld_from_json_file(file2, file3))



