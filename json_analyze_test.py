# analyzes json files

import pandas as pd
import json
from scipy.stats import entropy
from scipy.spatial import distance
import jsonlines
import numpy as np
import pickle
from scipy.stats import chisquare
from scipy.stats import kstest


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


def get_top_k_count_from_file(json_file, top1 = 10, top2 = 100, top3 = 1000):
    # takes in the json file and returns the counts of top1,2,3,4
    # top4 is just whatever is past the last number, for example >1000
    # returns list in order of top1 to top4 bins
    return get_top_k_count(json_file["result"]["real_topk"], top1, top2, top3)


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


def fracp_bin_counter_from_file(json_file):
    # takes json file (json structure) and returns bins count
    rtk = json_file["result"]["real_topk"]
    ptk = json_file["result"]["pred_topk"]
    return fracp_bin_counter(get_frac_p(rtk, ptk))


def zero_to_small_num(lst):
    # takes a list and replaces all 0 with a small number
    for i in range(len(lst)):
        if lst[i] == 0:
            lst[i] = 0.0000000000000000000001
    return lst


def get_kld(fp_bin1, fp_bin2):
    # given two list of bin counts (10 long by default)
    # returns KLD value
    return entropy(fp_bin1, fp_bin2)


def get_kld_from_json_file(file1, file2):
    # given two json objects
    # returns KLD value
    # this skips a lot of steps to make it easier
    realtk_1 = file1["result"]["real_topk"]
    predtk_1 = file1["result"]["pred_topk"]
    realtk_2 = file2["result"]["real_topk"]
    predtk_2 = file2["result"]["pred_topk"]

    bins1 = fracp_bin_counter(get_frac_p(realtk_1, predtk_1))
    bins2 = fracp_bin_counter(get_frac_p(realtk_2, predtk_2))
    print(str(bins1) + "                   " + str(bins2))
    # bins1 = zero_to_small_num(bins1)
    # bins2 = zero_to_small_num(bins2)
    print(str(bins1) + "                   " + str(bins2))
    return get_kld(bins1, bins2)


def get_jsd(fp_bin1, fp_bin2):
    # given two list of bin counts, (10 long by default)
    # returns JSD value
    return distance.jensenshannon(fp_bin1, fp_bin2)


def compare_json_files_kld(filename1, filename2):
    # given two file names, get json from it, then use kld
    # returns list of all kld values
    lst = []
    with open(filename1) as f1:
        d1 = json.load(f1)
    with open(filename2) as f2:
        d2 = json.load(f2)
    print(str(len(d1))+"       F2:"+str(len(d2)))

    for d1x in d1:
        for d2x in d2:
            # print("D1: " + str(d1x) + "           D2: " + str(d2x))
            lst.append(get_kld_from_json_file(d1x, d2x))
            print(lst[-1])
    return lst


def list_of_fracp_from_file(filename):
    # given two file names, get json from it, then return list
    # returns list of list of 10 frac p bins
    lst = []
    with open(filename) as f1:
        d1 = json.load(f1)

    for d1x in d1:
        lst.append(fracp_bin_counter_from_file(d1x))
    return lst


def list_of_fracp_from_jsonl_file(filename):
    # given two file names of json lines, get json from it, then return list
    # returns list of list of 10 frac p bins
    lst = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            lst.append(fracp_bin_counter_from_file(obj))
    return lst


def list_of_norm_fracp_from_file(filename):
    # given two file names, get json from it, then return list
    # returns list of list of 10 frac p bins that are normalized
    lst = []
    with open(filename) as f1:
        d1 = json.load(f1)

    for d1x in d1:
        bins = fracp_bin_counter_from_file(d1x)
        tot = sum(bins)
        for i in range(10):
            bins[i] = bins[i] / tot
        print(sum(bins))
        lst.append(bins)
    return lst


## open analyzed json to test
#with open("gpt2.analyzed.webtext-10.json") as f:
 #   data = json.load(f)
'''
x = 0
print(type(data))
print(len(data))

for item in data:
    print(x)
    print(get_top_k_count_from_file(item))
    print(fracp_bin_counter_from_file(item))
    x = x + 1'''

## open two json hGPT2 and mGPT2 and kld and then save as pickle
# kld_lst = compare_json_files_kld("gpt2.analyzed.webtext-1000.json", "gpt2.analyzed.medk40train-1000.json")
# pickle.dump(kld_lst, open("hGPT2mGPT2-1000-list-original.pickle", "wb"))
# print(len(kld_lst))
# print(kld_lst)
# kld_df = pd.DataFrame(kld_lst)
# kld_df.to_pickle("hGPT2mGPT2-1000-pd-original.pickle")
# kld_df = pd.DataFrame(kld_lst).replace([np.inf, -np.inf], np.nan).dropna()
# kld_df.to_pickle("hGPT2mGPT2-1000-pd-no_infinity.pickle")
# print(kld_df)
# print(kld_df.describe())


## unpickle gtp2gpt2
# unpickle_original_pd = pd.read_pickle("hGPT2mGPT2-100-pd-original.pickle")
# print(unpickle_original_pd)
# print(unpickle_original_pd.describe())
#
# unpickle_noinf_pd = pd.read_pickle("hGPT2mGPT2-100-pd-no_infinity.pickle")
# print(unpickle_noinf_pd)
# print(unpickle_noinf_pd.describe())
#
# ori = pickle.load(open("hGPT2mGPT2-100-list-original.pickle", "rb"))
# print(ori)
# print(len(ori))

# # open hgpt2 and mgpt3 and save as pickle
# kld_lst = compare_json_files_kld("gpt2.analyzed.webtext-1000.json", "gpt3.analyzed.machine-485.json")
# pickle.dump(kld_lst, open("hGPT2mGPT3-1000-list-original.pickle", "wb"))
# print(len(kld_lst))
# print(kld_lst)
# kld_df = pd.DataFrame(kld_lst)
# kld_df.to_pickle("hGPT2mGPT3-1000-pd-original.pickle")
# kld_df = pd.DataFrame(kld_lst).replace([np.inf, -np.inf], np.nan).dropna()
# kld_df.to_pickle("hGPT2mGPT3-1000-pd-no_infinity.pickle")
# print(kld_df)
# print(kld_df.describe())


# # open gpt2 and grover and save as pickle
# kld_lst = compare_json_files_kld("gpt2.analyzed.medk40train-1000.json", "grover.analyzed.machine-1000.json")
# pickle.dump(kld_lst, open("mGPT2mGROVER-1000-list-original.pickle", "wb"))
# print(len(kld_lst))
# kld_df = pd.DataFrame(kld_lst)
# print(kld_df.describe())
# kld_df.to_pickle("mGPT2mGROVER-1000-pd-original.pickle")
# kld_df = pd.DataFrame(kld_lst).replace([np.inf, -np.inf], np.nan).dropna()
# kld_df.to_pickle("mGPT2mGROVER-1000-pd-no_infinity.pickle")
# print(kld_df.describe())


# # unpickle gtp2gpt2
# unpickle_original_pd = pd.read_pickle("mGPT2mGROVER-1000-pd-original.pickle")
# print(unpickle_original_pd)
# print(unpickle_original_pd.describe())
#
# unpickle_noinf_pd = pd.read_pickle("mGPT2mGROVER-1000-pd-no_infinity.pickle")
# print(unpickle_noinf_pd)
# print(unpickle_noinf_pd.describe())
#
# ori = pickle.load(open("mGPT2mGROVER-1000-list-original.pickle", "rb"))
# #print(ori)
# print(len(ori))

''''# get frac p for each text analyzed
res = list_of_fracp_from_file("grover.analyzed.machine-5000.json")
pickle.dump(res, open("fracp.GROVER-machine-5000-lst-notNorm.pickle", "wb"))
df = pd.DataFrame(res)
df = df.div(df.sum(axis=1), axis=0)
# x = (df.sum(axis=1)).to_frame()
df.to_pickle("fracp.GROVER-machine-5000-pd-normalized.pickle")
df.to_csv("fracp.GROVER-machine-5000-normalized.csv")
print(df)
#des = df.describe()
#des.to_csv("describetest.csv")
print(df.describe())

unpick = pd.read_pickle("fracp.GROVER-machine-5000-pd-normalized.pickle")
print(unpick)
des = unpick.describe()
des.to_csv("fracp.GROVER-machine-5000-pd-normalized-describe.csv")'''


# # get frac p for each text analyzed from jsonlines
# res = list_of_fracp_from_jsonl_file("grover.analyzed.machine-10000.jsonl")
# pickle.dump(res, open("fracp.GROVER-machine-10000-lst-notNorm.pickle", "wb"))
# df = pd.DataFrame(res)
# df = df.div(df.sum(axis=1), axis=0)
# # x = (df.sum(axis=1)).to_frame()
# df.to_pickle("fracp.GROVER-machine-10000-pd-normalized.pickle")
# df.to_csv("fracp.GROVER-machine-10000-normalized.csv")
# print(df)
# #des = df.describe()
# #des.to_csv("describetest.csv")
# print(df.describe())
#
# unpick = pd.read_pickle("fracp.GROVER-machine-10000-pd-normalized.pickle")
# print(unpick)
# des = unpick.describe()
# des.to_csv("fracp.GROVER-machine-10000-pd-normalized-describe.csv")


# # Anderson-Darling Test
# from numpy.random import seed
# from numpy.random import randn
# from scipy.stats import anderson
# # seed the random number generator
# seed(1)
# # generate univariate observations
# data = 5 * randn(100) + 50
# # normality test
# result = anderson(data)
# print('Statistic: %.3f' % result.statistic)
# p = 0
# for i in range(len(result.critical_values)):
# 	sl, cv = result.significance_level[i], result.critical_values[i]
# 	if result.statistic < result.critical_values[i]:
# 		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
# 	else:
# 		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


# chi squared gpt2 machine vs human
observed = pd.read_pickle("fracp.GPT2-machine-25000-pd-normalized.pickle").mean(axis=0)
expected = pd.read_pickle("fracp.GPT2-human-25000-pd-normalized.pickle").replace(0, 0.00000000000000000000000000000001).mean(axis=0)
# expected = pd.read_pickle("fracp.GPT2-human-25000-pd-normalized.pickle")[:-3].replace(0, 0.00000000000000000000000000000001)
print(expected.shape)
chi_res = chisquare(observed, expected, 0, 0)
print(type(expected))
print(chi_res)
ks = kstest(observed, expected)
print(ks)
