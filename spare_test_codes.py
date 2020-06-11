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
#realdf.to_csv("real.csv")

print(realdf.describe())

lst = []

for item in file["result"]["pred_topk"]:
    lst.append(item[0][1])

'''for L in lst:
    f = open("top_pred.txt", "w")
    f.writelines(str(L)+'\n')'''


top_k_count = get_top_k_count(realtk)
frac_p = get_frac_p(realtk, predtk)
frac_p_bins = fracp_bin_counter(frac_p)
print(frac_p_bins)
kld = entropy(frac_p_bins, frac_p_bins)
print(kld)
jsd = distance.jensenshannon(frac_p_bins, frac_p_bins)
print(jsd)



###################
with open('test1.json') as json_file1:
    file1 = json.load(json_file1)

with open('test2.json') as json_file2:
    file2 = json.load(json_file2)

with open('test3.json') as json_file3:
    file3 = json.load(json_file3)

realtk1 = file1["result"]["real_topk"]
predtk1 = file1["result"]["pred_topk"]
realtk2 = file2["result"]["real_topk"]
predtk2 = file2["result"]["pred_topk"]
realtk3 = file3["result"]["real_topk"]
predtk3 = file3["result"]["pred_topk"]

a = fracp_bin_counter(get_frac_p(realtk1, predtk1))
b = fracp_bin_counter(get_frac_p(realtk2, predtk2))
c = fracp_bin_counter(get_frac_p(realtk3, predtk3))

print(get_kld(a, b))
print(get_kld(b, c))

#################
with open('test1.json') as json_file1:
    file1 = json.load(json_file1)

with open('test2.json') as json_file2:
    file2 = json.load(json_file2)

with open('test3.json') as json_file3:
    file3 = json.load(json_file3)

print(get_kld_from_json_file(file1, file2))
print(get_kld_from_json_file(file2, file3))



##############
output = {}

with jsonlines.open('gpt-2.medium-345M-k40.train.jsonl') as reader:
    for obj in reader:
        print(str(obj["id"]))
        raw_text = obj["text"]
        raw_text = remove_symbols_from_text(raw_text)
        # print(raw_text + "\n\n" + str(obj["id"]))
        payload = lm.check_probabilities(raw_text, topk=20)
        res = {
            "request": {'project': "new", 'text': raw_text},
            "result": payload
        }
        output = output.append(res)
        dict = {
            "id" : obj["id"],
            'res': res
        }
        if obj["id"] == 1000:
            break

with open('gpt2.analyzed.medk40train.json', 'w') as outfile:
    json.dump(output, outfile)


############

'''
x = 0
print(type(data))
print(len(data))

for item in data:
    print(x)
    print(get_top_k_count_from_file(item))
    print(fracp_bin_counter_from_file(item))
    x = x + 1'''

##############################

"""
kld_lst = compare_json_files_kld("gpt2.analyzed.webtext-100.json", "gpt2.analyzed.medk40train-100.json")
pickle.dump(kld_lst, open("hGPT2mGPT2-100-list-original.pickle", "wb"))
print(len(kld_lst))
print(kld_lst)
kld_df = pd.DataFrame(kld_lst)
kld_df.to_pickle("hGPT2mGPT2-100-pd-original.pickle")
kld_df = pd.DataFrame(kld_lst).replace([np.inf, -np.inf], np.nan).dropna()
kld_df.to_pickle("hGPT2mGPT2-100-pd-no_infinity.pickle")
print(kld_df)
print(kld_df.describe())"""


###########
unpickle_original_pd = pd.read_pickle("hGPT2mGPT2-100-pd-original.pickle")
print(unpickle_original_pd)
print(unpickle_original_pd.describe())

unpickle_noinf_pd = pd.read_pickle("hGPT2mGPT2-100-pd-no_infinity.pickle")
print(unpickle_noinf_pd)
print(unpickle_noinf_pd.describe())

ori = pickle.load(open("hGPT2mGPT2-100-list-original.pickle", "rb"))
print(ori)
print(len(ori))


###################################

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

res = list_of_norm_fracp_from_file("gpt2.analyzed.medk40train-1000.json")
pickle.dump(res, open("fracp.GPT2-machine-1000-lst-normalized.pickle", "wb"))
df = pd.DataFrame(res)
df.to_pickle("fracp.GPT2-machine-1000-pd-normalized.pickle")
print(df)
print(df.describe())



###################################################

