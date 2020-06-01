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
