import os
import sys
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import OrderedDict

#f = open("results_all_20240806_125634.txt")
#ff = open("results_all_20240807_124308.txt")
#f = open("results_all_20240807_175409.txt")
#f = open("results_all_20240807_233646.txt")
#f = open("results_all_20240808_073803.txt")
#f = open("results_all_20240812_223502.txt")
#f = open("results_all_20240813_074648.txt")
#f = open("results_all_20240813_104719.txt")
#f = open("results_all_20240813_121344.txt")
#f = open("results_20240813_215229.txt")
#f = open("results_20240813_222211.txt")
#f = open("results_all_20240820_195007.txt")
#f = open("results_all_20240820_225747.txt")
#f = open("results_all_20240821_145328.txt")
f = open("results_all_20240822_003339.txt")
test_results = []

for line in f:
    tokens = line.split('|')
    if (len(tokens) <= 1):
        continue
    test_result = OrderedDict()
    for token in tokens:
        name = token.split(":")[0].rstrip(" \n").lstrip(" ")
        try:
            value = token.split(":")[1].rstrip(" \n").lstrip(" ")
        except:
            if ("tier" in token):
                test_result["tier"] = token
            continue
        test_result[name]=value
    print (test_result)
    if (test_result["SSD_K"] == "0" and test_result["Network_K"] == "0"):
        test_result["Parity Type"] = "No_Parity"
    elif (test_result["SSD_K"] == "0" and test_result["Network_K"] != "0"):
        test_result["Parity Type"] = "Network_Parity_Only"
    elif (test_result["SSD_K"] != "0" and test_result["Network_K"] == "0"):
        test_result["Parity Type"] = "SSD_Parity_Only"
    else:
        test_result["Parity Type"] = "Both"

    test_result["Effective Capacity"] = str(float(test_result["Network_M"]) / (int(test_result["Network_M"]) + int(test_result["Network_K"])) * float(test_result["SSD_M"]) / (int(test_result["SSD_M"]) + int(test_result["SSD_K"])))
    test_results.append(test_result)
    
 
f_write = open("result.parse", "w")
idx = 0
data = {}
data['x'] = []
data['y'] = []
data['category'] = []
data['label'] = []
for test_result in test_results:
    sorted_dict = test_result
    if idx == 0:
        for name in sorted_dict:
            f_write.write(name + " | ")
            #print (test_result)
        f_write.write("\n")
    for name in sorted_dict:
        f_write.write(sorted_dict[name] + " | ")
        if not "3tier" in sorted_dict["tier"]:
            continue
        #if not "Network" in sorted_dict["Parity Type"]:
        #    continue
        if True:
            if (name == "Parity Type"):
                data['category'].append(sorted_dict[name])
            if (name == "Effective Capacity"):
                data['x'].append(float(sorted_dict[name]))
            if (name == "Availability"):
            #if (name == "Effective Availability"):
                data['y'].append(-math.log10(1-float(sorted_dict[name])))    
                data['label'].append(sorted_dict["Network_M"]+"_"+sorted_dict["Network_K"]+"_"+sorted_dict["SSD_M"]+"_"+sorted_dict["SSD_K"])
    f_write.write("\n")
    idx += 1


df = pd.DataFrame(data)

categories = df['category'].unique()
colors = ['red', 'blue', 'green', 'purple']
#colors = ['red']
color_dict = dict(zip(categories, colors))
print (categories, color_dict)

df['color'] = df['category'].apply(lambda x: color_dict[x])

# 분산 그래프 그리기
plt.figure(figsize=(10, 6))

for category in categories:
    subset = df[df['category'] == category]
    plt.scatter(subset['x'], subset['y'], c=subset['color'], label=category)
    for i in range(subset.shape[0]):
        plt.text(subset['x'].iloc[i], subset['y'].iloc[i], subset['label'].iloc[i], fontsize=5, ha='right')

plt.xlim(0, max(df['x']) + 0.2)
plt.ylim(0, max(df['y']) + 1)
plt.xlabel('X values : Effective Capacity')
#plt.ylabel('Y values : Availability')
plt.ylabel('Y values : Effective Availability')
plt.title('Scatter Plot with Different Colors by Category')
plt.legend(title='Category')
plt.grid(True)
plt.show()
