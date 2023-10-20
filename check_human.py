import csv
import os
import numpy as np
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict
def read_csv_sig(file_name):
    res_dict = dict()
    label_dict = dict()
    with open(file_name, "r") as f:
        f_reader = csv.DictReader(f)
        for line in f_reader:
            print(line)
            label_dict[line["guid"]] = line["label"]
            r = line["significant_or_not"].lower()
            if "y" in r:
                res_dict[line["guid"]] = 0
            elif "n" in r:
                res_dict[line["guid"]] = 1
    return res_dict, label_dict

def read_csv_label(file_name):
    res_dict = defaultdict(defaultdict)
    with open(file_name, "r") as f:
        f_reader = csv.DictReader(f)
        for line in f_reader:
            r = line["label"].strip()
            cid = line["guid"].split("-")[0]
            if "0" in r:
                res_dict[cid][line["guid"]] = "wrong"
            elif "1" in r:
                res_dict[cid][line["guid"]] = "right"
    return res_dict

def compare_labels(labeled, grounds):
    res_trans = dict()
    for cid, res_dict in labeled.items():
        truth = []
        for r, res in res_dict.items():
            if res == grounds[r]:
                truth.append(False)
            else:
                truth.append(True)
        if True in truth:
            res_trans[cid] = True
        else:
            res_trans[cid] = False
    return res_trans

ss_dir = "/home/shanshan/icq/prob/data/roc/ss_t_labeled"
roy_dir = "/home/shanshan/icq/prob/data/roc/roy_t_labeled"
#roy_dir = "/home/roy/shanshan_twice_labeled"

features = ["negation", "ner", "word", "tense", "overlap", "sentiment", "pronoun"]
all_ss_res = []
all_roy_res = []
labels_dict = dict()

# significant human-label evaluation
for f in features:
    ss_file = os.path.join(ss_dir, f"{f}.csv")
    roy_file = os.path.join(roy_dir, f"{f}.csv")
    ss_dict, s_labels = read_csv_sig(ss_file)
    for g, l in s_labels.items():
        if g not in labels_dict:
            labels_dict[g] = l
    roy_dict, r_labels = read_csv_sig(roy_file)
    
    ss_res = []
    roy_res = []
    diff_ids = []
    for i, r in ss_dict.items():
        if r != roy_dict[i]:
            diff_ids.append(i)
        ss_res.append(r)
        roy_res.append(roy_dict[i])
    all_ss_res.extend(ss_res)
    all_roy_res.extend(roy_res)
    print(f, len(ss_res))
    print(set(ss_dict.keys())-set(roy_dict.keys()))
    print(set(roy_dict.keys())-set(ss_dict.keys()))
    print(diff_ids)
    print(ss_res)
    print(roy_res)
    print("kappa_score:", cohen_kappa_score(ss_res, roy_res), cohen_kappa_score(roy_res, roy_res))
    print(np.sum(np.array(ss_res)== np.array(roy_res))/float(len(ss_res)))
print(len(all_ss_res), len(all_roy_res))
print("all_kappa_score:", cohen_kappa_score(all_ss_res, all_roy_res), cohen_kappa_score(all_roy_res, all_roy_res))
print(np.sum(np.array(ss_res)== np.array(roy_res))/float(len(ss_res)))

# label human-label evaluation
all_ss_res = []
all_roy_res = []
for f in features:
    ss_file = os.path.join(ss_dir, f"{f}_unk.csv")
    roy_file = os.path.join(roy_dir, f"{f}_unk.csv")
    ss_dict = read_csv_label(ss_file)
    roy_dict  = read_csv_label(roy_file)
    
    ss_res_dict = compare_labels(ss_dict,labels_dict)
    roy_res_dict = compare_labels(roy_dict,labels_dict)
    print(roy_res_dict) 
    print(roy_file) 
    print(ss_res_dict) 
    ss_keys = list(ss_res_dict.keys())
    ss_res = [ss_res_dict[x] for x in ss_keys]
    roy_res = [roy_res_dict[x] for x in ss_keys]
    all_ss_res.extend(ss_res)
    all_roy_res.extend(roy_res)
    print(ss_res)
    print(roy_res)
    print(f, len(ss_res))
    print("kappa_score:", cohen_kappa_score(ss_res, roy_res), cohen_kappa_score(roy_res, roy_res))
    print(np.sum(np.array(ss_res)== np.array(roy_res))/float(len(ss_res)))
print(len(all_ss_res), len(all_roy_res))
print("all_kappa_score:", cohen_kappa_score(all_ss_res, all_roy_res), cohen_kappa_score(all_roy_res, all_roy_res))
print(np.sum(np.array(ss_res)== np.array(roy_res))/float(len(ss_res)))


