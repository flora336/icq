import csv
import os
from collections import defaultdict
import random

def read_csv(f):
    lines = defaultdict(list)
    feature = "_".join(f[:-4].split("_")[1:3])

    with open(f, "r") as fr:
        freader = csv.DictReader(fr)
        for line in freader:
            d = {}
            d["guid"] = line["guid"]
            d["premise"] = line["premise"]
            d["hypothesis"] = line["chypothesis"]
            d["label"] = line["label"]
            d["feature"] = feature
            lines[line["guid"].split("-")[0]].append(d)
    return lines

def read_labeled_csv(f):
    lines = defaultdict(list)
    print(f)
    with open(f, "r") as fr:
        freader = csv.DictReader(fr)
        for line in freader:
            d = {}
            d["guid"] = line["guid"]
            d["premise"] = line["premise"]
            d["hypothesis"] = line["hypothesis"]
            d["label"] = line["label"]
            d["feature"] = line["feature"]
            lines[line["guid"].split("-")[0]].append(d)
    return lines


if __name__ == "__main__":
    data_dir = "/home/shanshan/icq/prob/data/roc/ss_labeled/"
    substitute_dir = "/home/shanshan/icq/prob/log/roc/dataf/substitute/train/"
    out_dir = "/home/shanshan/icq/prob/data/roc/labeled_substitute/"
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
    fs = os.listdir(data_dir)
    #sub_fs = os.listdir(substitute_dir)
    #sub_lines = defaultdict(defaultdict)
    cid_features = defaultdict(list)
    
    subs_dict = defaultdict(defaultdict)
    subs_fs = os.listdir(substitute_dir)
    for s in subs_fs:
        f_type = s.split("_")[1]
        f_sub_type = s.split("_")[2].split(".csv")[0]
        print("f_sub_type:", f_sub_type)
        f = os.path.join(substitute_dir, s)
        res_lines = read_csv(f)
        for cid, guid_lines in res_lines.items():
            subs_dict[f"{f_type}_{f_sub_type}"][cid] = guid_lines
    #print("word_not:", subs_dict["word_not"])
    for fn in fs:
        count = 0
        print(fn)
        cases_ids = defaultdict(list)
        f_type = fn.split(".csv")[0]
    
        f = os.path.join(data_dir, fn)
        res_lines = read_labeled_csv(f)
        for cid, guid_lines in res_lines.items():
            for guid_line in guid_lines:
                f_subtype = guid_line["feature"].split("_")[1]
                cases_ids[f_subtype].append(f"{f_subtype}-{cid}")
        print(cases_ids)
        out_file = os.path.join(out_dir, f"{f_type}.csv")
        with open(out_file, "w") as f_w:
            f_writer = csv.DictWriter(f_w, fieldnames=["feature", "guid", "premise", "hypothesis", "label", "significant_or_not"])
            f_writer.writeheader()
            for subword, subword_cids in cases_ids.items():
                print("subword:", subword)
                for subword_cid in subword_cids:
                    cid = subword_cid.split("-")[1]
                    print("cid:", cid)
                    count+=1
                    sub_word = subword_cid.split("-")[0]
                    if f"{f_type}_{sub_word}" in subs_dict:
                        if cid in subs_dict[f"{f_type}_{sub_word}"]:
                            for l in subs_dict[f"{f_type}_{sub_word}"][cid]:
                                #print("l:", l)
                                w = {}
                                w["feature"] = l["feature"] 
                                w["guid"] = l["guid"] 
                                w["premise"] = l["premise"] 
                                w["hypothesis"] = l["hypothesis"] 
                                w["label"] = l["label"] 
                                w["significant_or_not"] = ""
                                f_writer.writerow(w)
                        else:
                            print(f"cid:{cid} if not in subs_dict")
                    else:
                        print(f"feature is not exist in subs_dict")
        print(count)

        

