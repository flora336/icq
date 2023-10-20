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
            d["hypothesis"] = line["hypothesis"]
            d["label"] = line["label"]
            d["feature"] = feature
            lines[line["guid"].split("-")[0]].append(d)
    return lines
if __name__ == "__main__":
    data_dir = "/home/shanshan/icq/prob/log/snli/dataf/train/"
    substitute_dir = "/home/shanshan/icq/prob/log/roc/dataf/substitute/train/"
    out_dir = "/home/shanshan/icq/prob/data/roc/twice_labeled/"
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
    fs = os.listdir(data_dir)
    sub_fs = os.listdir(substitute_dir)
    lines = defaultdict(defaultdict)
    sub_lines = defaultdict(defaultdict)
    cases_ids = defaultdict(list)
    cid_features = defaultdict(list)
    
    for fn in fs:
        f_type = fn.split("_")[1]
        f_subtype = fn.split("_")[2]
    
        f = os.path.join(data_dir, fn)
        res_lines = read_csv(f)
        for cid, guid_lines in res_lines.items():
            if f_type == "word":
                cid_features[cid].append(f_subtype)
            lines[f_type][f"{f_subtype}-{cid}"] = guid_lines
            cases_ids[f_type].append(f"{f_subtype}-{cid}")

    '''for sub_fn in sub_fs:
        f_type = sub_fn.split("_")[1]
        f_subtype = sub_fn.split("_")[2]

        f = os.path.join(substitute_dir, sub_fn)
        res_lines = read_csv(f)
        for cid, guid_lines in res_lines.items():
            if f_type == "word":
                cid_features[cid].append(f_subtype)
            lines[f_type][f"{f_subtype}-{cid}"] = guid_lines
            cases_ids[f_type].append(f"{f_subtype}-{cid}") '''
    
    question_num = 5
    seed = 10
    random.seed(10) 
    w_ids = []
    for ft, cids in cases_ids.items():
        if len(cids) > question_num:
            chooses = list(random.sample(cids, question_num))
        else:
            chooses = cids
        # if type is word, I will find all posible cues for a question
        if ft == "word":
            out_file = os.path.join(out_dir, f"{ft}.csv")
            with open(out_file, "w") as f_w:
                f_writer = csv.DictWriter(f_w, fieldnames=["feature", "guid", "premise", "hypothesis", "label", "significant_or_not"])
                f_writer.writeheader()
                for subword_cid in chooses:
                    cid = subword_cid.split("-")[1]
                    sub_word = subword_cid.split("-")[0]
                    if cid not in w_ids:
                        w_ids.append(cid)
                        subword_cids = [f"{s}-{cid}" for s in cid_features[cid]]
                        for ls in subword_cids:
                            print(ls)
                            for l in lines[ft][ls]:
                                w = {}
                                w["feature"] = l["feature"] 
                                w["guid"] = l["guid"] 
                                w["premise"] = l["premise"] 
                                w["hypothesis"] = l["hypothesis"] 
                                w["label"] = l["label"] 
                                w["significant_or_not"] = ""
                                f_writer.writerow(w)
            out_file = os.path.join(out_dir, f"{ft}_unk.csv")
            w_ids = []
            with open(out_file, "w") as f_w:
                f_writer = csv.DictWriter(f_w, fieldnames=["feature", "guid", "premise", "hypothesis", "label"])
                f_writer.writeheader()
                for subword_cid in chooses:
                    cid = subword_cid.split("-")[1]
                    sub_word = subword_cid.split("-")[0]
                    if cid not in w_ids:
                        w_ids.append(cid)
                        subword_cids = [f"{s}-{cid}" for s in cid_features[cid]]
                        for ls in subword_cids:
                            print(ls)
                            for l in lines[ft][ls]:
                                h_tokens = l["hypothesis"].split()
                                for i, h in enumerate(h_tokens): 
                                    if "*" in h and i != len(h_tokens)-1:
                                        h_tokens[i] = "UNK"
                                    elif "*" in h and i == len(h_tokens)-1:
                                        h_tokens[i] = "UNK"+h_tokens[i][-1]
                                        
                                w = {}
                                w["feature"] = l["feature"] 
                                w["guid"] = l["guid"] 
                                w["premise"] = l["premise"] 
                                w["hypothesis"] = " ".join(h_tokens)
                                w["label"] = ""
                                f_writer.writerow(w)

        else:
            out_file = os.path.join(out_dir, f"{ft}.csv")
            with open(out_file, "w") as f_w:
                f_writer = csv.DictWriter(f_w, fieldnames=["feature", "guid", "premise", "hypothesis", "label", "significant_or_not"])
                f_writer.writeheader()
                for ls in chooses:
                    print(ls)
                    for l in lines[ft][ls]:
                        w = {}
                        w["feature"] = l["feature"] 
                        w["guid"] = l["guid"] 
                        w["premise"] = l["premise"] 
                        w["hypothesis"] = l["hypothesis"] 
                        w["label"] = l["label"] 
                        w["significant_or_not"] = ""
                        f_writer.writerow(w)
            out_file = os.path.join(out_dir, f"{ft}_unk.csv")
            with open(out_file, "w") as f_w:
                f_writer = csv.DictWriter(f_w, fieldnames=["feature", "guid", "premise", "hypothesis", "label"])
                f_writer.writeheader()
                for ls in chooses:
                    print(ls)
                    for l in lines[ft][ls]:
                        h_tokens = l["hypothesis"].split()
                        for i, h in enumerate(h_tokens): 
                            if "*" in h and i != len(h_tokens)-1:
                                h_tokens[i] = "UNK"
                            elif "*" in h and i == len(h_tokens)-1:
                                h_tokens[i] = "UNK"+h_tokens[i][-1]
                                
                        w = {}
                        w["feature"] = l["feature"] 
                        w["guid"] = l["guid"] 
                        w["premise"] = l["premise"] 
                        w["hypothesis"] = " ".join(h_tokens)
                        w["label"] = ""
                        f_writer.writerow(w)


