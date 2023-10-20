import csv
import os
from collections import defaultdict
import numpy as np
# this file is used to merge data for model evaluation
# this file is also used for split features to check how many data can be used for flip testing
# this file is also used to split features to check the flip rate for each cue.

# the guid is used to data merge and split merged guid have the structure: feature_name@@guid@@original/deleted

def read_labeled_file(file_name):
    cases = defaultdict(list)
    chooses = defaultdict()
    # sig_num_
    feature = "_".join(file_name.split("_")[2:])
    with open(file_name, "r") as f:
        f_reader = csv.DictReader(f)
        for line in f_reader:
            case_id = line["guid"].split("-")[0]
            cases[case_id].append(line["significant_or_not"])
    for cid in cases:
        choose_list = []
        c_list = cases[cid]
        for c in c_list:
            if "y" in c.lower():
                choose_list.append(False)
            elif "n" in c.lower():
                choose_list.append(True)
            else:
                choose_list.append(None)
        if False in choose_list:
            chooses[cid] = False
        elif True in choose_list:
            chooses[cid] = True
        else:
            chooses[cid] = False

    return chooses

def read_deleted_file(file_name):
    cases = defaultdict(list)
    feature = "_".join(file_name.split("_")[1:])
    with open(file_name, "r") as f:
        f_reader = csv.DictReader(f)
        for line in f_reader:
            w = {}
            cid = line["guid"].split("-")[0]
            w["guid"] = "@@".join([feature, line["guid"], "deleted"])
            w["premise"] = line["premise"]
            w["hypothesis"] = line["chypothesis"]
            w["label"] = line["label"]
            cases[cid].append(w)
    return cases


def read_original_file(file_name):
    cases = defaultdict(list)
    labels = defaultdict()
    with open(file_name, "r") as f:
        f_reader = csv.DictReader(f)
        for line in f_reader:
            #guid = "@@".join([feature, line[guid], file_type])
            cid = line["guid"].split("-")[0]
            cases[cid].append(line)
            labels[line["guid"]] = line["label"]
    return cases, labels
            
def merge_data(deleted_dir, out_file):
    if os.path.exists(deleted_dir):
        deleted_files = os.listdir(deleted_dir)

    '''for labeled_name in labeled_files:
        l_f = os.path.join(labeled_dir, labeled_name)
        choose_res = read_labeled_file(l_f)
        for cid, res in choose_res.items():
            if res and cid in res_c:'''
    with open(out_file, "w") as f:
        f_w = csv.DictWriter(f, fieldnames=["guid", "premise", "hypothesis", "label"])
        f_w.writeheader()
        for deleted_name in deleted_files:
            d_f = os.path.join(deleted_dir, deleted_name)
            deleted_cases = read_deleted_file(d_f)
            for cid, d_lines in deleted_cases.items():
                for d_line in d_lines:
                    f_w.writerow(d_line)
def read_result(file_name):
    res = defaultdict()
    with open(file_name, "r") as f:
        count = 0
        for line in f:
            if count != 0:
                r = line.strip("\n").split("\t")
                if len(r) == 2:
                    res[r[0]] = r[1]
            count += 1
    return res

def check_data(original_file, all_deleted_file, prompt_file, labeled_dir, original_res_file, deleted_res_file, out_file):
    original_cases, ground_labels = read_original_file(original_file)
    all_deleted_cases, delete_ground_labels = read_original_file(all_deleted_file)
    original_pred = read_result(original_res_file) 
    deleted_pred = read_result(deleted_res_file) 
    prompt_pred = read_result(prompt_file) 
    
    
    all_cases = defaultdict(int)
    alter_cases = defaultdict(int)

    deleted_flip = defaultdict(int)
    pred_res = defaultdict(defaultdict)
    deleted_pred = defaultdict(defaultdict)
    original_res = defaultdict()
    for unique_id, pred in deleted_pred.items():

        feature = unique_id.split("@@")[0]
        guid = unique_id.split("@@")[1].split("-")[0]
        deleted_pred[feature][guid] = pred
        if pred == delete_ground_labels[unique_id]:
            pred_res[feature][guid] = True
        else:
            pred_res[feature][guid] = False
    print("what:", pred_res["word_what.csv"]) 
    for uid, pred in original_pred.items():
        cid = uid.split("-")[0]
        if pred == ground_labels[uid]:
            original_res[cid] = True
        else:
            original_res[cid] = False
    if os.path.exists(labeled_dir):
        labeled_files = os.listdir(labeled_dir)

    if os.path.exists(deleted_dir):
        deleted_files = os.listdir(deleted_dir)

    with open(out_file, "w") as f:
        f_w = csv.DictWriter(f, fieldnames=["feature", "all_cases", "alter_cases", "all_acc", "no_change_acc", "change_acc", "flip_rate", "prompt_rate", "rank", "right_rate"])
        f_w.writeheader()
        for labeled_name in labeled_files:
            w = {}
            l_f = os.path.join(labeled_dir, labeled_name)
            choose_res = read_labeled_file(l_f)
            feature = "_".join(labeled_name.split("_")[2:])
            rank = labeled_name.split("_")[1]
            all_cases = len(choose_res)
            alter_cases = 0
            all_acc_list = []
            no_change_acc_list = []
            change_acc_list = []
            change_pred_list = []
            no_change_pred_list = []
            flip_list = []
            for cid, res in choose_res.items():
                all_acc_list.append(original_res[cid])
                if res:
                    alter_cases += 1
                    no_change_acc_list.append(original_res[cid])
                    change_acc_list.append(pred_res[feature][cid])
                    no_change_pred_list.append(original_pred[cid])
                    change_pred_list.append(deleted_pred[feature][cid])
                else:
                    print(choose_res)
            if len(all_acc_list) != 0 and alter_cases >0 :
                w["feature"] = feature
                w["all_cases"] = all_cases
                w["alter_cases"] = alter_cases
                w["rank"] = rank
                w["all_acc"] = sum(all_acc_list)/float(len(all_acc_list))
                w["no_change_acc"] = sum(no_change_acc_list)/float(len(no_change_acc_list))
                w["change_acc"] = sum(change_acc_list)/float(len(change_acc_list))
                w["flip_rate"] = sum(np.array(no_change_acc_list) != np.array(change_acc_list))/float(len(no_change_acc_list))
                w["right_rate"] = len([x for x in change_pred_list if x == "right"])/float(len(change_acc_list))
                f = feature.split(".csv")[0]
                if f in prompt_pred:
                    w["prompt_rate"] = prompt_pred[feature.split(".csv")[0]]
                else:
                    w["prompt_rate"] = ""

                f_w.writerow(w)
            else:
                print(f"{feature} is not exist")

if __name__ == "__main__":
    deleted_dir = "/home/shanshan/icq/prob/log/roc/dataf/delete/"
    all_deleted_file = "/home/shanshan/icq/prob/data/roc/deleted_test.csv"
    labeled_dir = "/home/shanshan/icq/prob/log/roc/dataf/nli_sig"
    original_file = "/home/shanshan/icq/prob/data/roc/test.csv"
    original_res_file = "/home/shanshan/generate_noise/transformers/output_dir/ROC/original_1_bert_base/test_results_original.txt"
    deleted_res_file = "/home/shanshan/generate_noise/transformers/output_dir/ROC/original_1_bert_base/test_results_features.txt"
    prompt_file = "/home/shanshan/icq/prob/log/roc/bert/prompt/flip_rate.txt"
    out_file = "roc.csv"
    merge_data(deleted_dir, all_deleted_file)
    check_data(original_file, 
            all_deleted_file, 
            prompt_file, 
            labeled_dir, 
            original_res_file, 
            deleted_res_file, 
            out_file)
