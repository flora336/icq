import csv
import os
from collections import defaultdict
import numpy as np
# this file is used to merge data for model evaluation
# this file is also used for split features to check how many data can be used for flip testing
# this file is also used to split features to check the flip rate for each cue.

# the guid is used to data merge and split merged guid have the structure: feature_name@@guid@@original/deleted

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
            w["exist"] = line["exist"]
            w["label"] = line["label"]
            cases[cid].append(w)
    return cases
            
def merge_data(deleted_dir, out_file):
    if os.path.exists(deleted_dir):
        deleted_files = os.listdir(deleted_dir)

    '''for labeled_name in labeled_files:
        l_f = os.path.join(labeled_dir, labeled_name)
        choose_res = read_labeled_file(l_f)
        for cid, res in choose_res.items():
            if res and cid in res_c:'''
    with open(out_file, "w") as f:
        f_w = csv.DictWriter(f, fieldnames=["guid", "premise", "hypothesis", "label", "exist"])
        f_w.writeheader()
        for deleted_name in deleted_files:
            d_f = os.path.join(deleted_dir, deleted_name)
            deleted_cases = read_deleted_file(d_f)
            for cid, d_lines in deleted_cases.items():
                for d_line in d_lines:
                    f_w.writerow(d_line)

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    
    parser.add_argument('--deleted_dir', type=str, default = None)
    parser.add_argument('--merge_changed_file', type=str, default = None)
    
    args = parser.parse_args()
    
    #deleted_dir = "/home/shanshan/icq/prob/log/roc/dataf/delete/"
    #all_deleted_file = "/home/shanshan/icq/prob/data/roc/deleted_test.csv"
    
    deleted_dir = args.deleted_dir
    all_deleted_file = args.merge_changed_file

    merge_data(deleted_dir, all_deleted_file)

if __name__ == "__main__":
    main() 
    

