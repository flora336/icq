# this file is used to split the test cases into different groups with a unique feature
# the input should be feature file 
# the output should be test cases with guid
import pickle
import json
from collections import defaultdict
def read_features(f_file):
    
    with open(f_file, "rb") as f:
        guid_features = pickle.load(f)
    
    return guid_features

def case_split(features, guid_fs):
    
    #guid_fs = read_features(feature_file)
    cases = defaultdict(list)
    for guid, f_set in guid_fs.items():
        print(f_set)
        for fea in f_set:
            cases[fea].append(guid)
    return cases

        
        





