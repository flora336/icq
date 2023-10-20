import nltk
import csv
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import math
import pickle
import multiprocessing
import time
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#def lmi(word, unigram_freq, bigram_freq, labels):
def lmi(count_w, count_l, count_w_l, count_all):
    '''lmi(w,l) = p(w, l)log(p(w|l)/p(l))
        p(w,l) = count(w,l)/count(w)
        p(l)   = count(l)/ count(all)'''
    #return math.exp((count_w_l/count_all)*math.log(count_w_l*count_all/float(count_w*count_l),2))
    return math.exp((count_w_l/count_all)*math.log(count_w_l*count_all/float(count_w*count_l),2))

#def pmi(word, unigram_freq, bigram_freq, labels):
def pmi(count_w, count_l, count_w_l, count_all):
    '''pmi(w,l)= log(p(l|w)/p(l))
        p(l|w) = count(w,l)/count(w)
        p(l)   = count(l)/ count(all)'''
    print(count_w_l, count_all, count_w, count_l)
    return math.log(count_w_l*count_all/float(count_w*count_l),2)

def jsd(count_wl, count_l, count_wf, count_lf, count_all):
    '''ratio(w,l) = |(St+1)/(Sf+1)-(Lt+1)/(Lf+1)|'''
    p0 = st = float(count_wl)
    p1 = sf = float(count_wf)
    lt = float(count_l)
    lf = float(count_lf)
    q0 = lt/float(lt+lf)
    q1 = lf/float(lt+lf)

    if p0 == 0.0 and p1 != 0.0:
        return 0.5*(p1*math.log(2*p1/(p1+q1),2)+q0*math.log(2*q0/(p0+q0),2) + q1*math.log(2*q1/(p1+q1),2))
    elif p0 != 0.0 and p1 == 0.0:
        return 0.5*(p0*math.log(2*p0/(p0+q0),2)+q0*math.log(2*q0/(p0+q0),2) + q1*math.log(2*q1/(p1+q1),2))
    elif p0 != 0.0 and p1 != 0.0:
        return 0.5*(p0*math.log(2*p0/(p0+q0),2)+p1*math.log(2*p1/(p1+q1),2)+\
            q0*math.log(2*q0/(p0+q0),2) + q1*math.log(2*q1/(p1+q1),2))
    else:
        return 0.5*(q0*math.log(2*q0/(p0+q0),2) + q1*math.log(2*q1/(p1+q1),2))

def freq(count_wl, count_l, count_wf, count_lf, count_all):
    st = float(count_wl)
    sf = float(count_wf)
    lt = float(count_l)
    lf = float(count_lf)
    return st
def condition(count_wl, count_l, count_wf, count_lf, count_all):
    st = float(count_wl)
    sf = float(count_wf)
    lt = float(count_l)
    lf = float(count_lf)
    return st/(st+sf)
#def ratio(word, unigram_freq, bigram_freq, labels):
def ratio(count_wl, count_l, count_wf, count_lf, count_all):
    '''ratio(w,l) = |(St+1)/(Sf+1)-(Lt+1)/(Lf+1)|'''
    st = float(count_wl)
    sf = float(count_wf)
    lt = float(count_l)
    lf = float(count_lf)
    return abs((st+1)/(sf+1)-(lt+1)/(lf+1))

def w_ratio(count_wl, count_l, count_wf, count_lf, count_all):
    '''ratio(w,l) = |(St+1)/(Sf+1)-(Lt+1)/(Lf+1)|'''
    st = float(count_wl)
    sf = float(count_wf)
    lt = float(count_l)
    lf = float(count_lf)
    return (st/count_all)*abs((st+1)/(sf+1)-(lt+1)/(lf+1))



#def angle_diff(word, unigram_freq, bigram_freq, labels):
def angle_diff(count_wl, count_l, count_wf, count_lf, count_all):
    '''angle_diff(w,l) = |arctan((Sf)/(St))-arctan((Lf)/(Lt))|'''
    st = float(count_wl)
    sf = float(count_wf)
    lt = float(count_l)
    lf = float(count_lf)
    return abs(math.atan(sf/st)- math.atan(lf/lt))

#def cos(word, unigram_freq, bigram_freq, labels):
def cos(count_wl, count_l, count_wf, count_lf, count_all):
    '''cos(w,l) = |(st*lt+sf*lf)/sqrt(st^2+ sf^2)*sqrt(lt^2+lf^2)|'''
    st = float(count_wl)
    sf = float(count_wf)
    lt = float(count_l)
    lf = float(count_lf)
    return 1-abs((st*lt+sf*lf)/(math.sqrt(pow(st,2)+ pow(sf,2))*math.sqrt(pow(lt,2)+pow(lf,2))))

#def wp(word, unigram_freq, bigram_freq, labels):
def wp(count_wl, count_l, count_wf, count_lf, count_all):
    '''wp(w,l) = |(st*lt+sf*lf)/sqrt(st^2+ sf^2)*sqrt(lt^2+lf^2)|'''
    st = float(count_wl)
    sf = float(count_wf)
    lt = float(count_l)
    lf = float(count_lf)
    cos = abs((st*lt+sf*lf)/(math.sqrt(pow(st,2)+ pow(sf,2))*math.sqrt(pow(lt,2)+pow(lf,2))))
    #cos = cos(count_wl, count_l, count_wf, count_lf, count_all)
    return (1-cos)*pow((st+sf)/count_all,cos)

'''def build_freq(train_file, log_dir, is_lower=True):
    
    unigram_list = []
    bigram_list = []
    pkl_file = os.path.join(log_dir, 'unigram.pkl')
    
    unigram_freq = defaultdict(int)
    unigram_label_freq = defaultdict(int)
    label_freq = defaultdict(int)
    list_count = 0
    
    with open(train_file, 'r') as f, open(pkl_file, 'wb') as f_w:
        reader = csv.DictReader(f)
        index = 0
        for line in reader:
            index +=1
            if index%1000 == 0:
                print(index)
            premise = line['premise']
            hypothesis = line['hypothesis']
            label = line['label']
            
            premise_tokens = []
            hypothesis_tokens = hypothesis.split()
            
            for i,hyp in enumerate(hypothesis_tokens):
                if i == len(hypothesis_tokens)-1:
                    break
                unigram_freq[hyp] += 1
                unigram_label_freq[(hyp, label)] += 1

            #unigram_freq["-"+label] += 1
            label_freq[label] += 1
        
        pkl_dict = {}
        pkl_dict['unigram'] = unigram_freq
        pkl_dict['unigram_label'] = unigram_label_freq
        pickle.dump(pkl_dict, f_w)

    return unigram_freq, unigram_label_freq, label_freq '''

def build_freq(guid_features, guid_labels, is_lower=True):
    
    unigram_list = []
    bigram_list = []
    
    unigram_freq = defaultdict(int)
    unigram_label_freq = defaultdict(int)
    label_freq = defaultdict(int)
    list_count = 0
    
    for guid, features in guid_features.items():
        label = guid_labels[guid]
        label_freq[label] += 1
        for f in features:
            unigram_freq[f] += 1
            unigram_label_freq[(f, label)] += 1
    #print(unigram_label_freq[("word\treplace", "right")])
    #print(unigram_freq["word\treplace"])
    #exit()
    #print("unigram_label_freq:", unigram_label_freq)
    #print("unigram_freq:", unigram_freq)
    #print("label_freq:", label_freq)
    #exit()
    return unigram_freq, unigram_label_freq, label_freq 


def init():
    global tf
    global sess
    #config = tf.ConfigProto()
    config =  tf.compat.v1.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)

def trans_false_dic(unigram_freq, unigram_label_freq, label_freq):
    '''bigram_freq:Counter trans to bigram_freq_f:dict 
    with format word_label:num(word_otherlabel)'''
    labels = label_freq.keys()
    unigram_label_freq_f = {}
    for w, num in unigram_label_freq.items():
        unigram_label_freq_f[w] = 0
        word = w[0]
        label = w[1]
        for l in label_freq:
            if l != label and (word,l) in unigram_label_freq:
                #print(word, l)
                unigram_label_freq_f[w] += unigram_label_freq[(word, l)]
    label_f = {}
    for l in labels:
        label_f[l] = 0
        for l_f in label_freq:
            if l_f != l:
                label_f[l] += label_freq[l_f]
    return unigram_label_freq_f, label_f

def calculate_feature_score(unigram_freq, unigram_label_freq, label_freq, score_type):
    score_function = score_dict[score_type]
    labels = label_freq.keys()
    index = 0
    cores = 10
    pool = multiprocessing.Pool(processes=cores, initializer=init)
    w_list = list(unigram_label_freq.keys())
    count_all = float(sum(unigram_freq.values()))
    #print("count_all:",count_all)
    #exit()
    if score_type in ['pmi','lmi']:
        inputs = [(unigram_freq[w[0]], label_freq[w[1]], unigram_label_freq[w], count_all,) for w in w_list]
    else:
        unigram_label_freq_f, label_f = trans_false_dic(unigram_freq, unigram_label_freq, label_freq)
        inputs = [(unigram_label_freq[w], label_freq[w[1]], unigram_label_freq_f[w], label_f[w[1]], count_all,) for w in w_list]
    start_time = time.time()
    score_list = pool.starmap(score_function, inputs)
    pool.close()
    pool.join()
    end_time = time.time()
    score_list = [x for x in score_list]
    pkl_score_dict = {}
    for w in w_list:
        #print(w)
        st = sf = lt = lf = 0.0
        #score = score_function(w, unigram_freq, bigram_freq, labels)
        score = score_list[index]
        words = "\t".join(w)
        #f.write(words+'\t'+str(score)+'\n')
        label = w[1]
        for l in labels:
            if l == label:
                st = unigram_label_freq[w]
                lt = label_freq[l]
            else:
                if (w[0],l) in unigram_label_freq:
                    sf += unigram_label_freq[(w[0],l)]
                lf += label_freq[l]
                #pkl_score_dict["-".join(w)] = score
            pkl_score_dict["-".join(w)] = score
        #print(st, lt, sf, lf)
        index += 1

        #with open(os.path.join(output_dir, f'{score_type}.pkl'), 'wb') as f_pkl:
        #    pickle.dump(pkl_score_dict, f_pkl)
    return pkl_score_dict


def write_score_file(log_dir, unigram_freq, unigram_label_freq, label_freq, score_type):
    with open(os.path.join(log_dir, f'{score_type}.txt'), 'w') as f:
        score_function = score_dict[score_type]
        f.write('token\tfscore\tst\tsf\tlt\tlf\n')
        index = 0
        cores = 10
        #pool = multiprocessing.Pool(processes=cores)
        pool = multiprocessing.Pool(processes=cores, initializer=init)
        w_list = list(unigram_label_freq.keys())
        count_all = float(sum(unigram_freq.values()))
        if score_type in ['pmi','lmi']:
            inputs = [(unigram_freq[w[0]], label_freq[w[1]], unigram_label_freq[w], count_all,) for w in w_list]
        else:
            unigram_label_freq_f, label_f = trans_false_dic(unigram_freq, bigram_freq, label_freq)
            inputs = [(unigram_label_freq[w], label_freq[w[1]], unigram_label_freq_f[w], label_f[w[1]], count_all,) for w in w_list]
        start_time = time.time()
        score_list = pool.starmap(score_function, inputs)
        pool.close()
        pool.join()
        end_time = time.time()
        score_list = [x for x in score_list]
        pkl_score_dict = {}
        for w in unigram_label_freq:
            st = sf = lt = lf = 0.0
            #score = score_function(w, unigram_freq, bigram_freq, labels)
            score = score_list[index]
            words = "\t".join(w)
            #f.write(words+'\t'+str(score)+'\n')
            label = w[1]
            for l in labels:
                if l == label:
                    st = unigram_label_freq[w]
                    lt = label_freq[l]
                else:
                    if (w[0],l) in unigram_label_freq:
                        sf += unigram_label_freq[(w[0],l)]
                    lf += label_freq[l]
            index += 1
            f.write(words+'\t'+str(score)+'\t'+str(st) + '\t'+str(sf)+'\t'+str(lt)+'\t'+str(lf)+'\n')
            pkl_score_dict["-".join(w)] = score

        with open(os.path.join(output_dir, f'{score_type}.pkl'), 'wb') as f_pkl:
            pickle.dump(pkl_score_dict, f_pkl)
    return pkl_score_dict

score_dict = {"lmi":lmi, "pmi":pmi, "condition": condition, "frequency": freq}
def get_feature_score(id_features, id_labels, score_type, device_n):
    start_time = time.time()
    global device_num
    global score_dict
    device_num = device_n
    unigram_freq, unigram_label_freq, labels_freq = build_freq(id_features, id_labels, is_lower=True)

    word_label_score = calculate_feature_score(unigram_freq, unigram_label_freq, labels_freq, score_type)
    end_time = time.time()
    print("cal_bi_score time cost:", end_time-start_time)
    return word_label_score, unigram_label_freq

def cal_mixed_feature_score(word_label_score, unigram_label_freq):
    labels = set()
    words = set()
    cueness_score = defaultdict(float)
    distribution = defaultdict()
    for wl, score in word_label_score.items():
        l = wl.split("-")[-1]
        w = "-".join(wl.split("-")[:-1])
        labels.add(l)
        words.add(w)
    for w in words:
        cueness = 0
        cueness_score[w] = 0
        distribution[w] = {}
        for l in labels:
            wl = "-".join([w, l])
            if wl in word_label_score.keys():
                cueness += word_label_score[wl]
            ave_score = cueness/float(len(labels))
        for l in labels:
            if (w,l) in unigram_label_freq:
                distribution[w][l] = unigram_label_freq[(w,l)]
            else:
                distribution[w][l] = 0
            wl = "-".join([w, l])
            if wl in word_label_score.keys():
                cueness_score[w] += pow((word_label_score[wl]-ave_score), 2)/float(len(labels))
            else:
                cueness_score[w] += pow((0 - ave_score), 2)/float(len(labels))
    return cueness_score, distribution


    
'''def cal_score(task, output_dir, input_file, score_types, device_n):
    start_time = time.time()
    global device_num
    global score_dict
    device_num = device_n
    
    unigram_freq, unigram_label_freq, labels_freq = build_freq(input_file, is_lower=True)
    #print('unigram_size:', len(unigram_freq))
    #print('bigram_size:', len(bigram_freq))
    #labels = labels_dict[task_name]
    word_label_scores = defaultdict()
    for score_type in score_types:
        print('score_type:', score_type)
        word_label_scores[score_type] = write_score_file(output_dir, unigram_freq, unigram_label_freq, labels_freq, score_type)
    end_time = time.time()
    print("cal_bi_score time cost:", end_time-start_time)
    return word_label_scores'''
#if __name__ == "__main__":
#    cal_bi_score()
