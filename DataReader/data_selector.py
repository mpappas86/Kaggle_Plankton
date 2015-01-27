import numpy as np
np.random.seed(1)
from gen_synthetic import transform_images
from training_list import classcounts

def select_subset(alldata, desired_samples = 500, percent_range = [0,1], include_synthetic=True):
    labelnames = [x[0] for x in classcounts]
    ret = []
    samples_flag = [True]
    index_flag = [True]
    num_all_samples = 0
    for key, val in alldata.iteritems():
        lower_index = int(percent_range[0]*len(val))
        upper_index = int(percent_range[1]*len(val))
        if lower_index == upper_index:
            index_flag[0] = False
            index_flag.append(key)
            if upper_index == len(val):
                lower_index = lower_index - 1
            else:
                upper_index = upper_index + 1
        working_val = val[lower_index:upper_index]
        if(include_synthetic):
            pool = np.array(transform_images(working_val))
        else:
            pool = np.array(working_val)
        if pool.shape[0] < desired_samples:
            samples_flag[0] = False
            samples_flag.append(key)
            numsamples = pool.shape[0]
            choice = pool.reshape(numsamples, pool.shape[1]*pool.shape[2])
        else:
            numsamples = desired_samples
            choice = pool[np.random.choice(pool.shape[0], numsamples, replace=False),...].reshape(numsamples, pool.shape[1]*pool.shape[2])
        label = np.array([x == key for x in labelnames])
        rlabel = np.repeat(label[np.newaxis,:],numsamples,axis=0)
        labeledchoice = np.concatenate((choice, rlabel),axis=1)
        ret.append(labeledchoice)
        num_all_samples = num_all_samples + numsamples
    final = np.concatenate(ret)
    np.random.shuffle(final)
    return final.T, samples_flag, index_flag