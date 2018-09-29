# -*- coding: utf-8 -*-
import sys
sys.path.extend(['..'])

from nms_wrapper import nms, soft_nms
import pandas as pd
import numpy as np
import time

def read_csv(filename = "../../ImageSets/box.csv", sep = ' '):
    df1 = pd.read_csv(filename, sep = sep).fillna(0)
    data = np.array(df1).astype("float32")
    return data

def main():
    data = read_csv(filename = "../../ImageSets/gen_box.csv", sep = ',')
    
    assert(data.shape[1] % 5 == 0)
    for i in range(data.shape[0]):
        dets = data[i].reshape(-1, 5)
        init_time = time.time()
        nms(dets, 0.5, force_cpu=True)
        middle_time = time.time()
        cpu_time = (middle_time - init_time)
#        nms(dets, 0.5, force_cpu=False)
        gpu_time = (time.time() - middle_time)
        print("number of picture is: {}, the cpu_time is: {}, the gpu_time is: {} \n".format(
                dets.shape[0], cpu_time, gpu_time))
        
        np.random.shuffle(dets)
        init_time = time.time()
        nms(dets, 0.5, force_cpu=True)
        middle_time = time.time()
        cpu_time = (middle_time - init_time)
#        nms(dets, 0.5, force_cpu=False)
        gpu_time = (time.time() - middle_time)
        print("After shuffle, the cpu_time is: {}, the gpu_time is: {} \n".format(cpu_time, gpu_time))
    
    
if __name__ == "__main__":
    main()


