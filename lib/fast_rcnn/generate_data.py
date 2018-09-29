# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def generate_data(target_window, noise_number = 50000):

    #result_array = np.zeros(([target_window.shape[0] * (noise_number), target_window.shape[1]]))
    result_array = np.zeros(([target_window.shape[0], target_window.shape[1] * (noise_number+1)]))
    for i in range(target_window.shape[0]):
        
        # generate the noise shift of x and y for data
        shift_x = np.random.randint(20, 80, (noise_number))
        shift_y = np.random.randint(20, 80, (noise_number))
        scale_confident = np.random.randint(1, 50000, (noise_number)) * 0.00001
    
        
        # get the shift result
        noise_array = np.tile(target_window[i].reshape(-1, 1), noise_number)
        noise_array[0] = noise_array[0] + shift_x
        noise_array[1] = noise_array[1] + shift_y
        noise_array[2] = noise_array[2] + shift_x
        noise_array[3] = noise_array[3] + shift_y
        noise_array[4] = noise_array[4] * scale_confident
        
        result_array[i] = np.append(np.transpose(noise_array).reshape(1, -1), target_window[i])
        
    output = pd.DataFrame(result_array.reshape(1, -1))
    output.to_csv("../../ImageSets/gen_box.csv", index = False, columns = None)
    
    
 
def main():
    noise_number = 10000
    target_window = np.array([(100, 100, 200, 200, 0.99), (1600, 1600, 1700, 1700, 0.97), 
                          (100, 1600, 200, 1700, 0.98), (1600, 100, 1700, 200, 0.96)])
    generate_data(target_window, noise_number)
    
if __name__ == "__main__":
    main()
   


