import  numpy as np 
import  torch 
import os
import pickle
from typing import Any, Callable, TypeVar

T= TypeVar('T')


def disk_cache(func:Callable[[],T],filename:str,is_force_load:bool=False)->T:
    data:Any

    cache_dir = "cache"
    filename_path = os.path.join(cache_dir,filename)

    if(cache_dir not in os.listdir()):
        os.mkdir(cache_dir)


    if(is_force_load): 
        data = func()  
    else:
        if(not os.path.exists(filename_path)):
           
            data = func()    
            with open(filename_path,"wb") as pkl_file:
                pickle.dump(data,pkl_file)
        else:
        

            with open(filename_path,"rb") as pkl_file:
                data = pickle.load(pkl_file)
    
    return data


