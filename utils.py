import pandas as pd
import numpy as np

def conv_str_to_int(str_lst):
    lst = str_lst.strip('][').strip(" ").split(" ")
    lst = list(filter(lambda a: a != '', lst))
    lst = list(map(int, lst))
    return lst

def conv_str_to_float(str_lst):
    lst = str_lst.strip('][').strip(" ").split(" ")
    lst = list(filter(lambda a: a != '', lst))
    lst = list(map(float, lst))
    return lst

def prepare_dataframe(df):
    df['bbox'] = df['bbox'].apply(conv_str_to_int)
    df['category'] = df['category'].apply(conv_str_to_float)
    df['attributes'] = df['attributes'].apply(conv_str_to_float)
    return df