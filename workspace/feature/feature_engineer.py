import pandas as pd
from tqdm import tqdm

def feature_building(df,edge_type):
    '''

    Args:
        df:DataFrame which store basic info
        edge_type:the set of edge types existed in graph
    '''
    
    types = set(edge_type)
    num_type = len(types)
    
    for i in tqdm(range(1,num_type+1)):
        df[f'start_type{i}_count'] = [len(i) for i in df[f'start_type{i}'].values]
        df[f'start_type{i}_unique_count'] = [len(set(i)) for i in df[f'start_type{i}'].values]
        df[f'start_type{i}_max_min_gap'] = [max(i)-min(i) if i else 0 for i in df[f'start_type{i}'].values]
    
    for i in tqdm(range(1,num_type+1)):
        df[f'end_type{i}_count'] = [len(i) for i in df[f'end_type{i}'].values]
        df[f'end_type{i}_unique_count'] = [len(set(i)) for i in df[f'end_type{i}'].values]
        df[f'end_type{i}_max_min_gap'] = [max(i)-min(i) if i else 0 for i in df[f'start_type{i}'].values]
    
    feats = [f'start_type{i}_count' for i in range(1,num_type+1)] +\
    [f'end_type{i}_count' for i in range(1,num_type+1)] +\
    [f'start_type{i}_unique_count' for i in range(1,num_type+1)] +\
    [f'end_type{i}_unique_count' for i in range(1,num_type+1)] +\
    [f'start_type{i}_max_min_gap' for i in range(1,num_type+1)] +\
    [f'end_type{i}_max_min_gap' for i in range(1,num_type+1)]
    
    return df[feats]


def feature_comb(user_feats,user_embedd):
    '''

    Args:
        user_feats
        user_embedd
    '''
    for i in tqdm(range(user_embedd.shape[1])):
        user_feats[f'x{i}'] = user_embedd[:,i]
    
    return user_feats