import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def graph_data_loader(data_path):
    data = np.load(data_path)

    return data

def graph_info_extractor(num_users,edge_index,edge_type,edge_timestamp):
    '''
    extract graph data info

    Args:
        num_users:number of existing nodes in the graph
        edge_index: set of existed connections
        edge_type: connection type of each connection
        edge_timestamp: connection time of each connection
    
    Returns:
        user_start_edge_time_collector
        user_end_edge_time_collector
    '''
    user_start_edge_time_collector = defaultdict(dict)
    user_end_edge_time_collector = defaultdict(dict)
    
    # init
    types = {1,2,3,4,5,6,7,8,9,10,11}
    
    for tp in types:
        for idx in tqdm(range(num_users)):
            user_start_edge_time_collector[tp][idx] = list()
            user_end_edge_time_collector[tp][idx] = list()

    
    for users,tp,timestamp in tqdm(zip(edge_index,edge_type,edge_timestamp)):
        if tp in types:
            user_start_edge_time_collector[tp][users[0]].append(timestamp)
            user_end_edge_time_collector[tp][users[1]].append(timestamp)
    
   
    
    return user_start_edge_time_collector,user_end_edge_time_collector


def graph2table(num_users,start_info,end_info):
    '''
    transfer the graph info into table

    Args:
        num_users:number of users
        start_info
        end_info
    
    Returns:
        init_df

    '''
    
    init_df = pd.DataFrame({'user_id':[i for i in range(num_users)]})
    
    for k,v in tqdm(start_info.items()):
        init_df[f'start_type{k}'] = list(v.values())
    
    for k,v in tqdm(end_info.items()):
        init_df[f'end_type{k}'] = list(v.values())
    
    return init_df


