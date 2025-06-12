#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import json
from typing import Union, List
from pathlib import Path
from importlib.machinery import SourceFileLoader
from easydict import EasyDict as AttrDict
    
    
def load_dataset(dataset_path: str, split: Union[str, List[str]]=[], node_rank: int=0, world_size: int=1) -> List[dict]:
    """load specific datasets"""
    if isinstance(split, str):
        split = [split]
    
    datasets = []
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r') as f:
            datasets = json.load(f)
        datasets = [data for data in datasets if not split or data['split'] in split] 
    elif dataset_path.endswith('.jsonl'):
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if not split or data['split'] in split:
                    datasets.append(data)          
    else:
        raise TypeError("only support '.json' or '.jsonl' datasets")
    
    datasets = [data for i, data in enumerate(datasets) if i % world_size == node_rank]
    print('Number of TODO Problems: {}'.format(len(datasets)))
    return datasets


def load_config(fname: str):
    """load config from specific file"""
    name = Path(fname).stem
    mod = SourceFileLoader(name, fname).load_module()

    config = {}
    for n in dir(mod):
        if not n.startswith("__"):
            config[n] = getattr(mod, n)
    config = AttrDict(config)

    return config
