import torch
import pandas as pd
from config import DefaultConfig
import json
import pickle
import torchsnooper


class DatasetFromList(torch.utils.data.Dataset):
    def __init__(self, data: list):
        self.data = data

    def __getitem__(self, index):
        try:
            data = self.data[index]
            return data
        except KeyError:
            raise IndexError

    def __len__(self):
        return len(self.data)


def rids_list_to_mask_matrix(rid_list, num_of_nodes):
    mask = torch.zeros([num_of_nodes], dtype=torch.float32)
    mask = mask.index_fill(dim=0, index=torch.tensor(rid_list), value=1)
    return mask


# @torchsnooper.snoop()
def prepareDataloader(opt: DefaultConfig):
    print("Preparing dataloader...")

    data = []

    with open(opt.gcn_train_file, mode='r', encoding='utf-8') as f:
        for line in f:
            mee_search_result_s, gps_match_result_s = line.split('|')  # [[],[]] []
            mee_search_result = json.loads(mee_search_result_s)
            gps_match_result = json.loads(gps_match_result_s)
            data.append((mee_search_result, gps_match_result))

    with open(opt.nxgraph_path, mode='rb') as f:
        graph = pickle.load(f)

    l = int(0.75 * len(data))
    train_data = data[:l]
    valid_data = data[l:]

    print("Done prepare dataloader.")

    return torch.utils.data.DataLoader(dataset=DatasetFromList(data=train_data), shuffle=opt.SHUFFLE_DATASET,
                                       batch_size=opt.BATCH_SIZE), \
           torch.utils.data.DataLoader(dataset=DatasetFromList(data=valid_data), shuffle=opt.SHUFFLE_DATASET,
                                       batch_size=opt.BATCH_SIZE), \
           graph
