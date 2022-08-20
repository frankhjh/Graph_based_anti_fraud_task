from torch.utils.data import Dataset


class User_info(Dataset):
    def __init__(self,feat,label):
        super(User_info,self).__init__()
        self.feat = feat
        self.label = label
    
    def __len__(self):
        return len(self.feat)
    
    def __getitem__(self,idx):
        return self.feat[idx],self.label[idx]