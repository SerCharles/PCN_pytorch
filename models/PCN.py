import torch 
import torch.nn as nn
import torchvision



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        #shared mlp0:1d conv
        self.shared_mlp0 = nn.Sequential()
        self.shared_mlp0.add_module('mlp0', nn.Conv1d(in_channels = 3, out_channels = 128, kernel_size = 1))
        self.shared_mlp0.add_module('relu0', nn.ReLU(inplace = True))
        self.shared_mlp0.add_module('mlp1', nn.Conv1d(in_channels =128, out_channels = 256, kernel_size = 1))
        
        #shared mlp1:1d conv
        self.shared_mlp1 = nn.Sequential()
        self.shared_mlp1.add_module('mlp0', nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 1))
        self.shared_mlp1.add_module('relu0', nn.ReLU(inplace = True))
        self.shared_mlp1.add_module('mlp1', nn.Conv1d(in_channels = 512, out_channels = 1024, kernel_size = 1))


    def forward(self, x):
        x = x.permute(0, 2, 1) #b*3*n
        feature_f = self.shared_mlp0(x) #b*256*n
        global_feature_g, _ = torch.max(feature_f, -1, keepdim = True) #b*256*1
        expanded_global_feature_g = global_feature_g.repeat(1, 1, feature_f.size(2)) #b*256*n
        concated_feature = torch.cat([expanded_global_feature_g, feature_f], 1) #b*512*n
        global_feature_v = self.shared_mlp1(concated_feature) #b*1024*n
        global_feature_v, _ = torch.max(global_feature_v, -1, keepdim = True) #b*1024*1
        del _
        global_feature_v = global_feature_v.view(global_feature_v.size(0), -1) #b*1024
        return global_feature_v


#测试代码
if __name__ == '__main__':
    print(torch.__version__) 
    b = 20
    n = 35
    x = torch.rand(b, n, 3)
    model = Encoder()
    print(model)
    out = model(x)