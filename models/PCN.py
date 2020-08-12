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


class Decoder(nn.Module):
    def __init__(self, grid_scale = 0.05, grid_size = 2, num_coarse = 512):
        super(Decoder, self).__init__()
        self.grid_scale = grid_scale
        self.grid_size = grid_size
        self.num_coarse = num_coarse
        self.num_fine = num_coarse * grid_size * grid_size

        #mlp0
        self.mlp = nn.Sequential()
        self.mlp.add_module('mlp0', nn.Linear(in_features = 1024, out_features = 1024, bias = True))
        self.mlp.add_module('relu0', nn.ReLU(inplace = True))
        self.mlp.add_module('mlp1', nn.Linear(in_features = 1024, out_features = 1024, bias = True))
        self.mlp.add_module('relu1', nn.ReLU(inplace = True))
        self.mlp.add_module('mlp2', nn.Linear(in_features = 1024, out_features = num_coarse * 3 , bias = True))
        
        #folding_mlp:1d conv
        self.folding_mlp = nn.Sequential()
        self.folding_mlp.add_module('mlp0', nn.Conv1d(in_channels = 1024 + 3 + 2, out_channels = 512, kernel_size = 1))
        self.folding_mlp.add_module('relu0', nn.ReLU(inplace = True))
        self.folding_mlp.add_module('mlp1', nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 1))
        self.folding_mlp.add_module('relu1', nn.ReLU(inplace = True))
        self.folding_mlp.add_module('mlp2', nn.Conv1d(in_channels = 512, out_channels = 3, kernel_size = 1))

        #grid
        x = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
        y = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
        self.grid_x, self.grid_y = torch.meshgrid(x, y)

    def forward(self, feature):
        #生成coarse的参数
        coarse = self.mlp(feature) #b * (3 * coarse_size)
        coarse = coarse.view(coarse.size(0), 3, -1) #B * 3 * coarse_size
        #print(coarse.size())
        center = coarse.view(coarse.size(0), coarse.size(1), coarse.size(2), 1)
        center = center.repeat(1, 1, 1, self.grid_size ** 2)
        center = center.view(center.size(0), center.size(1), -1)
        #print(center.size())
        
        #生成folding用的x，y网格参数
        grid = torch.cat([self.grid_x, self.grid_y], -1) #u*u*2
        grid = grid.view(1, 2, self.grid_size ** 2) #1 * 2 * t
        expanded_grid = grid.repeat(1, 1, self.num_coarse) #1 * 2 * (t*coarse_size) = 1*2*fine_size
        expanded_grid = expanded_grid.repeat(feature.size(0), 1, 1) #B * 2* fine_size
        #print(expanded_grid.size())

        #生成encoder信息对应的参数
        encoder_info = feature.view(feature.size(0), feature.size(1), 1) #B*1024*1
        encoder_info = encoder_info.repeat(1, 1, self.num_fine)#B*1024*fine_size
        #print(encoder_info.size())

        #结合得到整个feture
        full_feature = torch.cat([expanded_grid, center, encoder_info], 1) #B * 1029 *fine_size
        fine = self.folding_mlp(full_feature) #B*3*fine_size
        fine = fine + center #B*3*fine_size
        coarse = coarse.permute(0, 2, 1) #B*coarse_size*3
        fine = fine.permute(0, 2, 1) #B*fine_size * 3
        #print(coarse.size())
        #print(fine.size())
        return coarse, fine

class PCN(nn.Module):
    def __init__(self):
        super(PCN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def to(self, device, **kwargs):
        self.decoder.grid_x = self.decoder.grid_x.to(device)
        self.decoder.grid_y = self.decoder.grid_y.to(device)
        super(PCN, self).to(device, **kwargs)

#测试代码
if __name__ == '__main__':
    print(torch.__version__) 
    b = 20
    n = 35
    x = torch.rand(b, n, 3)
    model = PCN()
    print(model)
    out = model(x)


