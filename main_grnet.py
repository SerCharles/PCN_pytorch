import torch
import chamfer3D.dist_chamfer_3D
from data import MyDataset
from models.grnet import GRNet
from torch.utils.data import DataLoader
import os
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
DATA_PATH = '/home/shenguanlin/shapenet/train'
VALID_PATH = '/home/shenguanlin/shapenet/val'
RESULT_PATH = 'result.txt'
TYPE_CODE_LIST = []
TYPE_CODE = '04256520'
BATCH_SIZE = 8


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d or \
       type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)

def train_net(device=None, epochs=100):
    the_type_code = TYPE_CODE
    best_train_dis = 14530529
    best_valid_dis = 14530529
    print('training start')
    print('getting dataset')
    dataset = MyDataset(DATA_PATH, TYPE_CODE)
    valid_dataset = MyDataset(VALID_PATH, TYPE_CODE)
    print('getting dataloader')
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_data_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print('data got!')

    # Create the networks
    grnet = GRNet()
    grnet.apply(init_weights)

    #if device:
    #    grnet.to(device)
    if torch.cuda.is_available():
        grnet = torch.nn.DataParallel(grnet).cuda()

    # Create the optimizers
    grnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, grnet.parameters()),
                                       lr=1e-4,
                                       weight_decay=0,
                                       betas=(.9, .999))
    grnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(grnet_optimizer,
                                                              milestones=[50],
                                                              gamma=.5)


    # Load pretrained model if exists
    init_epoch = 0
    best_metrics = None



    for epoch in range(1, epochs+1):
        print('============='*4, 'epoch: ', epoch, '============='*4)
        total_train_dis = 0
        total_train_num = 0
        total_valid_dis = 0
        total_valid_num = 0

        #train
        grnet.train()
        for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader):
            if device:
                the_data = the_data.to(device)
                ground_truth_fine = ground_truth_fine.to(device)
                ground_truth_coarse = ground_truth_coarse.to(device)
            '''print(ground_truth_fine.size())
            print(ground_truth_coarse.size())
            print(the_data.size())
            print(the_data.device)
            print(ground_truth_fine.device)
            print(ground_truth_coarse.device)'''
            coarse, fine = grnet(the_data)
            if device:
                coarse.to(device)
                fine.to(device)
            '''print(coarse.device)
            print(fine.device)
            print(coarse.size())
            print(fine.size())'''
            dis_fine1, dis_fine2, _, _ = chamLoss(fine, ground_truth_fine)
            dis_fine = torch.mean(dis_fine1) + torch.mean(dis_fine2)
            dis_coarse1, dis_coarse2, _, _ = chamLoss(coarse, ground_truth_coarse)
            dis_coarse = torch.mean(dis_coarse1) + torch.mean(dis_coarse2)
            dis = dis_fine + 0.5 * dis_coarse
            grnet_optimizer.zero_grad()
            dis.backward()
            grnet_optimizer.step()
            grnet_lr_scheduler.step()

            total_train_dis += dis.item()
            total_train_num += 1
            print('epoch:[{}/{}] batch {}, dis: {}'.format(epoch, epochs, i+1, dis.item() * 10000))
        avg_dis = (total_train_dis / total_train_num) * 10000
        if avg_dis < best_train_dis:
            best_train_dis = avg_dis


        #valid
        grnet.eval()
        for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(valid_data_loader):
            if device:
                the_data = the_data.to(device)
                ground_truth_fine = ground_truth_fine.to(device)
                ground_truth_coarse = ground_truth_coarse.to(device)
            coarse, fine = grnet(the_data)
            dis_fine1, dis_fine2, _, _ = chamLoss(fine, ground_truth_fine)
            dis_fine = torch.mean(dis_fine1) + torch.mean(dis_fine2)
            dis_coarse1, dis_coarse2, _, _ = chamLoss(coarse, ground_truth_coarse)
            dis_coarse = torch.mean(dis_coarse1) + torch.mean(dis_coarse2)
            dis = dis_fine + 0.5 * dis_coarse

            total_valid_dis += dis.item()
            total_valid_num += 1
            print(' batch {}, dis: {}'.format(i+1, dis.item() * 10000))
        avg_dis = (total_valid_dis / total_valid_num) * 10000
        if avg_dis < best_valid_dis:
            best_valid_dis = avg_dis
            torch.save(grnet.state_dict(), 'best_grnet.pt')

if __name__ == "__main__":
    print('getting device...', end='')
    DEVICE = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print('device got')
    code, train_dis, valid_dis = train_net(device = DEVICE, epochs = 100)
    file = open(RESULT_PATH, "a")
    file.write('dataset:{} train_dist:{} val_dist:{}\n'.format(code, train_dis, valid_dis))
    file.close()