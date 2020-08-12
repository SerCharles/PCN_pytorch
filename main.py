import torch
import chamfer3D.dist_chamfer_3D
from data import MyDataset
from models.PCN import PCN
from torch.utils.data import DataLoader
import os
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
DATA_PATH = '/home/shenguanlin/shapenet/train'
VALID_PATH = '/home/shenguanlin/shapenet/val'
RESULT_PATH = 'result.txt'
TYPE_CODE_LIST = []
TYPE_CODE = '02691156'
BATCH_SIZE = 8


print('getting device...', end='')
DEVICE = torch.device('cuda:8')
torch.cuda.empty_cache()
print('device got')

def EnumTypes(path):
    if not os.path.isdir(path):
        print('Error:"',path,'" is not a directory or does not exist.')
        return
    list_dirs = os.walk(path)
    for root, dirs, files in list_dirs:
        for d in dirs:
            TYPE_CODE_LIST.append(str(d))

    
def train(device=None, epochs=100):
    the_type_code = TYPE_CODE
    the_train_dis = 0
    the_val_dis = 0
    print('training start')
    model = PCN()
    print('getting dataset')
    dataset = MyDataset(DATA_PATH, TYPE_CODE)
    print('getting dataloader')
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print('data got!')
    lr = 0.0001
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs+1):
        print('============='*4, 'epoch: ', epoch, '============='*4)
        total_dis = 0
        total_num = 0
        for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader):
            if device:
                the_data = the_data.to(device)
                ground_truth_fine = ground_truth_fine.to(device)
                ground_truth_coarse = ground_truth_coarse.to(device)
            coarse, fine = model(the_data)
            dis_fine1, dis_fine2, _, _ = chamLoss(fine, ground_truth_fine)
            dis_fine = torch.mean(dis_fine1) + torch.mean(dis_fine2)
            dis_coarse1, dis_coarse2, _, _ = chamLoss(coarse, ground_truth_coarse)
            dis_coarse = torch.mean(dis_coarse1) + torch.mean(dis_coarse2)
            dis = dis_fine + 0.5 * dis_coarse
            optimizer.zero_grad()
            dis.backward()
            optimizer.step()

            total_dis += dis.item()
            total_num += 1
            print('epoch:[{}/{}] batch {}, dis: {}'.format(epoch, epochs, i+1, dis.item() * 10000))
        avg_dis = (total_dis / total_num) * 10000
        if(epoch == epochs):
            the_train_dis = avg_dis


    print('validating start')
    print('getting dataset')
    dataset = MyDataset(VALID_PATH, TYPE_CODE)
    print('getting dataloader')
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    total_dis = 0
    total_num = 0
    for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader):
        if device:
            the_data = the_data.to(device)
            ground_truth_fine = ground_truth_fine.to(device)
            ground_truth_coarse = ground_truth_coarse.to(device)
        coarse, fine = model(the_data)
        dis_fine1, dis_fine2, _, _ = chamLoss(fine, ground_truth_fine)
        dis_fine = torch.mean(dis_fine1) + torch.mean(dis_fine2)
        dis_coarse1, dis_coarse2, _, _ = chamLoss(coarse, ground_truth_coarse)
        dis_coarse = torch.mean(dis_coarse1) + torch.mean(dis_coarse2)
        dis = dis_fine + 0.5 * dis_coarse
        #optimizer.zero_grad()
        #dis.backward()
        #optimizer.step()

        total_dis += dis.item()
        total_num += 1
        print(' batch {}, dis: {}'.format(i+1, dis.item() * 10000))
    avg_dis = (total_dis / total_num) * 10000
    the_val_dis = avg_dis
    return the_type_code, the_train_dis, the_val_dis

if __name__ == "__main__":
    EnumTypes(DATA_PATH +'/gt')
    print(TYPE_CODE_LIST)
    try:
        os.remove(RESULT_PATH)
    except:
        pass
    for item in TYPE_CODE_LIST:
        TYPE_CODE = item
        code, train_dis, valid_dis = train(device=DEVICE, epochs=100)
        file = open(RESULT_PATH, "a")
        file.write(code + ' ' + str(train_dis) + ' ' + valid_dis +"\n")
        file.close()
