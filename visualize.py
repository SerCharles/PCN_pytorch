import torch
from data import MyDataset
from models.PCN import PCN
from torch.utils.data import DataLoader
import os
import open3d
import numpy as np

#DATA_PATH = '/home/shenguanlin/shapenet/train'
DATA_PATH = 'E:\\dataset\\shapenet\\train'
TYPE_CODE = '04256520'
BATCH_SIZE = 8



    
def train():
    print('getting dataset')
    dataset = MyDataset(DATA_PATH, TYPE_CODE)
    print('getting dataloader')
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print('data got!')

    for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader):
        model = PCN()
        model.load_state_dict(torch.load('best.pt',  map_location='cpu'))
        model.eval()
        result_coarse, result_fine = model(the_data)

        result_fine_np = result_fine[0].detach().numpy()

        result_fine_pcd = open3d.geometry.PointCloud()

        # From numpy to Open3D
        result_fine_pcd.points = open3d.utility.Vector3dVector(result_fine_np)

        open3d.visualization.draw_geometries([result_fine_pcd])
        open3d.io.write_point_cloud('result_fine.ply', result_fine_pcd)
        return


def show():
    print('getting dataset')
    dataset = MyDataset(DATA_PATH, TYPE_CODE)
    print('getting dataloader')
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print('data got!')

    for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader):

        ground_truth_fine_np = ground_truth_fine[0].detach().numpy()

        ground_truth_fine_pcd = open3d.geometry.PointCloud()

        # From numpy to Open3D
        ground_truth_fine_pcd.points = open3d.utility.Vector3dVector(ground_truth_fine_np)

        open3d.visualization.draw_geometries([ground_truth_fine_pcd])
        open3d.io.write_point_cloud('ground_truth_fine.ply', ground_truth_fine_pcd)
        return

if __name__ == "__main__":
    train()
    #show()
