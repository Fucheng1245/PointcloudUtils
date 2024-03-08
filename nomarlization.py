
import torch
import open3d as o3d
import numpy as np

def normalize_point_cloud(input, centroid=None, furthest_distance=None):
    # input: (b, 3, n) tensor
    assert len(input.shape)==3, f"The input pointcloud's shape need (b,3,n) or (b,n,3), rather than {input.shape}"
    if input.shape[2]==3:
        check=0
        input=input.transpose(2,1)
    else :
        check=None
    if centroid is None:
        # (b, 3, 1)
        centroid = torch.mean(input, dim=-1, keepdim=True)
    # (b, 3, n)
    input = input - centroid
    if furthest_distance is None:
        # (b, 3, n) -> (b, 1, n) -> (b, 1, 1)
        furthest_distance = torch.max(torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True)[0]
    input = input / furthest_distance
    if check is not None:
        input=input.transpose(2,1)
        centroid=centroid.transpose(2,1)
    return input, centroid, furthest_distance

def printlogrange(pc):
    if pc.shape[1]==3:
        pc=pc.transpose(2,1)
    print(f"The data range of x is {torch.max(pc[:,:,0],dim=1)[0]},{torch.min(pc[:,:,0],dim=1)[0]}")
    print(f"The data range of y is {torch.max(pc[:,:,1],dim=1)[0]},{torch.min(pc[:,:,1],dim=1)[0]}")
    print(f"The data range of z is {torch.max(pc[:,:,2],dim=1)[0]},{torch.min(pc[:,:,2],dim=1)[0]}")
if __name__=="__main__":

    complete= o3d.io.read_point_cloud("/home_exp_2/fucheng.niu/Toothdata/Toothdata_raw/dataset/train/complete/0/3.pcd")
    complete_pc_np = np.asarray(complete.points)
    pc=torch.from_numpy(complete_pc_np)
    pc=torch.unsqueeze(pc,0)
    # pc=pc.transpose(2,1)
    print("The Raw DATA:")
    printlogrange(pc)
    pc, centroid, furthest_distance = normalize_point_cloud(pc)
    print("The Normarlization data:")
    printlogrange(pc)
    print(furthest_distance.shape)
    print(centroid.shape)

    upsampled_pcd = centroid + pc * furthest_distance




