# 伪代码

```
 from utils.normalization import *
 
 for idx, (taxonomy_ids, model_ids, data) in enumerat (train_dataloader):

    #Normalization
    data, centroid, furthest_distance=normalize_point_cloud(data) 
    data.cuda() #put it behind the normalization to save the space in gpu
    train(data)
    loss



 for idx, (taxonomy_ids, model_ids, data,gt) in enumerat (val_dataloader):
    #Normalization
    data, centroid, furthest_distance=normalize_point_cloud(data) 
    data.cuda()
    
    completed_data=valiation(data)
    centroid.cuda()
    furthest_distance.cuda()

    #UnNormalization

    completed_data*=furthest_distance
    completed_data+=centroid

    del centroid
    del furthest_distance

    Loss=get_loss(completed_data,gt)

```
