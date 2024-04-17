import os
import shutil
from fastreid.utils.compute_dist import compute_cosine_distance
import numpy as np
import random
import torch
from sklearn.cluster import DBSCAN, KMeans

def ADMGriddle(dataset,dist,query_pids,gallery_pids,eps=0.51):
    all_pids=np.unique(query_pids)
    i=0
    print(dist.mean(),dist.std())
    to_remove=np.empty((0,))
    for pid in all_pids:
        select_query=(query_pids==pid)
        this_dist=dist[select_query,:][:,select_query]
        this_gallery=gallery_pids[select_query]
        assert (this_gallery==np.ones_like(this_gallery)*pid).all()
        this_index=select_query.nonzero()[0].squeeze()
        choosed_query=(this_dist<=eps)
        within_count=choosed_query.sum(axis=1)
        most_id=np.argmax(within_count) #int
        second=this_dist[most_id].argsort()[1]
        remove_=(1-choosed_query[most_id]).nonzero()[0]
        least_two=remove_!=second
        remove_=remove_[least_two]
        remove_index=this_index[remove_]
        to_remove=np.concatenate([to_remove,remove_index])
    print('start delete, {} will be removed'.format(len(to_remove)))
    for id in to_remove:
        p=dataset[int(id)]['img_paths']
        if os.path.exists(p):
            os.remove(p)
    print('finish')

def DBSCANGriddle(dataset,query_features,query_pids,gallery_pids):
    all_pids=np.unique(query_pids)
    i=0
    db=KMeans(2)
    to_remove=np.empty((0,))
    for pid in all_pids:
        select_query=(query_pids==pid)
        features=query_features[select_query]
        this_gallery=gallery_pids[select_query]
        assert (this_gallery==np.ones_like(this_gallery)*pid).all()
        this_index=select_query.nonzero()[0].squeeze()
        result=db.fit(features)
        labels = result.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        num_count=[]
        for i in range(n_clusters_):
            num_count.append(list(labels).count(i))
        max_label=num_count.index(max(num_count))
        remove_01=1-(labels==max_label)
        remove_index=remove_01.nonzero()[0]
        remove_index=this_index[remove_index]
        to_remove=np.concatenate([to_remove,remove_index])
    print('start delete')   
    for id in to_remove:
        p=dataset[int(id)]['img_paths']
        if os.path.exists(p):
            os.unlink(p)
    print('finish')

import numpy as np
def ISD(dist):
    min_dist=dist.min(1)
    np.save('dist.npy',min_dist)
    import matplotlib.pyplot as plt
    plt.hist(min_dist,50)
    plt.savefig('gendist.png')
    print("max {},mean {},std {}".format(min_dist.max(),min_dist.mean(),min_dist.std()))

def getIDCentroid(query_features,query_pids):
    all_pids=np.unique(query_pids)
    maxd=0
    all_d=np.array([])
    center_pids=[]
    center_features=[]
    for pid in all_pids:
        select_query=(query_pids==pid)
        features=query_features[select_query]
        center=features.mean(0)[None,...]
        distance=compute_cosine_distance(center,features).squeeze()
        maxd=max(maxd,distance.max())
        all_d=np.concatenate([all_d,distance])
        center_features.append(center[0])
        center_pids.append(pid)
    meand=all_d.mean()
    np.save('sysu_original.npy',all_d)
    torch.save(torch.stack(center_features),'sysu_train_center_features.pt')
    np.save('sysu_train_center_pids.npy',np.stack(center_pids))
    import matplotlib.pyplot as plt
    plt.hist(all_d,200)
    plt.savefig('dist.png')
    print("max {},mean {},std {}".format(maxd,meand,all_d.std()))

def CentroidFilter(dataset,query_features,query_pids,gallery_pids,k=8):
    all_pids=np.unique(query_pids)
    i=0
    to_remove=np.empty((0,))
    for pid in all_pids:
        select_query=(query_pids==pid)
        features=query_features[select_query]
        this_gallery=gallery_pids[select_query]
        assert (this_gallery==np.ones_like(this_gallery)*pid).all()
        this_index=select_query.nonzero()[0].squeeze()
        center=features.mean(axis=0)[None,...]
        distance=compute_cosine_distance(center,features).squeeze()#(1,num_q)
        sorted_idx=np.argsort(distance)[::-1] #descend
        remove_=sorted_idx[:k]
        remove_index=this_index[remove_]
        to_remove=np.concatenate([to_remove,remove_index])
    print('start delete, {} will be deleted.'.format(len(to_remove)))
    for id in to_remove:
        p=dataset[int(id)]['img_paths']
        if os.path.exists(p):
            os.remove(p)
    print('finish')
    
    
def tightClusterGriddle(dataset,dist,query_pids,gallery_pids,k=8):
    all_pids=np.unique(query_pids)
    i=0
    to_remove=np.empty((0,))
    for pid in all_pids:
        select_query=(query_pids==pid)
        this_dist=dist[select_query,:][:,select_query]
        this_gallery=gallery_pids[select_query]
        assert (this_gallery==np.ones_like(this_gallery)*pid).all()
        this_index=select_query.nonzero()[0].squeeze()
        sum_dist=this_dist.sum(axis=1)
        sorted_idx=np.argsort(sum_dist)[::-1] #descend
        k=random.randint(11,12)
        remove_=sorted_idx[:k]
        remove_index=this_index[remove_]
        to_remove=np.concatenate([to_remove,remove_index])
    print('start delete, {} will be deleted.'.format(len(to_remove)))
    for id in to_remove:
        p=dataset[int(id)]['img_paths']
        if os.path.exists(p):
            os.unlink(p)
    print('finish')
    