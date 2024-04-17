import json 
import os
import re
from tqdm import tqdm
from collections import defaultdict
gendata='/root/fast-reid/datasets/sysu'
dataset_path='/rt/SYSUimg'
all_img=os.listdir(gendata)

p=r'(\d+)_s\d+_(\d+)_(\d+)'
p=re.compile(p)
real=defaultdict(list)
for img in tqdm(all_img):
    gen_pid,folder,subfolder=p.search(img).groups()
    prompt=os.path.join(dataset_path,folder,subfolder,'prompt.json')
    with open(prompt,'r') as f:
        lst=json.load(f)
    id1,id2,id3=lst[3:-1]
    id1=id1[1:]
    id2=id2[1:]
    id3=id3[1:]
    if(not int(gen_pid) in real[int(id1)]):
        real[int(id1)].append(int(gen_pid))
    if(not int(gen_pid) in real[int(id2)]):
        real[int(id2)].append(int(gen_pid))
    if(not int(gen_pid) in real[int(id3)]):
        real[int(id3)].append(int(gen_pid))
    
with open('sysu_privacy.json','w') as f:
    json.dump(real,f)