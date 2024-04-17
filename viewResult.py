import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import json
import random

_,path=sys.argv#path of generated images,e.g. PIS/9/4
img_num=2
dataset='/datasets/CUHK-SYSU/cropped_images/'
base_path='/diffusionReid/'
path=os.path.join(base_path,path)

json_path=os.path.join(path,'prompt.json')
with open(json_path,'r') as f:
    js=json.load(f)
prompt=''.join(js[:3])

json_path='/'.join(path.split('/')[:-1])
json_path+='/prompts.json'
with open(json_path,'r') as f:
    origin=json.load(f)[0]
id1=js[3]
id2=js[4]
id3=js[5]
p1=origin[id1]
p2=origin[id2]
p3=origin[id3]

fig, axs = plt.subplots(4, img_num, figsize=(15, 20))

all_img=os.listdir(dataset)
img1=random.sample([os.path.join(dataset,x) for x in all_img if x.startswith(id1)],img_num)
img2=random.sample([os.path.join(dataset,x) for x in all_img if x.startswith(id2)],img_num)
img3=random.sample([os.path.join(dataset,x) for x in all_img if x.startswith(id3)],img_num)
result=random.sample([os.path.join(path,x) for x in os.listdir(path) if not x.endswith('json')],img_num)
images=[*img1,*img2,*img3,*result]
titles=[id1+' '+p1,id2+' '+p2,id3+' '+p3,prompt]

for i in range(4):
    for j in range(img_num):
        img = Image.open(images[i*img_num + j])
        axs[i, j].imshow(img)
        axs[i, j].axis('off')
        if j == 1:
            axs[i, j].set_title(titles[i])

plt.tight_layout()
# plt.show()

import shutil
if os.path.exists(base_path+'viewresult'):
    shutil.rmtree(base_path+'viewresult')
os.mkdir(base_path+'viewresult')
for i in images:
    shutil.copy(i,base_path+'viewresult/')

plt.savefig(base_path+'viewresult/result.png')
