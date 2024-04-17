from PIL import Image
import os
import torch
from tqdm import tqdm
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset

dataset_path='/datasets/MSMT17/train'
json_name='/diffusion/MSMTcaptions/filewise/gender.json'
optional_threshold=-1
template='A photo of a {}.'
folder="./clip-vit-l"
device=3

with open('gender.txt','r') as f:
    raw=f.readlines()
    phrase=[s.strip() for s in raw]
    
model = CLIPModel.from_pretrained(folder).to(torch.device('cuda:'+str(device)))
processor = CLIPProcessor.from_pretrained(folder)
text_input=[]
for word in phrase:
    text_input.append(template.format(word))
class CDataset(Dataset):
    def __init__(self,path):
        with open('/datasets/MSMT17/list_train.txt','r') as f:
            lst=f.readlines()
        
        self.img_path=['/datasets/MSMT17/train/'+x.split(' ')[0] for x in lst]
        self.path=path
        self.t=T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
        ])
    
    def __getitem__(self,index):
        now_path=os.path.join(self.path,self.img_path[index])
        img=Image.open(now_path)
        img=img.crop((0,0,img.size[0],img.size[1]/2)) #Tune HERE to CROP the image. TODO: Tune other preprocessing methods.(normalize, resize, etc.)
        # img=self.t(img)
        return img,self.img_path[index]
    
    def __len__(self):
        return len(self.img_path)
    
dataset=CDataset(dataset_path)
# dataloader=DataLoader(dataset,batch_size)# it seems that it's better to handle images one by one, using default preprocessor
results={}
q=1
with torch.no_grad():
    for images,paths in tqdm(dataset):
        inputs = processor(text=text_input, images=images, return_tensors="pt", padding=True).to(torch.device('cuda:'+str(device)))
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image #(image_num, class_num)
        probs = logits_per_image.softmax(dim=1)
        selections=torch.argmax(probs,1).cpu().detach()
        if(probs.max()>optional_threshold):
            results[paths]=phrase[selections]
        else:
            results[paths]='nothing'

import json
with open(json_name,'w') as f:
    json.dump(results,f)
