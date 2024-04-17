n=10
p=0.4
t=300
id_per_group=10
img_per_id=10
min_camera_num=0
reset=True
dataset_path='/datasets/CUHK-SYSU/cropped_image'
import json
import random
import shutil
from collections import defaultdict
import os
import re

def get_gpu_memory():
    import subprocess
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split())) # ['memory.free [MiB]', ' 11306 MiB', ' 11306 MiB', ' 11306 MiB']
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info) if i != 0]
    return memory_free_values

if reset:
    if os.path.exists('/diffusionReid/PIS'):
        shutil.rmtree('/diffusionReid/PIS')
    os.makedirs('/diffusionReid/PIS/dataset')

with open('/diffusion/SYSUcaptions/hair.json','r') as f:
    hair=json.load(f)
    
with open('/diffusion/SYSUcaptions/gender.json','r') as f:
    gender=json.load(f)
    
with open('/diffusion/SYSUcaptions/upper_wear.json','r') as f:
    uw=json.load(f)  

with open('/diffusion/SYSUcaptions/upper_wear_color.json') as f:
    uwc=json.load(f)
    
with open('/diffusion/SYSUcaptions/lower_wear_color.json') as f:
    lwc=json.load(f)
    
with open('/diffusion/SYSUcaptions/lower_wear.json') as f:
    lw=json.load(f)
    
#get pid-label pair from full market metadata
# meta_label={}
# with open(os.path.join(dataset_path,'metadata.jsonl'),'r') as f:
#     for l in f.readlines():
#         js=json.loads(l)
#         id=js['file_name'].split('_')[0]
#         meta_label[id]=js['label']


all_img_name=os.listdir(dataset_path)
prefix=[s.split('_')[0] for s in all_img_name]
for times in range(t):
    total=0
    pattern=re.compile(r'(p\d+)_n(\d)')
    if not os.path.exists('/diffusionReid/PIS/{}'.format(times)):
        os.mkdir('/diffusionReid/PIS/{}'.format(times))
    while True:
        total+=1
        ids=random.sample(prefix,n)
        uwc_types={}
        lw_types={}
        for id in ids:
            try:
                uwc_types[uwc[id]]=uwc_types.get(uwc[id],0)+1
                lw_desc=lwc[id]+' '+lw[id]
                lw_types[lw_desc]=lw_types.get(lw_desc,0)+1
            except:
                continue
        if max(uwc_types.values())>n*p or max(lw_types.values())>n*p:
            continue
        
        id_files=defaultdict(list)
        if min_camera_num:
            camera_has_id=defaultdict(set)
        for file_name in all_img_name:
            result=pattern.search(file_name)
            if not result:
                continue
            start,camera=result.groups()
            if start in ids:
                id_files[start].append(file_name)
                if min_camera_num:
                    camera_has_id[camera].add(start)
        if min_camera_num:
            camera_list=[x for x,v in camera_has_id.items() if len(v)==len(ids)]
            if(len(camera_list)<min_camera_num):
                continue
        map(lambda x: prefix.remove(x),ids)
        break
    
    if min_camera_num:
        print('camera:',camera_list)
    print('{}, tried:{}'.format(times,total))
    
    check_bc=[]
    prompt={}
    prompt_text={}
    label={}
    for num,name in enumerate(ids):
        a='A {} with {} '.format(gender[name],hair[name])
        b='is wearing {} {} '.format(uwc[name],uw[name])
        c='and {} {}.'.format(lwc[name],lw[name])
        label[name]=num       
        prompt[name]=[a,b,c]
        prompt_text[name]=a+b+c
        check_bc.append(b+c)                
    with open('/diffusionReid/PIS/{}/prompts.json'.format(times),'w') as f:
        json.dump([prompt_text],f)
        
    all_prompts=[]
    for a_id in prompt.keys():
        for b_id in prompt.keys():
            if b_id==a_id:
                continue
            for c_id in prompt.keys():
                if c_id==b_id or c_id==a_id:
                    continue
                if ''.join([prompt[a_id][0],prompt[b_id][1],prompt[c_id][2]]) in prompt_text.values():
                    continue
                all_prompts.append([prompt[a_id][0],prompt[b_id][1],prompt[c_id][2],a_id,b_id,c_id])
    random.shuffle(all_prompts) 
    check_b=[]
    check_c=[]
    backup=[]
    j=0
    while j<id_per_group:
        if len(all_prompts)==0:
            all_prompts=backup
            check_b=[]
            check_c=[]
            backup=[]
        prompt_selected=all_prompts.pop()
        ok=False
        if prompt_selected[1] in check_b or prompt_selected[2] in check_c or prompt_selected[1]+prompt_selected[2] in check_bc:
            backup.append(prompt_selected)
        else:
            check_b.append(prompt_selected[1])
            check_c.append(prompt_selected[2])
            ok=True
        if not ok:
            continue
        prompt_selected.append(label[prompt_selected[-1]])#label of lower wear , best for keeping consistency.maybe not best for privacy protection.
        name='/diffusionReid/PIS/{}/{}'.format(times,j)
        if not os.path.exists(name):
            os.makedirs(name)
        with open(name+'/prompt.json','w') as f:
            json.dump(prompt_selected,f)
        j+=1
            
    if not min_camera_num:
        camera_list=['0']
    for cam in camera_list:
        shutil.rmtree('/diffusionReid/PIS/dataset')
        os.mkdir('/diffusionReid/PIS/dataset')
        cam_str='c'+cam
        for id,files in id_files.items():
            count=0
            for file in files:
                if(cam_str in file or min_camera_num==0):
                    shutil.copy(os.path.join(dataset_path,file),os.path.join('/diffusionReid/PIS/dataset',file))
        
        with open('/diffusionReid/PIS/dataset/metadata.jsonl','w') as g:
            for img in os.listdir('/diffusionReid/PIS/dataset'):
                if not (img.endswith('jpg') or img.endswith('png')):
                    continue
                cid=img.split('_')[0]
                cprompt=prompt_text[cid]
                output = {"file_name": img, "text": cprompt,'label':label[cid]}
                g.write(json.dumps(output)+'\n')
        
        if reset:
            if os.system('bash /diffusionReid/train.sh'):
                print('error of training. Maybe run out of memory.')
                continue
           #cam modified for SYSU 
        if os.system('python /diffusionReid/sample_generation.py {} {} {} {}'.format(id_per_group,img_per_id,0,times)):
            print('error of sampling.')