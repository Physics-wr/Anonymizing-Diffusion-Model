from diffusers import StableDiffusionPipeline
import torch
import sys
import json
import random

_,id_per_group,img_per_id,camera_id,times=sys.argv
id_per_group=int(id_per_group)
img_per_id=int(img_per_id)
camera_id=int(camera_id)
times=int(times)

model_id = "/diffusionReid/PIS/model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,requires_safety_checker=False).to("cuda:0")

pipe.safety_checker=None

for pid_gen in range(id_per_group):
    f=open('/diffusionReid/PIS/{}/{}/prompt.json'.format(times,pid_gen),'r')
    pl=json.load(f)
    prompt=''.join(pl[:3])
    id=pl[-1]
    prompts=[prompt]*img_per_id
    class_labels=torch.tensor([int(id)],dtype=torch.int64).to('cuda:0')
    images=pipe(prompts,height=256,width=128,class_labels=class_labels).images
    if camera_id>=0:
        for i in range(img_per_id):
            images[i].save("/diffusionReid/PIS/{}/{}/{}_c{}.png".format(times,pid_gen,i,camera_id))
    else:
        for i in range(img_per_id):
            images[i].save("/diffusionReid/PIS/{}/{}/{}_c{}.png".format(times,pid_gen,i,random.randint(1,6)))   
            
del pipe
torch.cuda.empty_cache()














def useless():
    count_b={}
    count_c={}
    count_a={}
    check_ab=[]
    with open('/diffusion/PIS/{}/prompts.json'.format(times),'w') as f:
        prompt={}
        prompt_text={}
        for img in os.listdir('/diffusion/PIS/dataset'):
            name='_'.join(img.split('_')[:2])
            if not name in prompt.keys():
                a='A {} with {} '.format(gender[name],hair[name])
                b='is wearing {} {} '.format(uwc[name],uw[name])
                c='and {} {}.'.format(lwc[name],lw[name])
                count_a[a]=count_a.get(a,0)+1
                count_b[b]=count_b.get(b,0)+1
                count_c[c]=count_c.get(c,0)+1
                prompt[name]=[a,b,c]
                prompt_text[name]=a+b+c
                check_ab.append(a+b)                
        json.dump([camera_id,prompt_text,pid_dict],f)
    sorted_a=sorted(list(count_a.keys()),key=lambda x:count_a[x])
    sorted_b=sorted(list(count_b.keys()),key=lambda x:count_b[x])
    sorted_c=sorted(list(count_c.keys()),key=lambda x:count_c[x])
    
            # check_a=[]
        # check_bc=[]
        # for a_id in prompt.keys():
        #     for b_id in prompt.keys():
        #         if b_id==a_id:
        #             continue
        #         for c_id in prompt.keys():
        #             if c_id==b_id or c_id==a_id:
        #                 continue
        #             # score=len(sorted_a)-sorted_a.index(prompt[a_id][0])+len(sorted_b)-sorted_b.index(prompt[b_id][1])+len(sorted_c)-sorted_c.index(prompt[c_id][2])
        #             score=25
        #             if(gender[a_id]!=gender[b_id]):
        #                 score+=1
        #             syn_prompt=[score,prompt[a_id][0],prompt[b_id][1],prompt[c_id][2]]
        #             if syn_prompt[1] in check_a:
        #                 syn_prompt[0]-=check_a.count(syn_prompt[1])
        #             check_a.append(syn_prompt[1])
        #             if syn_prompt[2] in check_b:
        #                 syn_prompt[0]-=check_b.count(syn_prompt[2]) 
        #             check_b.append(syn_prompt[2])
        #             if syn_prompt[3] in check_c:
        #                 syn_prompt[0]-=check_c.count(syn_prompt[3])
        #             check_c.append(syn_prompt[3])
        #             if(syn_prompt[2]+syn_prompt[3] in check_bc):
        #                 syn_prompt[0]=0
        #             check_bc.append(syn_prompt[2]+syn_prompt[3])
        #             all_prompts.append(syn_prompt+[a_id,b_id,c_id])
        # all_prompts.sort(key=lambda x:x[0])

    def genImage(i,pipe,prompt,id,pid_gen,device):
        image = pipe(prompt,height=256,width=128,class_labels=torch.tensor([int(id)],dtype=torch.int64).to(device)).images[0]
        if camera_id>=0:
            image.save("/diffusionReid/PIS/{}/{}/{}_c{}.png".format(times,pid_gen,i,camera_id))
        else:
            image.save("/diffusionReid/PIS/{}/{}/{}_c{}.png".format(times,pid_gen,i,random.randint(1,8)))