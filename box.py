from cmath import cos, sin
from email.policy import default
import json
import os
from os.path import join
import cv2
from tqdm import tqdm
ROOT="./9_8"
JSONPATH="./intersection2.json"
GTPATH='./instances_train.json'
DEFAULT_RAD=20
RAD_PRICIOUS=5
intersections={}

def checkInter(pos,bboxList,clipSize):
    inter=[]
    for _bbox in bboxList:
        bbox,id=_bbox['bbox'],_bbox['category_id']
        boxCenter=(bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2)
        _x,_y=boxCenter
        y,x=pos
        if(_x>x-clipSize//2 and _x<x+clipSize//2 and _y>y-clipSize//2 and _y<y+clipSize//2):
            inter.append(bbox)
    return inter,id
def getRect(pos,imgSize,bbox):
    y,x=pos
    H,W=imgSize
    boxCenter=(bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2)
    _W,_H=bbox[2],bbox[3]
    _x,_y=boxCenter 
    wl,wr,ht,hb=124,125,124,125
    if(x<124): wl=W
    if(x+125>W): wr=W-x
    if(y<124): ht=H
    if(y+125>H): hb=H-y    
    _wl,_wr,_ht,_hb=min(wl,_W//2),min(wr,_W//2),min(ht,_H//2),min(hb,_H//2)
    return [_x-_wl,_y-_ht,_wl+_wr+1,_ht+_hb+1]
    
def test():
    images=[]
    annotations=[]
    categories=[{"id": 1, "name": "uncross"}, {"id": 2, "name": "cross"}]
    with open(JSONPATH,"r") as fp:
        intersections=json.load(fp)
    with open(GTPATH,'r') as fp:
        gtFile=json.load(fp)
    gtImages=gtFile['images']
    gtAnn=gtFile['annotations']
    annId=0
    imageId=0
    for intersection in tqdm(intersections):
        bboxList=[] 
        inter=intersection['inter']
        path=intersection['path'].split('\\')[-1]
        imageName=path.replace('mask','').replace('_','')
        path=join(ROOT,path)
        image=cv2.imread(path)
        H,W,_=image.shape
        for tmp in gtImages:
            if(tmp['file_name']==imageName):
                imageId=tmp['id']
        for tmp in gtAnn:
            if(tmp['id']==imageId):
                bboxList.append({'bbox':tmp['bbox'],'category_id':tmp['category_id']}) # x,y,width,height
        for pos in inter:
            images.append({"file_name": imageName.split('.')[0]+"_{}_{}".format(pos[1],pos[0]), "height": H, "width": W, "id": imageId})
            bboxList,category_id=checkInter(pos,bboxList,256)
            for bbox in bboxList:
                rect=getRect(pos,(H,W),bbox)
                annotations.append({"area": rect[2]*rect[3], "iscrowd": 0, "image_id": imageId, "bbox": rect, "category_id": category_id, "id": annId, "ignore": 0, "segmentation": []})
                annId+=1
            imageId+=1
        with open('newAnn.json','w')  as fp:
            json.dump({'images':images,'annotations':annotations,'categories':categories})