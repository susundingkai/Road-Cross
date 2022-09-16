from cmath import cos, sin
from email.policy import default
import json
import os
from os.path import join
import cv2
ROOT="./9_8"
JSONPATH="./intersection2.json"
GTPATH='./instances_train.json'
DEFAULT_RAD=20
RAD_PRICIOUS=5
intersections={}

def checkInter(pos,bbox,clipSize):
    boxCenter=(bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2)
    _x,_y=boxCenter
    y,x=pos
    if(_x>x-clipSize//2 and _x<x+clipSize//2 and _y>y-clipSize//2 and _y<y+clipSize//2):
        return True
    return False
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
    for intersection in intersections:
        gtAnns=[] 
        path=intersection['path'].split('\\')[-1]
        imageName=path.replace('mask','').replace('_','')
        for tmp in gtImages:
            if(tmp['file_name']==imageName):
                imageId=tmp['id']
        for tmp in gtAnns:
            if(tmp['id']==imageId):
                gtAnns.append({'bbox':tmp['bbox'],'category_id':tmp['category_id']}) # x,y,width,height
        path=join(ROOT,path)
        inter=intersection['inter']
        image=cv2.imread(path)
        for pos in inter:
            if checkInter(pos, image)