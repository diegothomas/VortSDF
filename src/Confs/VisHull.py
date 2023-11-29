
from cmath import isnan, nan
import sys, os
import numpy as np
from tqdm import tqdm

def Load_Visual_Hull(data_name, dataset):
    visual_hull = [-0.6,-0.6,-0.6,0.6,0.6,0.6]

    if data_name == 'bmvs_dog':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.3
        visual_hull[0] = visual_hull[0] - 0.1 
        visual_hull[1] = visual_hull[1] - 0.3 
        visual_hull[2] = visual_hull[2] - 0.1 
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.3
        visual_hull[3] = visual_hull[3] + 0.0
        visual_hull[4] = visual_hull[4] - 0.5
        visual_hull[5] = visual_hull[5] + 0.1
    
    elif data_name == 'bmvs_man':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.3
        visual_hull[0] = visual_hull[0] - 0.4
        visual_hull[1] = visual_hull[1] - 0.0
        visual_hull[2] = visual_hull[2] - 0.1
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.3
        visual_hull[3] = visual_hull[3] + 0.3
        visual_hull[4] = visual_hull[4] + 0.0
        visual_hull[5] = visual_hull[5] + 0.2

    elif data_name == 'bmvs_sculpture':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.3
        visual_hull[0] = visual_hull[0] - 0.1
        visual_hull[1] = visual_hull[1] - 0.1
        visual_hull[2] = visual_hull[2] - 0.0
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.3
        visual_hull[3] = visual_hull[3] + 0.1
        visual_hull[4] = visual_hull[4] + 0.3
        visual_hull[5] = visual_hull[5] + 0.1

    
    elif data_name == 'bmvs_clock':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.4
        visual_hull[0] = visual_hull[0] - 0.2
        visual_hull[1] = visual_hull[1] - 0.1
        visual_hull[2] = visual_hull[2] - 0.2
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.4
        visual_hull[3] = visual_hull[3] + 0.1
        visual_hull[4] = visual_hull[4] + 0.1
        visual_hull[5] = visual_hull[5] + 0.1
        
    elif data_name == 'bmvs_stone':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.6
        visual_hull[0] = visual_hull[0] - 0.0
        visual_hull[1] = visual_hull[1] - 0.0
        visual_hull[2] = visual_hull[2] - 0.0
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.6
        visual_hull[3] = visual_hull[3] + 0.0
        visual_hull[4] = visual_hull[4] + 0.0
        visual_hull[5] = visual_hull[5] + 0.0
        
    elif data_name == 'bmvs_bear':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.2
        visual_hull[0] = visual_hull[0] - 0.0
        visual_hull[1] = visual_hull[1] - 0.0
        visual_hull[2] = visual_hull[2] - 0.2
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.2
        visual_hull[3] = visual_hull[3] + 0.2
        visual_hull[4] = visual_hull[4] + 0.0
        visual_hull[5] = visual_hull[5] - 0.2

    elif data_name == 'bmvs_jade':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.4
        visual_hull[0] = visual_hull[0] - 0.0
        visual_hull[1] = visual_hull[1] - 0.0
        visual_hull[2] = visual_hull[2] - 0.0
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.4
        visual_hull[3] = visual_hull[3] + 0.6
        visual_hull[4] = visual_hull[4] + 0.2
        visual_hull[5] = visual_hull[5] + 0.6

    elif data_name == 'bmvs_durian':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.2
        visual_hull[0] = visual_hull[0] - 0.4
        visual_hull[1] = visual_hull[1] - 0.7
        visual_hull[2] = visual_hull[2] - 0.4
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.2
        visual_hull[3] = visual_hull[3] + 0.4
        visual_hull[4] = visual_hull[4] + 0.5
        visual_hull[5] = visual_hull[5] + 0.4
        
    elif data_name == 'Barn':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.1
        visual_hull[0] = visual_hull[0] - 0.0
        visual_hull[1] = visual_hull[1] - 0.0
        visual_hull[2] = visual_hull[2] - 0.0
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.1
        visual_hull[3] = visual_hull[3] + 0.0
        visual_hull[4] = visual_hull[4] + 0.0
        visual_hull[5] = visual_hull[5] + 0.0
        
    elif data_name == 'Courthouse':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.1
        visual_hull[0] = visual_hull[0] - 0.0
        visual_hull[1] = visual_hull[1] - 0.0
        visual_hull[2] = visual_hull[2] - 0.0
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.1
        visual_hull[3] = visual_hull[3] + 0.0
        visual_hull[4] = visual_hull[4] + 0.0
        visual_hull[5] = visual_hull[5] + 0.0
        
    elif data_name == 'Family':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.1
        visual_hull[0] = visual_hull[0] - 0.0
        visual_hull[1] = visual_hull[1] - 0.0
        visual_hull[2] = visual_hull[2] - 0.0
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.1
        visual_hull[3] = visual_hull[3] + 0.0
        visual_hull[4] = visual_hull[4] + 0.0
        visual_hull[5] = visual_hull[5] + 0.0

    elif data_name == 'Fountain':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.1
        visual_hull[0] = visual_hull[0] - 0.0
        visual_hull[1] = visual_hull[1] - 0.0
        visual_hull[2] = visual_hull[2] - 0.0
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.1
        visual_hull[3] = visual_hull[3] + 0.0
        visual_hull[4] = visual_hull[4] + 0.0
        visual_hull[5] = visual_hull[5] + 0.0
        
    elif data_name == 'Character':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.1
        visual_hull[0] = visual_hull[0] - 0.0
        visual_hull[1] = visual_hull[1] - 0.0
        visual_hull[2] = visual_hull[2] - 0.0
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.1
        visual_hull[3] = visual_hull[3] + 0.0
        visual_hull[4] = visual_hull[4] + 0.0
        visual_hull[5] = visual_hull[5] + 0.0
        
    elif data_name == 'Statues':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.1
        visual_hull[0] = visual_hull[0] - 0.0
        visual_hull[1] = visual_hull[1] - 0.0
        visual_hull[2] = visual_hull[2] - 0.0
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.1
        visual_hull[3] = visual_hull[3] + 0.0
        visual_hull[4] = visual_hull[4] + 0.0
        visual_hull[5] = visual_hull[5] + 0.0
        
        
    elif data_name == 'Ignatius':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.1
        visual_hull[0] = visual_hull[0] - 0.0
        visual_hull[1] = visual_hull[1] - 0.0
        visual_hull[2] = visual_hull[2] - 0.0
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.1
        visual_hull[3] = visual_hull[3] + 0.0
        visual_hull[4] = visual_hull[4] + 0.0
        visual_hull[5] = visual_hull[5] + 0.0
        
    elif data_name == 'Meetingroom':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.1
        visual_hull[0] = visual_hull[0] - 0.2
        visual_hull[1] = visual_hull[1] - 0.0
        visual_hull[2] = visual_hull[2] - 0.0
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.1
        visual_hull[3] = visual_hull[3] + 0.0
        visual_hull[4] = visual_hull[4] + 0.3
        visual_hull[5] = visual_hull[5] + 0.0
    
    elif data_name == 'dtu_scan24':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.1
        visual_hull[0] = visual_hull[0] - 0.0
        visual_hull[1] = visual_hull[1] - 0.0
        visual_hull[2] = visual_hull[2] + 0.2
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.1
        visual_hull[3] = visual_hull[3] + 0.0
        visual_hull[4] = visual_hull[4] - 0.1
        visual_hull[5] = visual_hull[5] + 0.0
        
    elif data_name == 'dtu_scan37':
        visual_hull[0:3] = dataset.object_bbox_min[0:3] - 0.1
        visual_hull[0] = visual_hull[0] + 0.1
        visual_hull[1] = visual_hull[1] + 0.1
        visual_hull[2] = visual_hull[2] + 0.6
        visual_hull[3:6] = dataset.object_bbox_max[0:3] + 0.1
        visual_hull[3] = visual_hull[3] - 0.1
        visual_hull[4] = visual_hull[4] - 0.1
        visual_hull[5] = visual_hull[5] - 0.4
    
    else:
        print("WRONG DATA NAME")
        exit()
    return visual_hull