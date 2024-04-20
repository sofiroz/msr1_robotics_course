#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Zoheb Abai
# Repo : https://github.com/ZohebAbai/mobile_sensing_robotics/

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh
from typing import List



def plot_gridmap(gridmap):
    gridmap = np.array(gridmap, dtype=np.float64)
    plt.figure()
    plt.imshow(gridmap, cmap='Greys',vmin=0, vmax=1)
    
def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size/res)), int(np.ceil(size/res))])
    return gridmap + 0.50

def world2map(pose, gridmap, map_res):
    origin = np.array(gridmap.shape)/2
    new_pose = np.zeros((pose.shape))
    new_pose[0:] = np.round(pose[0:]/map_res) + origin[0]
    new_pose[1:] = np.round(pose[1:]/map_res) + origin[1]
    return new_pose.astype(int)

def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr    

def ranges2points(ranges):
    # laser properties
    start_angle = -1.5708
    angular_res = 0.0087270
    max_range = 30
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    angles = np.linspace(start_angle, start_angle + (num_beams*angular_res), num_beams)[idx]
    points = np.array([np.multiply(ranges[idx], np.cos(angles)), np.multiply(ranges[idx], np.sin(angles))])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom

def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # covert to map frame
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2,:]
    return m_points

def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose  

def bresenham(x0: int, y0: int, x1: int, y1: int) -> bh.bresenham:
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l
    
def prob2logodds(prob):
    logoods = np.log(prob / (1 - prob))
    return logoods

def logodds2prob(log: float) -> float:
    return 1 / (1 + np.exp(-log))
    
def inv_sensor_model(start_cell: List[int], endpoint: List[int], prob_occ: float, prob_free:float) -> np.array:
    point_prob = []
    line = list(bresenham(start_cell[0], start_cell[1], endpoint[0], endpoint[1]))
    line.pop()
    point_prob.append(np.append(endpoint, prob_occ)) 
    
    for lcell in line:
        point_prob.append(np.append(lcell, prob_free))
        
    return point_prob
     

def grid_mapping_with_known_poses(gridmap, cell_poses, line_ranges, map_res, prob_occ, prob_free):
    poses = poses2cells(cell_poses, gridmap, map_res)
    log_gridmap = prob2logodds(gridmap)

    for i in range(line_ranges.shape[0]):
        line_cells = ranges2cells(line_ranges[i], cell_poses[i], log_gridmap, map_res).transpose()
        for endpoint in line_cells:
            sensor_model_out = inv_sensor_model([poses[i][0], poses[i][1]], endpoint, prob_occ, prob_free)
            
            for cell in sensor_model_out:
                log_gridmap[int(cell[0]),int(cell[1])] += prob2logodds(cell[2]) 

    prob_gridmap = logodds2prob(log_gridmap)        

    return prob_gridmap
        

        
        
