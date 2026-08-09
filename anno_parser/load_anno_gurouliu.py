# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import json

def load_annotation_gurouliu(annotation_path):
    if os.path.exists(annotation_path) == False:
        return None
    with open(annotation_path) as fp:
        annotations = json.load(fp)
    n=0
    region_dict = {}
    for annotation in annotations["features"]:
        c=0
        for annotationCoordinates in annotation['geometry']['coordinates']:
            c = 0
            num = len(annotationCoordinates)
            if num==1:
                continue
            coords = np.zeros((num, 2))
            for coordinate in annotationCoordinates:
                x_coord = coordinate[0]
                y_coord = coordinate[1]
                coords[c][0] = x_coord
                coords[c][1] = y_coord
                c += 1
            coords = coords[:c, :]
            region_dict[n]=coords
            n=n+1
    return region_dict
