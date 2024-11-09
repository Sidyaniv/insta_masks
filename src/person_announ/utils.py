import cv2
import numpy as np
from typing import List, Tuple 


def preparate_points(coords_list: List[int]) -> List[List[str]]:
    if len(coords_list) % 2 != 0:
        raise Exception("The length of the list must be even")
    
    list_of_points = []
    coordinates = np.array(coords_list,np.int16)
    x_indexes = range(0, len(coordinates) - 1, 2)
    for iter in x_indexes:
        list_of_points.append([coordinates[iter], coordinates[iter + 1]])   
    return list_of_points


def get_avg_face_point(points: list) -> Tuple[int]: #[4]
    average_x = np.average(np.array(points[::2], np.int16))
    average_y = np.average(np.array(points[1::2], np.int16))
 
    return round(average_x), round(average_y)


if __name__ == '__main__': 
    preparate_points([4,2,5,2,4,2,66,74,765,87])