import cv2
import numpy as np 
import sys 
import math
np.set_printoptions(threshold=sys.maxsize)

h = 1
sigma_h = h / math.sqrt(8*math.log(2, 10))
sigma_r = 0.005

def dist(tuple1, tuple2):
    x = tuple1[0] - tuple2[0]
    y = tuple1[1] - tuple2[1]
    return x ** 2 + y ** 2 

def to_s(cartesian):
    return (cartesian[0], cartesian[2])

def G_i(c_i, u_i, normal_map):
    return lambda u, s : c_i * math.exp((-dist(u, u_i)) / (2*sigma_h**2)) * math.exp((-dist(s, to_s(normal_map[u_i[1]][u_i[0]]))) / (2 * sigma_r**2))

def sign(x):
    if x < 0:
        return -1
    elif x == 0:
        return 0
    return 1

def to_spherical(cartesian):
    theta = math.acos(cartesian[2])
    phi = sign(cartesian[1]) * math.acos(cartesian[0] / math.sqrt(cartesian[0] ** 2 + cartesian[1] ** 2))
    return (theta, phi)


def convert_normal_map_to_flat_elements(normal_map):
    dim_x = normal_map.shape[1]
    dim_y = normal_map.shape[0]

    G_i_functions = []
    u_i_coords = []
    for x in range(0, dim_x, h):
        for y in range(0, dim_y, h):
            u_i_coords.append((x,y))
            G_i_functions.append(G_i(1, (x,y), normal_map))
            
    # Loop over texel map
    resulting_img = np.zeros(normal_map.shape)
    for x in range(0, normal_map.shape[0]):
        for y in range(0, normal_map.shape[1]):
            if(x == 256 and y == 277):
                for i, g_i in enumerate(G_i_functions):
                    result = g_i((x,y), to_s(normal_map[x][y]))
                    
                    if(result != 0.0):
                        resulting_img[u_i_coords[i][0]][u_i_coords[i][1]] = 1
      
                cv2.imshow("impact", resulting_img)
                cv2.waitKey()
                return


def main():
    normal_map = cv2.imread("NormalMap.png") / 255
    cv2.imshow("NormalMap", normal_map)
    cv2.waitKey()
    convert_normal_map_to_flat_elements(normal_map)

main()