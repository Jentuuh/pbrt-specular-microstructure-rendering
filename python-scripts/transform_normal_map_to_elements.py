import cv2
import numpy as np 
import sys 
import math
from gaussian import Gaussian
np.set_printoptions(threshold=sys.maxsize)

h = 1
sigma_h = h / math.sqrt(8*math.log(2, 10))
sigma_r = 0.005

def dist_squared(tuple1, tuple2):
    x = tuple1[0] - tuple2[0]
    y = tuple1[1] - tuple2[1]
    return x ** 2 + y ** 2 

def delta_vector(vector1, vector2):
    print("delta_vector", vector2[0] - vector1[0], vector2[1] - vector1[1])
    return (vector2[0] - vector1[0], vector2[1] - vector1[1])

def vector_squared(vector):
    print("vector_squared:", vector[0] ** 2 + vector[1] ** 2)
    return vector[0] ** 2 + vector[1] ** 2

def product_matrix_vector(matrix, vector):
    np_vector = np.array([vector[0], vector[1]])
    product = matrix @ np_vector
    product = product.tolist()
    print("product_matrix_vector: ", product)
    return (product[0][0], product[0][1])

def to_s(cartesian):
    return (cartesian[0], cartesian[2])

def G_i(c_i, u_i, normal_map):
    return lambda u, s : c_i * math.exp((-dist_squared(u, u_i)) / (2*sigma_h**2)) * math.exp((-dist_squared(s, to_s(normal_map[u_i[1]][u_i[0]]))) / (2 * sigma_r**2))

def G_i_curved(c_i, u_i, J, normal_map):
    return lambda u, s : c_i * math.exp((-dist_squared(u, u_i)) / (2*sigma_h**2)) * math.exp((vector_squared(delta_vector(delta_vector(sample_normal_map(normal_map, u_i), s), product_matrix_vector(J, delta_vector(u_i, u))))) / (2 * sigma_r**2))

def sample_normal_map(normal_map, uv):
    w = normal_map.shape[0]
    h = normal_map.shape[1]

    x = int(uv[0] * float(w))
    y = int(uv[1] * float(h))
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))

    st = (uv[0] * float(w) - x, uv[1] * float(h) - y)
    p1 = to_s(normal_map[x][y])
    p2 = to_s(normal_map[max(0, min(w - 1, x + 1))][y])
    l1 = ((1.0 - st[0]) * p1[0] + st[0] * p2[0], (1.0 - st[0]) * p1[1] + st[0] * p2[1])

    p3 = to_s(normal_map[x][max(0, min(h - 1, y + 1))])
    p4 = to_s(normal_map[max(0, min(w - 1, x + 1))][max(0, min(h - 1, y + 1))])
    l2 = ((1.0 - st[0]) * p3[0] + st[0] * p4[0], (1.0 - st[0]) * p3[1] + st[0] * p4[1])

    # Bilinear interpolation
    return np.array([(1.0 - st[1]) * l1[0] + st[1] * l2[0], (1.0 - st[1]) * l1[1] + st[1] * l2[1]])

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

def sample_normal_map_jacobian(normal_map, uv):
    hX = 1.0 / normal_map.shape[0]
    hY = 1.0 / normal_map.shape[1]

    # Sobel filter
    x_diff = np.array([0.0, 0.0])
    x_diff += sample_normal_map(normal_map, uv + np.array([hX, hY]) * 1.0)
    x_diff += sample_normal_map(normal_map, uv + np.array([hX, 0]) * 2.0)
    x_diff += sample_normal_map(normal_map, uv + np.array([hX, -hY]) * 1.0)

    x_diff += sample_normal_map(normal_map, uv - np.array([hX, hY]) * -1.0)
    x_diff += sample_normal_map(normal_map, uv - np.array([hX, 0]) * -2.0)
    x_diff += sample_normal_map(normal_map, uv - np.array([hX, -hY]) * -1.0)

    y_diff = np.array([0.0, 0.0])
    y_diff += sample_normal_map(normal_map, uv + np.array([hX, hY]) * 1.0)
    y_diff += sample_normal_map(normal_map, uv + np.array([0, hY]) * 2.0)
    y_diff += sample_normal_map(normal_map, uv + np.array([-hX, hY]) * 1.0)

    y_diff += sample_normal_map(normal_map, uv - np.array([hX, hY]) * -1.0)
    y_diff += sample_normal_map(normal_map, uv - np.array([0, hY]) * -2.0)
    y_diff += sample_normal_map(normal_map, uv - np.array([-hX, hY]) * -1.0)

    x_diff /= (8.0 * hX)
    y_diff /= (8.0 * hY)

    # Jacobian
    return np.matrix([[x_diff[0], x_diff[1]], [y_diff[0], y_diff[1]]])

def curved_elements_4d_ndf(normal_map, width, height):
    mX = normal_map.shape[0]
    mY = normal_map.shape[1]
    gaussians = []

    sigmaH2 = sigma_h * sigma_h
    sigmaR2 = sigma_r * sigma_r

    invSigmaH2 = 1.0 / sigmaH2
    invSigmaR2 = 1.0 / sigmaR2

    for i in range(0, mX * mY):
        x = float(i % mX)
        y = float(i) / float(mX)

        u_i = np.array([x/float(mX), y/float(mY)])

        data = Gaussian()

        data.position = u_i
        data.normal = sample_normal_map(normal_map, u_i)

        jacobian = sample_normal_map_jacobian(normal_map, u_i)
        tr_jacobian = np.transpose(jacobian)

        data.normal_map_jacobian_u_i = jacobian
        data.A = ((tr_jacobian @ jacobian) * invSigmaR2) + (np.identity(2) * invSigmaH2)
        data.B = (-1) * tr_jacobian * invSigmaR2
        data.C = (np.identity(2) * invSigmaR2)
     

        # Upper left
        invCov4D = np.zeros((4,4))
        invCov4D[0][0] = data.A.tolist()[0][0]
        invCov4D[1][0] = data.A.tolist()[1][0]
        invCov4D[0][1] = data.A.tolist()[0][1]
        invCov4D[1][1] = data.A.tolist()[1][1]

        # Upper right
        invCov4D[0][2] = data.B.tolist()[0][0]
        invCov4D[1][2] = data.B.tolist()[1][0]
        invCov4D[0][3] = data.B.tolist()[0][1]
        invCov4D[1][3] = data.B.tolist()[1][1]

        # Lower left
        trB = (-1) * jacobian * invSigmaR2
        invCov4D[2][0] = trB.tolist()[0][0]
        invCov4D[3][0] = trB.tolist()[1][0]
        invCov4D[2][1] = trB.tolist()[0][1]
        invCov4D[3][1] = trB.tolist()[1][1]

        invCov4D[2][2] = data.C.tolist()[0][0]
        invCov4D[3][2] = data.C.tolist()[1][0]
        invCov4D[2][3] = data.C.tolist()[0][1]
        invCov4D[3][3] = data.C.tolist()[1][1]

        det = np.linalg.det(np.linalg.inv(invCov4D) * 2.0 * math.pi)

        if(det <= 0.0):
            print("det negative!")
            data.coeff = 0.0
        else:
            data.coeff = h * (h / math.sqrt(det))
        
        gaussians.append(data)
    
    for x in range(0, normal_map.shape[0]):
        for y in range(0, normal_map.shape[1]):
            if(x == 0 and y == 0):
                for i, g in enumerate(gaussians):
                    uv = np.array([float(x) / float(normal_map.shape[0]) ,float(y) / float(normal_map.shape[1])])
                    print(g.evaluate(uv, sample_normal_map(normal_map, uv), invSigmaR2))

    

def curved_elements_integration(width, height, mX, mY, gaussians):
    ndf = np.zeros(width * height)

    region_center = np.array([256, 256])
    region_size = np.array([256, 256])
    origin = region_center - (region_size * 0.5)

    footprint_radius = region_size[0] * 0.5 / float(mX)
    sigma_p = footprint_radius * 0.5

    footprint_cov_inv = np.linalg.inv(np.identity() * (sigma_p ** 2))
    footprint_mean = (origin + (region_size * 0.5)) * np.array([1.0 / float(mX), 1.0 / float(mY)])





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


def convert_normal_map_to_curved_elements(normal_map):
    dim_x = normal_map.shape[1]
    dim_y = normal_map.shape[0]

    G_i_functions = []
    u_i_coords = []
    for x in range(0, dim_x, h):
        for y in range(0, dim_y, h):
            u_i_coords.append((x,y))
            J = sample_normal_map_jacobian(normal_map, (float(x) / float(dim_x), float(y) / float(dim_y)))
            if(x == 256 and y == 277):
                print(J)
            G_i_functions.append(G_i_curved(1, (float(x) / float(dim_x), float(y) / float(dim_y)), J, normal_map))

    # Loop over texel map
    resulting_img = np.zeros(normal_map.shape)
    for x in range(0, normal_map.shape[0]):
        for y in range(0, normal_map.shape[1]):
            if(x == 256 and y == 277):
                for i, g_i in enumerate(G_i_functions):
                    result = g_i((float(x) / float(dim_x), float(y) / float(dim_y)), to_s(normal_map[x][y]))
                    
                    if(result != 0.0):
                        resulting_img[u_i_coords[i][0]][u_i_coords[i][1]] = 1
      
                cv2.imshow("impact", resulting_img)
                cv2.waitKey()
                return
def main():
    normal_map = cv2.imread("NormalMap.png") / 255
    cv2.imshow("NormalMap", normal_map)
    cv2.waitKey()
    # convert_normal_map_to_flat_elements(normal_map)
    # convert_normal_map_to_curved_elements(normal_map)
    curved_elements_4d_ndf(normal_map, 512, 512)
main()