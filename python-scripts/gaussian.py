import numpy as np
import math 

class Gaussian:
    def __init__(self):
        self.A = np.matrix([])
        self.B = np.matrix([])
        self.C = np.matrix([])
        self.position = np.array([])
        self.normal = np.array([])
        self.coeff = 1.0
        self.normal_map_jacobian_u_i = np.matrix([])

    def invCovarianceMatrix(self , invSigmaR2):
        # Upper left
        invCov4D = np.zeros((4,4))
        invCov4D[0][0] = self.A.tolist()[0][0]
        invCov4D[1][0] = self.A.tolist()[1][0]
        invCov4D[0][1] = self.A.tolist()[0][1]
        invCov4D[1][1] = self.A.tolist()[1][1]

        # Upper right
        invCov4D[0][2] = self.B.tolist()[0][0]
        invCov4D[1][2] = self.B.tolist()[1][0]
        invCov4D[0][3] = self.B.tolist()[0][1]
        invCov4D[1][3] = self.B.tolist()[1][1]

        # Lower left
        trB = (-1) * self.normal_map_jacobian_u_i * invSigmaR2
        invCov4D[2][0] = trB.tolist()[0][0]
        invCov4D[3][0] = trB.tolist()[1][0]
        invCov4D[2][1] = trB.tolist()[0][1]
        invCov4D[3][1] = trB.tolist()[1][1]

        invCov4D[2][2] = self.C.tolist()[0][0]
        invCov4D[3][2] = self.C.tolist()[1][0]
        invCov4D[2][3] = self.C.tolist()[0][1]
        invCov4D[3][3] = self.C.tolist()[1][1]

        return np.linalg.inv(invCov4D)

    def evaluate(self, position, normal, invSigmaR2):
        x_i = np.array([self.position[0], self.position[1], self.normal[0], self.normal[1]]).transpose()
        x = np.array([position[0], position[1], normal[0], normal[1]]).transpose()
        
        inner = (x - x_i) @ self.invCovarianceMatrix(invSigmaR2) @ (x - x_i)
        return self.coeff * math.exp(-0.5 * inner)
        # return self.coeff * math.exp((-0.5) * (x - x_i) @ self.invCovarianceMatrix(invSigmaR2) @ (x - x_i))

