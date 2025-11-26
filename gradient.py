import numpy as np
import cv2
import math

class GradientCompute:
    @staticmethod
    def compute_magnitude(fx, fy, scale_factor=255):
        
        M = np.sqrt(fx.astype(np.float64)**2 + fy.astype(np.float64)**2)
        M = M / scale_factor
        M = cv2.normalize(M, None, 0, 255, cv2.NORM_MINMAX)
        return M.astype(np.uint8)

    @staticmethod
    def compute_direction(fx, fy):
        theta_deg = np.zeros_like(fx, dtype=np.float32)
        for i in range(fx.shape[0]):
            for j in range(fx.shape[1]):
                angle_rad = math.atan2(fy[i, j], fx[i, j])   # (-pi, pi]
                angle_deg = math.degrees(angle_rad)          # (-180, 180]
                angle_deg = angle_deg + 180                  # shift to (0, 360)
                if angle_deg >= 360:
                    angle_deg -= 360
                theta_deg[i, j] = angle_deg
        return theta_deg

    @staticmethod
    def quantize_direction(theta_deg):
        
        #Quantize gradient direction into 4, according to table
        quad_dir = np.zeros_like(theta_deg, dtype=np.uint8)

        # Horizontal
        mask0 = ((theta_deg >= 0) & (theta_deg < 22.5)) | \
                ((theta_deg >= 157.5) & (theta_deg < 202.5)) | \
                ((theta_deg >= 337.5) & (theta_deg <= 360))
        quad_dir[mask0] = 0

        # +45 diagonal
        mask1 = ((theta_deg >= 22.5) & (theta_deg < 67.5)) | \
                ((theta_deg >= 202.5) & (theta_deg < 247.5))
        quad_dir[mask1] = 1

        # Vertical
        mask2 = ((theta_deg >= 67.5) & (theta_deg < 112.5)) | \
                ((theta_deg >= 247.5) & (theta_deg < 292.5))
        quad_dir[mask2] = 2

        # -45 diagonal
        mask3 = ((theta_deg >= 112.5) & (theta_deg < 157.5)) | \
                ((theta_deg >= 292.5) & (theta_deg < 337.5))
        quad_dir[mask3] = 3

        return quad_dir

    @staticmethod
    def non_maximum_suppression(M, theta_quantized):
        
        #Applying Non-Maximum Suppression to thin edges
        
        rows, cols = M.shape
        suppressed = np.zeros((rows, cols), dtype=np.uint8)

        for r in range(1, rows-1):
            for c in range(1, cols-1):
                direction = theta_quantized[r, c]

                if direction == 0:  # horizontal 
                    neighbors = [M[r, c-1], M[r, c+1]]
                elif direction == 1:  # +45 diagonal
                    neighbors = [M[r-1, c+1], M[r+1, c-1]]
                elif direction == 2:  # vertical 
                    neighbors = [M[r-1, c], M[r+1, c]]
                elif direction == 3:  # -45 diagonal
                    neighbors = [M[r-1, c-1], M[r+1, c+1]]
                else:
                    neighbors = [0, 0]

                if M[r, c] >= max(neighbors):
                    suppressed[r, c] = M[r, c]
                else:
                    suppressed[r, c] = 0

        return suppressed
