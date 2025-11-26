import numpy as np

class GaussianFilter:
    def __init__(self, sigma=0.5, T=0.3):
        self.sigma = sigma
        self.T = T
        self.N, self.sHalf, self.X, self.Y = self.calculate_filter_size()
        self.Gx, self.Gy, self.scale = self.calculate_gradient()

    def calculate_filter_size(self):
        if self.sigma < 0.5:
            raise ValueError("Sigma must be >= 0.5")
        if not (0 < self.T < 1):
            raise ValueError("T must be between 0 and 1")

        # Half mask size
        sHalf = int(round(np.sqrt(-np.log(self.T) * 2 * self.sigma**2)))
        # Full mask size
        N = 2 * sHalf + 1
        # Meshgrid
        Y, X = np.meshgrid(np.arange(-sHalf, sHalf+1),
                           np.arange(-sHalf, sHalf+1))
        return N, sHalf, X, Y

    def calculate_gradient(self):
        sHalf = self.N // 2
        Y, X = np.meshgrid(np.arange(-sHalf, sHalf+1),
                           np.arange(-sHalf, sHalf+1))

        # Gaussian
        G = np.exp(-(X**2 + Y**2) / (2 * self.sigma**2))

        # First derivatives
        Gx = -X / (self.sigma**2) * G
        Gy = -Y / (self.sigma**2) * G

        # Scale to integers
        scale_factor = 255
        Gx_scaled = np.round(Gx * scale_factor).astype(int)
        Gy_scaled = np.round(Gy * scale_factor).astype(int)

        return Gx_scaled, Gy_scaled, scale_factor
