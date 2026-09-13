import numpy as np
import matplotlib.pyplot as plt
from mpmath import zetazero

class PrimeSpectrumKernel:
    def __init__(self, num_zeros=500):
        print(f"Loading {num_zeros} non-trivial zeros of Riemann zeta...")
        self.zeros = [float(zetazero(n).imag) for n in range(1, num_zeros + 1)]
        
    def test_function_transform(self, t, f_type="gaussian", sigma=1.0):
        if f_type == "gaussian":
            return np.exp(- (t**2) / (2 * sigma**2))
        return np.ones_like(t)

    def evaluate_spectral_measure(self, t_range, f_type="gaussian", sigma=1.0):
        t_vals = np.linspace(t_range[0], t_range[1], 1000)
        f_tilde = self.test_function_transform(t_vals, f_type, sigma)

        spectrum_density = np.zeros_like(t_vals)
        for gamma in self.zeros:
            spectrum_density += np.exp(-((t_vals - gamma)**2) / 0.05)
            
        return t_vals, f_tilde * spectrum_density

    def plot_output(self):
        t_vals, result = self.evaluate_spectral_measure((0, 50))
        
        plt.figure(figsize=(10, 4))
        plt.plot(t_vals, result, color='black', lw=1.2)
        plt.title("Prime Spectrum Kernel Output", fontsize=10)
        plt.xlabel("t (Spectral Parameter)", fontsize=9)
        plt.ylabel(r"$\tilde{f}(t) d\nu_{spec}(t)$", fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("prime_spectrum_output.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    kernel = PrimeSpectrumKernel(num_zeros=300)
    kernel.plot_output()
