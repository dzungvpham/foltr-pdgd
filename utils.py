import numpy as np


def generate_dp_gamma_noise(
        rng: np.random.Generator, sensitivity: float, epsilon: float,
        num_clients: int, output_shape: tuple[int, int]):
    """Generate difference of two Gamma (to generate Laplace).
    Used for distributed DP. See https://arxiv.org/pdf/2002.08423.pdf 
    """
    shape = 1/num_clients
    scale = sensitivity / epsilon
    gamma1 = rng.gamma(shape, scale, output_shape)
    gamma2 = rng.gamma(shape, scale, output_shape)
    return gamma1 - gamma2
