from tinygrad.tensor import Tensor


def photometric_loss(value: Tensor, expected: Tensor, alpha: float = 0.85) -> Tensor:
    return (1 - alpha) * l1_loss(value, expected) + alpha * ssim_loss(value, expected)


def l1_loss(value: Tensor, expected: Tensor) -> Tensor:
    return (value - expected).abs().mean()


def ssim_loss(value: Tensor, expected: Tensor) -> Tensor:
    """Calculate structural similarity (ssim) loss."""
    c_1 = 0.01**2
    c_2 = 0.03**2

    mu_x = value.avg_pool2d((3, 3), stride=1, padding=1)
    var_x = (value**2).avg_pool2d((3, 3), stride=1, padding=1) - mu_x**2

    mu_y = expected.avg_pool2d((3, 3), stride=1, padding=1)
    var_y = (expected**2).avg_pool2d((3, 3), stride=1, padding=1) - mu_y**2

    sigma_xy = (value * expected).avg_pool2d((3, 3), stride=1, padding=1) - (mu_x * mu_y)

    ssim_num = (2 * mu_x * mu_y + c_1) * (2 * sigma_xy + c_2)
    ssim_den = (mu_x**2 + mu_y**2 + c_1) * (var_x + var_y + c_2)
    ssim = ssim_num / ssim_den

    return ((1 - ssim) / 2).mean()
