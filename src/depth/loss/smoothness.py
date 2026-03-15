from tinygrad.tensor import Tensor


def smoothness_loss(disparity_map: Tensor, image: Tensor) -> Tensor:
    # Normalize the disparity map
    disparity_map = disparity_map / disparity_map.mean(axis=(2, 3), keepdim=True)

    disparity_grad_x = disparity_map[:, :, :, 1:] - disparity_map[:, :, :, :-1]
    disparity_grad_y = disparity_map[:, :, 1:, :] - disparity_map[:, :, :-1, :]
    image_grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    image_grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]

    disparity_x = disparity_grad_x.abs() * (-1 * image_grad_x.abs()).exp()
    disparity_y = disparity_grad_y.abs() * (-1 * image_grad_y.abs()).exp()
    return (disparity_x.mean() + disparity_y.mean()) / 2
