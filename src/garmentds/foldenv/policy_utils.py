import numpy as np
from scipy.interpolate import CubicSpline, interp1d


def get_2d_rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])


def theta_from_2d_rotation_matrix(rot: np.ndarray) -> float:
    assert rot.shape == (2, 2), f"{rot.shape}"
    return np.arctan2(rot[1, 0], rot[0, 0])


def shape_match_xy(xy1: np.ndarray, xy2: np.ndarray, sample_num: int = 1000):
    """
    args:
        - xy1: [P, 2]
        - xy2: [P, 2]
    
    return:
        - rot: [2, 2]
        - trans: [2]
        - err: float
        rot @ xy1 + trans is close to xy2
    """
    assert xy1.shape == xy2.shape and xy1.shape[1] == 2 and xy2.shape[1] == 2, f"{xy1.shape}, {xy2.shape}"

    angle = np.arange(sample_num) * np.pi * 2 / sample_num
    rotation = get_2d_rotation_matrix(angle)
    rotation = rotation.transpose(2, 0, 1) # [N, 2, 2]
    xy2_rot = np.einsum('ilk,jl->ijk', rotation, xy2) # [N, P, 2]
    translation = np.mean(xy2_rot - xy1, axis=1) # [N, 2]
    sqrerr = (xy2_rot - xy1 - translation.reshape(sample_num, 1, 2)) ** 2 # [N, P, 2]
    sqrerr = np.mean(sqrerr.reshape(sample_num, -1), axis=1) # [N]
    best_sample = np.argmin(sqrerr)
    
    rot: np.ndarray = rotation[best_sample]
    trans: np.ndarray = np.einsum('ij,j->i', rot, translation[best_sample])
    err: float = sqrerr[best_sample]
    return rot, trans, err


def interpolate_bezier(data: np.ndarray, total_points: int, zero_grad_at_endpoints: bool = True, expand_length: float = 0.2):
    """
    data: [N, D]
    interpolate (N - 1) segments, by default do not include the start point
    """
    N, D = data.shape
    A = expand_length
    points_t = np.arange(N)
    if zero_grad_at_endpoints:
        points_t = np.concatenate([[-A * 2, -A], points_t, [N - 1 + A, N - 1 + A * 2]])
        data = np.pad(data, ((2, 2), (0, 0)), 'edge')

    t_dense = (1 + np.arange(total_points)) / total_points * (N - 1)
    result = []
    for d in range(D):
        points_x = data[:, d]
        x_dense = CubicSpline(points_t, points_x)(t_dense)
        result.append(x_dense)
    result = np.stack(result, axis=1)
    return result


if __name__ == "__main__":
    xy1 = np.array([[0, 0], [1, 1], [2, 2]])
    xy2 = np.array([[4, 0], [3, 1], [2, 2]])
    print(shape_match_xy(xy1, xy2))
