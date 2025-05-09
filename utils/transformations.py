import numpy as np
import random
import enum

class Direction(enum.Enum):
    PITCH = lambda angle : np.array([
        [np.cos(angle), 0, np.sin(angle)], 
        [0, 1, 0], 
        [-np.sin(angle), 0, np.cos(angle)]
        ])
    YAW = lambda angle : np.array([
        [np.cos(angle), -np.sin(angle), 0], 
        [np.sin(angle), np.cos(angle), 0], 
        [0, 0, 1]
        ])
    ROLL = lambda angle : np.array([
        [1, 0, 0], 
        [0, np.cos(angle), -np.sin(angle)], 
        [0, np.sin(angle), np.cos(angle)]
        ])

def choose_random_rotate(angle : float):
    choice = random.randint(0, 2)
    result = None
    if choice == 0:
        result = (Direction.PITCH(angle), "pitch")
    elif choice == 1:
        result = (Direction.YAW(angle), "yaw")
    else:
        result = (Direction.ROLL(angle), "roll")
    return result

    


def rotate_3d(volume: np.ndarray, direction: np.ndarray) -> np.ndarray:
    points = np.argwhere(volume == 1)
    rotated = (points @ direction.T).round().astype(np.int64)
    shape_max = np.array(volume.shape) - 1   
    rotated = np.clip(rotated, 0, shape_max)
    out = np.zeros_like(volume)
    out[rotated[:,0], rotated[:,1], rotated[:,2]] = 1

    return out