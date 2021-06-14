"""
Declare all constants used for this module
"""
import math



VELOCITY_SKIP_FRAME = 4
# VELOCITY_COSINE_THRES = math.cos(math.pi/12)
# DISTANCE_COSINE_THRES = math.cos(math.pi/12)
# POSITION_COSINE_THRES = math.cos(math.pi/12)

THRES_LEVEL = [
    math.cos(math.pi/60),
    math.cos(math.pi/36),
    math.cos(math.pi/18),
    math.cos(math.pi/15),
    math.cos(math.pi/12),
    math.cos(math.pi/10),
    math.cos(math.pi/9),
    ]

MAX_TRAJ_THRES_LEVEL = [
    math.cos(math.pi/36),
    math.cos(math.pi/20),
    math.cos(math.pi/15),
    math.cos(math.pi/12),
    math.cos(math.pi/10),
    math.cos(math.pi/9),
    ]
DISTANCE_THRES = 4.5

FOLLOW_STATE_THRES = 3

