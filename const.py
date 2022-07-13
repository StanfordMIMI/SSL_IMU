import os


CODE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CODE_PATH, '../../data/camargo/data_processed/')
AMBULATIONS = ['LevelGround', 'Treadmill', 'Stair', 'Ramp']
IMU_SEGMENT_LIST = ['foot', 'shank', 'thigh', 'trunk']

GRAVITY = 9.81
STEP_TYPES = STANCE, STANCE_SWING = range(2)
