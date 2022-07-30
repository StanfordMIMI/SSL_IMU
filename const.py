import os


CODE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CODE_PATH, 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/camargo/data_processed/')
AMBULATIONS = ['LevelGround', 'Treadmill', 'Stair', 'Ramp']
IMU_SEGMENT_LIST = ['foot', 'shank', 'thigh', 'trunk']

GRAVITY = 9.81
STEP_TYPES = STANCE, STANCE_SWING = range(2)
IMU_SAMPLE_RATE = 200
EMG_SAMPLE_RATE = 1000

MISSING_TRIALS = {
    'AB07': ['levelground_ccw_fast_01_05'],
    'AB08': ['stair_1_r_01_01', 'stair_1_r_01_02', 'stair_1_r_01_03', 'stair_1_r_01_04', 'stair_1_r_01_05'],
    'AB09': ['levelground_ccw_normal_01_05'],
    'AB11': ['levelground_ccw_fast_01_05', 'ramp_6_r_01_05'],
    'AB13': ['LevelGround_cw_normal_01_05', 'LevelGround_cw_slow_01_05', 'Stair_1_R_01_05'],
    'AB14': ['LevelGround_ccw_slow_01_05'],
    'AB15': ['LevelGround_ccw_normal_01_05', 'LevelGround_cw_normal_01_04', 'LevelGround_cw_normal_01_05',
             'LevelGround_cw_slow_01_03', 'LevelGround_cw_slow_01_04', 'LevelGround_cw_slow_01_05'],
    'AB21': ['LevelGround_cw_fast_01_04', 'LevelGround_cw_fast_01_05'],
    'AB24': ['Ramp_6_L_01_04', 'Ramp_6_L_01_05'],
}

