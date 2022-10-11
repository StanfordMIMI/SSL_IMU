import os

IMU_MOVI_SEGMENT_LIST = [
    'Hip', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightShoulder',
    'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'Head', 'Neck']
DICT_TRIAL_MOVI = {index: i for i, index in enumerate(['I1', 'I2', 'S1', 'S2'])}

CODE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CODE_PATH, 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/camargo/data_processed/')
RESULTS_PATH = 'D://SSL_training_results'
TRIAL_TYPES = ['LevelGround', 'Treadmill', 'Stair', 'Ramp']
IMU_CARMARGO_SEGMENT_LIST = ['foot', 'shank', 'thigh', 'trunk']

DICT_TRIAL_TYPE_ID = {type: i for i, type in enumerate(TRIAL_TYPES)}
DICT_LABEL = {'treadmill_walking': -1, 'walk': 0, 'idle': 1, 'stand-walk': 2, 'turn': 3, 'rampascent': 4, 'rampdescent': 5,
              'walk-ramp': 6, 'stairascent': 7, 'stairdescent': 8, 'walk-stair': 9}
DICT_AXIS_DIRECTION = {'x': 'medio-lateral', 'y': 'vertical', 'z': 'anterior-posterior'}

GRAVITY = 9.81
STANCE_V_GRF_THD = 10.
STEP_TYPES = STANCE, STANCE_SWING = range(2)
IMU_CARMARGO_SAMPLE_RATE = 200
GRF_CARMARGO_SAMPLE_RATE = EMG_CARMARGO_SAMPLE_RATE = 1000

FONT_SIZE_LARGE = 12
FONT_SIZE = 10
FONT_SIZE_SMALL = 8
FONT_DICT = {'fontsize': FONT_SIZE, 'fontname': 'Arial'}
FONT_DICT_LARGE = {'fontsize': FONT_SIZE_LARGE, 'fontname': 'Arial'}
FONT_DICT_SMALL = {'fontsize': FONT_SIZE_SMALL, 'fontname': 'Arial'}
FONT_DICT_X_SMALL = {'fontsize': 15, 'fontname': 'Arial'}
LINE_WIDTH = 2
LINE_WIDTH_THICK = 3

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


TRIAL_LIST = [
    'LevelGround_ccw_fast_01_01', 'LevelGround_ccw_fast_01_02', 'LevelGround_ccw_fast_01_03',
    'LevelGround_ccw_fast_01_04', 'LevelGround_ccw_fast_01_05', 'LevelGround_ccw_normal_01_01',
    'LevelGround_ccw_normal_01_02', 'LevelGround_ccw_normal_01_03', 'LevelGround_ccw_normal_01_04',
    'LevelGround_ccw_normal_01_05', 'LevelGround_ccw_slow_01_01', 'LevelGround_ccw_slow_01_02',
    'LevelGround_ccw_slow_01_03', 'LevelGround_ccw_slow_01_04', 'LevelGround_ccw_slow_01_05',
    'LevelGround_cw_fast_01_01', 'LevelGround_cw_fast_01_02', 'LevelGround_cw_fast_01_03', 'LevelGround_cw_fast_01_04',
    'LevelGround_cw_fast_01_05', 'LevelGround_cw_normal_01_01', 'LevelGround_cw_normal_01_02',
    'LevelGround_cw_normal_01_03', 'LevelGround_cw_normal_01_04', 'LevelGround_cw_normal_01_05',
    'LevelGround_cw_slow_01_01', 'LevelGround_cw_slow_01_02', 'LevelGround_cw_slow_01_03', 'LevelGround_cw_slow_01_04',
    'LevelGround_cw_slow_01_05', 'Ramp_1_L_01_01', 'Ramp_1_L_01_02', 'Ramp_1_L_01_03', 'Ramp_1_L_01_04',
    'Ramp_1_L_01_05', 'Ramp_1_R_01_01', 'Ramp_1_R_01_02', 'Ramp_1_R_01_03', 'Ramp_1_R_01_04', 'Ramp_1_R_01_05',
    'Ramp_2_L_01_01', 'Ramp_2_L_01_02', 'Ramp_2_L_01_03', 'Ramp_2_L_01_04', 'Ramp_2_L_01_05', 'Ramp_2_R_01_01',
    'Ramp_2_R_01_02', 'Ramp_2_R_01_03', 'Ramp_2_R_01_04', 'Ramp_2_R_01_05', 'Ramp_3_L_01_01', 'Ramp_3_L_01_02',
    'Ramp_3_L_01_03', 'Ramp_3_L_01_04', 'Ramp_3_L_01_05', 'Ramp_3_R_01_01', 'Ramp_3_R_01_02', 'Ramp_3_R_01_03',
    'Ramp_3_R_01_04', 'Ramp_3_R_01_05', 'Ramp_4_L_01_01', 'Ramp_4_L_01_02', 'Ramp_4_L_01_03', 'Ramp_4_L_01_04',
    'Ramp_4_L_01_05', 'Ramp_4_R_01_01', 'Ramp_4_R_01_02', 'Ramp_4_R_01_03', 'Ramp_4_R_01_04', 'Ramp_4_R_01_05',
    'Ramp_5_L_01_01', 'Ramp_5_L_01_02', 'Ramp_5_L_01_03', 'Ramp_5_L_01_04', 'Ramp_5_L_01_05', 'Ramp_5_R_01_01',
    'Ramp_5_R_01_02', 'Ramp_5_R_01_03', 'Ramp_5_R_01_04', 'Ramp_5_R_01_05', 'Ramp_6_L_01_01', 'Ramp_6_L_01_02',
    'Ramp_6_L_01_03', 'Ramp_6_L_01_04', 'Ramp_6_L_01_05', 'Ramp_6_R_01_01', 'Ramp_6_R_01_02', 'Ramp_6_R_01_03',
    'Ramp_6_R_01_04', 'Ramp_6_R_01_05', 'Stair_1_L_01_01', 'Stair_1_L_01_02', 'Stair_1_L_01_03', 'Stair_1_L_01_04',
    'Stair_1_L_01_05', 'Stair_1_R_01_01', 'Stair_1_R_01_02', 'Stair_1_R_01_03', 'Stair_1_R_01_04', 'Stair_1_R_01_05',
    'Stair_2_L_01_01', 'Stair_2_L_01_02', 'Stair_2_L_01_03', 'Stair_2_L_01_04', 'Stair_2_L_01_05', 'Stair_2_R_01_01',
    'Stair_2_R_01_02', 'Stair_2_R_01_03', 'Stair_2_R_01_04', 'Stair_2_R_01_05', 'Stair_3_L_01_01', 'Stair_3_L_01_02',
    'Stair_3_L_01_03', 'Stair_3_L_01_04', 'Stair_3_L_01_05', 'Stair_3_R_01_01', 'Stair_3_R_01_02', 'Stair_3_R_01_03',
    'Stair_3_R_01_04', 'Stair_3_R_01_05', 'Stair_4_L_01_01', 'Stair_4_L_01_02', 'Stair_4_L_01_03', 'Stair_4_L_01_04',
    'Stair_4_L_01_05', 'Stair_4_R_01_01', 'Stair_4_R_01_02', 'Stair_4_R_01_03', 'Stair_4_R_01_04', 'Stair_4_R_01_05',
    'Treadmill_01_01', 'Treadmill_02_01', 'Treadmill_03_01', 'Treadmill_04_01', 'Treadmill_05_01']

SUB_LIST_ALL = [
    'AB06', 'AB07', 'AB08', 'AB09', 'AB10', 'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16', 'AB17',
    'AB18', 'AB19', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28', 'AB30'
]

DICT_SUBJECT_ID = {subject: i for i, subject in enumerate(SUB_LIST_ALL)}
DICT_TRIAL_ID = {trial: i for i, trial in enumerate(TRIAL_LIST)}

