# -*- coding: utf-8 -*-
"""
Central session configuration for all Bpod analysis scripts.

    from session_config import mice, sorted_mice, mouse_sessions
"""

# Training- and Expert-task folders per mouse
_BASE = 'F:/bpod/'

mice = {
    'SNA-145894_(1)': {
        'base_path': _BASE,
        'Training': ['W2T_left'],
        'Expert':   ['W2T_Opto_left', 'W2T_Opto_left_blocked'],
    },
    'MLA-026805_(2)': {
        'base_path': _BASE,
        'Training': ['W2T_right'],
        'Expert':   ['W2T_Opto_right', 'W2T_Opto_right_blocked'],
    },
    'MLA-026806_(3)': {
        'base_path': _BASE,
        'Training': ['W2T_right'],
        'Expert':   ['W2T_Opto_right', 'W2T_Opto_right_blocked'],
    },
    'MLA-026807_(4)': {
        'base_path': _BASE,
        'Training': ['W2T_left'],
        'Expert':   ['W2T_Opto_left', 'W2T_Opto_left_blocked'],
    },
}

sorted_mice = ['SNA-145894_(1)', 'MLA-026805_(2)', 'MLA-026806_(3)', 'MLA-026807_(4)']

# Explicit session file paths for opto analysis (opto_hitrate / opto_latency)
mouse_sessions = {
    'MLA-026805': {
        'S1': {
            'W2T': ['F:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right/Session Data/MLA-026805_(2)_W2T_Opto_right_20250916_132254.mat'],
            'opto_0.5s': ['F:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right/Session Data/MLA-026805_(2)_W2T_Opto_right_20250916_132254.mat'],
        },
        'M1': {
            'W2T': [
                'F:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right/Session Data/MLA-026805_(2)_W2T_Opto_right_20250916_125033.mat',
                'F:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right/Session Data/MLA-026805_(2)_W2T_Opto_right_20250916_125633.mat',
                'F:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026805_(2)_W2T_Opto_right_blocked_20250917_142117.mat',
                'F:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right/Session Data/MLA-026805_(2)_W2T_Opto_right_20250917_144321.mat',
            ],
            'opto_0.5s': [
                'F:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right/Session Data/MLA-026805_(2)_W2T_Opto_right_20250916_125033.mat',
                'F:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right/Session Data/MLA-026805_(2)_W2T_Opto_right_20250916_125633.mat',
                'F:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026805_(2)_W2T_Opto_right_blocked_20250917_142117.mat',
                'F:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right/Session Data/MLA-026805_(2)_W2T_Opto_right_20250917_144321.mat',
            ],
        },
    },
    'MLA-026806': {
        'S1': {
            'W2T': [
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250917_131243.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250917_124955.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250923_141950.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250923_140042.mat',
            ],
            'opto_0.5s': [
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250917_131243.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250917_124955.mat',
            ],
            'opto_2s': [
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250922_153745.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250922_160545.mat',
            ],
        },
        'M1': {
            'W2T': [
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250916_145114.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250916_152034.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250922_153745.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250922_160545.mat',
            ],
            'opto_0.5s': [
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250916_145114.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250916_152034.mat',
            ],
            'opto_2s': [
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250922_153745.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250922_160545.mat',
            ],
        },
        'M2': {
            'W2T': [
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250918_134923.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250918_132651.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250924_135305.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250924_132852.mat',
            ],
            'opto_0.5s': [
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250918_134923.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250918_132651.mat',
            ],
            'opto_2s': [
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250924_135305.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250924_132852.mat',
            ],
        },
        'mPFC': {
            'W2T': [
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250919_134643.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250919_132132.mat',
            ],
            'opto_0.5s': [
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250919_134643.mat',
                'F:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250919_132132.mat',
            ],
        },
    },
    'MLA-026807': {
        'S1': {
            'W2T': [
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250918_163451.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250918_161128.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250925_164056.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250925_164329.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250923_152241.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250923_150023.mat',
            ],
            'opto_0.5s': [
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250918_163451.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250918_161128.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250925_164056.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250925_164329.mat',
            ],
            'opto_2s': [
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250923_152241.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250923_150023.mat',
            ],
        },
        'M1': {
            'W2T': [
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250916_172933.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250916_170507.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250922_191941.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250922_185914.mat',
            ],
            'opto_0.5s': [
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250916_172933.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250916_170507.mat',
            ],
            'opto_2s': [
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250922_191941.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250922_185914.mat',
            ],
        },
        'M2': {
            'W2T': [
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250917_161142.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250917_155023.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250924_161436.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250924_155127.mat',
            ],
            'opto_0.5s': [
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250917_161142.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250917_155023.mat',
            ],
            'opto_2s': [
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250924_161436.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250924_155127.mat',
            ],
        },
        'mPFC': {
            'W2T': [
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250919_122818.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250919_120444.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250925_171037.mat',
            ],
            'opto_0.5s': [
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250919_122818.mat',
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250919_120444.mat',
            ],
            'opto_2s': [
                'F:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250925_171037.mat',
            ],
        },
    },
    'SNA-145894': {
        'S1': {
            'W2T': [
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250917_180415.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250917_174027.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250923_164205.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250923_160816.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250923_161449.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250923_162024.mat',
            ],
            'opto_0.5s': [
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250917_180415.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250917_174027.mat',
            ],
            'opto_2s': [
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250923_164205.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250923_160816.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250923_161449.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250923_162024.mat',
            ],
        },
        'M1': {
            'W2T': [
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250916_191935.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250916_185353.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250922_172307.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250922_165834.mat',
            ],
            'opto_0.5s': [
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250916_191935.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250916_185353.mat',
            ],
            'opto_2s': [
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250922_172307.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250922_165834.mat',
            ],
        },
        'M2': {
            'W2T': [
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250918_175954.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250918_173555.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250924_171953.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250924_165518.mat',
            ],
            'opto_0.5s': [
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250918_175954.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250918_173555.mat',
            ],
            'opto_2s': [
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250924_171953.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250924_165518.mat',
            ],
        },
        'mPFC': {
            'W2T': [
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250919_151616.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250919_145147.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250925_182049.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250925_185943.mat',
            ],
            'opto_0.5s': [
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250919_151616.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250919_145147.mat',
            ],
            'opto_2s': [
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250925_182049.mat',
                'F:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250925_185943.mat',
            ],
        },
    },
}
