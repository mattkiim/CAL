def get_threshold(env, constraint='velocity'):
    if constraint == 'safetygym':
        thresholds = {'Safety-CarButton1-v0': 10,
                        'Safety-CarButton2-v0': 10,
                        'Safety-PointButton1-v0': 10,
                        'Safety-PointButton2-v0': 10,
                        'Safety-PointPush1-v0':10,
                        'SafetyCarButton1-v0': 10,
                        'SafetyCarButton2-v0': 10,
                        'SafetyPointButton1-v0': 10,
                        'SafetyPointButton2-v0': 10,
                        'SafetyPointPush1-v0': 10,
                        'Quad2D': 10,
                      }
    elif constraint == 'velocity':
        thresholds = {'Ant-v3': 103.115,
                      'HalfCheetah-v3': 151.989,
                      'Hopper-v3': 82.748,
                      'Humanoid-v3': 20.140,
                      }
    return thresholds[env]