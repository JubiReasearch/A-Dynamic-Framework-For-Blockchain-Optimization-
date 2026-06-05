import numpy as np
def compute_metrics(x):
    tps_range = (0.0, 1000.0)
    lat_range = (0.0, 2000.0)
    dec_range = (0.0, 1.0)
    eff_range = (0.0, 200.0)
    tps_score = np.clip((x[0] - tps_range[0]) / (tps_range[1] - tps_range[0]), 0.0, 1.0)
    lat_score = 1.0 - np.clip((x[1] - lat_range[0]) / (lat_range[1] - lat_range[0]), 0.0, 1.0)
    dec_score = np.clip((x[2] - dec_range[0]) / (dec_range[1] - dec_range[0]), 0.0, 1.0)
    eff_score = 1.0 - np.clip((x[3] - eff_range[0]) / (eff_range[1] - eff_range[0]), 0.0, 1.0)
    return np.array([tps_score, lat_score, dec_score, eff_score])
