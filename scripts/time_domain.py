import pandas as pd

def get_time_domain_features(df):
    time_cols = ['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD', 'pNN25', 'pNN50', 'KURT', 'SKEW',
                 'MEAN_REL_RR', 'MEDIAN_REL_RR', 'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR',
                 'KURT_REL_RR', 'SKEW_REL_RR', 'uuid']
    time_features = df[time_cols].to_dict(orient='records')
    return time_features
