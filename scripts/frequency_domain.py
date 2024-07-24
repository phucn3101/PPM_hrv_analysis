import pandas as pd

def get_frequency_domain_features(df): # uuid,VLF,VLF_PCT,LF,LF_PCT,LF_NU,HF,HF_PCT,HF_NU,TP,LF_HF,HF_LF
    frequency_cols = ['uuid', 'VLF', 'VLF_PCT', 'LF', 'LF_PCT', 'LF_NU', 'HF' , 'HF_PCT','HF_NU','TP','LF_HF', 'HF_LF']

    frequency_features = df[frequency_cols].to_dict(orient='records')
    return frequency_features
