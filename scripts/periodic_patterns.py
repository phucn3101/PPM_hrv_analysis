import pandas as pd

def get_periodic_patterns(df):
    periodic_patterns = []

    for _, row in df.iterrows():
        # Using SD1 and SD2 to form a pattern
        uuid = row['uuid']
        sd1 = row['SD1']
        sd2 = row['SD2']

        # Simple periodic pattern detection (example approach)
        patterns = {}
        pattern = (sd1, sd2)
        if pattern in patterns:
            patterns[pattern] += 1
        else:
            patterns[pattern] = 1

        # Convert patterns to a list of dicts for easier rendering
        pattern_list = [{'Pattern': str(p), 'Frequency': f} for p, f in patterns.items()]

        periodic_patterns.append({
            'UUID': uuid,
            'Patterns': pattern_list
        })
    
    return periodic_patterns
