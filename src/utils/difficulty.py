import numpy as np

def calculate_difficulty(weather_type, param_value):
    """
    Calculates difficulty score based on the paper's definition.
    Diff_phys(x(lambda)) = lambda
    """
    # Assuming param_value is already normalized to [0, 1] relative to the max severity
    return param_value

def categorize_data(meta_data):
    """
    Partition dataset into subsets based on NuScenes categories.
    """
    # Placeholder for logic mapping nuscenes scene descriptions to categories
    # NuScenes has 'description' like "Rain, night"
    description = meta_data.get('description', '').lower()
    
    if 'rain' in description:
        return 'rain'
    elif 'fog' in description:
        return 'fog'
    elif 'snow' in description:
        return 'snow'
    elif 'night' in description:
        return 'night'
    else:
        return 'clear'
