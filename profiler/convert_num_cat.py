import numpy as np

def convert_numeric_to_categorical(lst: np.ndarray, feature):
    converted = None

    if feature == 'price':
        converted = np.power(np.log1p(lst), 2) / 3
        converted = np.clip(converted, 0, 32.99)
    elif feature == 'average_rating':
        converted = np.power(lst, 2)
    elif feature == 'rating_number':
        converted = np.power(np.log1p(lst), 2) / 5
        converted = np.clip(converted, 0, 32.99)
    else:
        raise ValueError(f"Unsupported feature: {feature}")
    
    converted = np.int32(np.floor(converted))
    return converted
