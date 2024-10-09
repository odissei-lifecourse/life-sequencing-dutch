import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
import warnings

def z_normalization(target_array, inverse=False, transformation_info=None):
    """Applies or inverses Z-normalization to the target array."""
    target_data = target_array.reshape(-1, 1)

    if inverse:
        scaler = transformation_info['scaler']
        return scaler.inverse_transform(target_data).flatten()
    else:
        scaler = StandardScaler()
        transformed_data = scaler.fit_transform(target_data)
        return transformed_data.flatten(), {'scaler': scaler}


def min_max_normalization(target_array, inverse=False, transformation_info=None):
    """Applies or inverses Min-Max normalization to the target array."""
    target_data = target_array.reshape(-1, 1)

    if inverse:
        scaler = transformation_info['scaler']
        return scaler.inverse_transform(target_data).flatten()
    else:
        feature_range = (0, 1)
        scaler = MinMaxScaler(feature_range=feature_range)
        transformed_data = scaler.fit_transform(target_data)
        return transformed_data.flatten(), {'scaler': scaler}


def log_transformation(target_array, inverse=False, transformation_info=None):
    """Applies or inverses log transformation to the target array."""
    target_data = target_array.reshape(-1, 1)

    if inverse:
        constant = transformation_info['constant']
        return (np.expm1(target_data) - constant).flatten()
    else:
        constant = abs(target_data.min()) + 1  # Ensure all values are positive
        transformed_data = np.log1p(target_data + constant)
        return transformed_data.flatten(), {'constant': constant}


def yeo_johnson_transformation(target_array, inverse=False, transformation_info=None):
    """Applies or inverses Yeo-Johnson transformation to the target array."""
    target_data = target_array.reshape(-1, 1)

    if inverse:
        transformer = transformation_info['transformer']
        return transformer.inverse_transform(target_data).flatten()
    else:
        transformer = PowerTransformer(method='yeo-johnson')
        transformed_data = transformer.fit_transform(target_data)
        return transformed_data.flatten(), {'transformer': transformer}


# Dictionary to map transformation names to functions
TRANSFORMATION_FUNCTIONS = {
    'z-normalization': z_normalization,
    'min-max': min_max_normalization,
    'log': log_transformation,
    'yeo-johnson': yeo_johnson_transformation
}

def target_transformation(target_array, transformation_type, inverse=False, transformation_info=None):
    """Main function to apply or inverse a transformation based on the provided type."""
    if transformation_type not in TRANSFORMATION_FUNCTIONS:
        raise ValueError(f"Unknown transformation type: {transformation_type}")

    # Call the appropriate transformation function
    func = TRANSFORMATION_FUNCTIONS[transformation_type]
    
    if inverse:
        # Inverse transformation, pass the transformation info
        return func(target_array, inverse=True, transformation_info=transformation_info)
    else:
        # Forward transformation
        return func(target_array, inverse=False)

