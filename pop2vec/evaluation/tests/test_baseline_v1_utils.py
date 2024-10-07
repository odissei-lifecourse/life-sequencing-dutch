from pop2vec.evaluation.baseline_v1_utils import target_transformation, TRANSFORMATION_FUNCTIONS 
import pandas as pd
import numpy as np

# Main function to test the code
def test_transformation():
    # Create a DataFrame with two columns
    df = pd.DataFrame({
        'col1': np.random.normal(25000, 40000, 200),  # 200 Random values from a normal distribution
        'col2': np.array(
                  [-1000, -10, 0, 10, 1000] + np.random.randint(-1000, 1000, 195).tolist()
                )  # Random values between -1000 and 1000
    })

    # Epsilon for floating point comparison
    eps = 1e-6

    # Iterate over all transformation functions
    for transformation_type in TRANSFORMATION_FUNCTIONS.keys():
        print(f"\nTesting transformation: {transformation_type}")
        for col in ['col1', 'col2']:
          # Apply the transformation to 'col1'
          transformed_col, transformation_info = target_transformation(
            df[col].to_numpy(), transformation_type
          )
          
          # Apply the inverse transformation
          inverse_transformed_col = target_transformation(
            transformed_col, 
            transformation_type, 
            inverse=True, 
            transformation_info=transformation_info
          )
          
          # Assert that the inverse transformation is within epsilon of the original data
          assert np.allclose(
            df[col].to_numpy(), 
            inverse_transformed_col, 
            atol=eps
          ), f"Inverse transformation failed for {transformation_type}"

          print(f"Transformation {transformation_type} passed.")

if __name__ == "__main__":
    test_transformation()
