import joblib
import json
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load the original V1 model
print("Loading model.joblib...")
edge_pipeline = joblib.load("edge-system/app/model.joblib")

# Extract Yeo-Johnson lambda for the target variable (y)
y_transformer = edge_pipeline.named_steps['rf_model_with_y_scaler'].transformer_
y_lambda = float(y_transformer.lambdas_[0])

with open("edge-system/app-v2/y_lambda.json", "w") as f:
    json.dump(y_lambda, f)
print(f"Exported y_lambda.json: {y_lambda}")

# Extract the base feature scaler (X) and the base RF regressor
x_scaler = edge_pipeline.named_steps['x_scaler']
rf_model = edge_pipeline.named_steps['rf_model_with_y_scaler'].regressor_

from sklearn.pipeline import Pipeline
# Create a pure forward-pass pipeline (X_scaler -> RF)
forward_pipeline = Pipeline([
    ('x_scaler', x_scaler),
    ('rf_model', rf_model)
])

# Convert to ONNX with target_opset=15 to ensure compatibility with onnxruntime 1.17.1
print("Converting to ONNX...")
initial_type = [('float_input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(forward_pipeline, initial_types=initial_type, target_opset=15)

with open("edge-system/app-v2/model_v2.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("Exported model_v2.onnx successfully with target_opset=15.")
