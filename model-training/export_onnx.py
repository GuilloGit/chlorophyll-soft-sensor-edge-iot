"""
===============================================================================
Module Name:       export_onnx.py
Project:           Chlorophyll-a Soft-Sensor Edge-IoT System
Tier / Subsystem:  Machine Learning Workstation Tier (Model Serialization)

Description:       Decomposes the fitted Scikit-Learn production pipeline
                   (PowerTransformer + RandomForestRegressor) into an optimized
                   ONNX graph (Opset 15) and extracts the target power transform
                   parameter (lambda) for edge-side inverse unscaling.
                   
                   To prevent memory bloat and satisfy Protobuf 2GB graph limits,
                   the 100-tree Random Forest is segmented into 10 discrete chunks
                   of 10 trees each with scaled weights (w * 0.1), unified via an
                   ONNX 'Sum' node.

Data Interfaces:
  - Upstream:      edge-system/app/model.joblib (fitted Scikit-Learn pipeline)
  - Downstream:    edge-system/app-v2/model_v2.onnx (chunked ONNX ensemble)
                   edge-system/app-v2/target_transform.json (target transform parameter specification)
  - Storage / IPC: Local filesystem read/write.

References:        Mozo et al. (2022); ONNX Opset 15 Specification.
===============================================================================
"""

import os
import json
import joblib
import numpy as np
import onnx
from onnx import helper, TensorProto
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODEL_IN_PATH = os.path.join(PROJECT_ROOT, "edge-system", "app", "model.joblib")
TRANSFORM_OUT_PATH = os.path.join(PROJECT_ROOT, "edge-system", "app-v2", "target_transform.json")
UNSCALER_OUT_PATH = TRANSFORM_OUT_PATH  # Backward compatibility alias
LAMBDA_OUT_PATH = TRANSFORM_OUT_PATH    # Backward compatibility alias
MODEL_OUT_PATH = os.path.join(PROJECT_ROOT, "edge-system", "app-v2", "model_v2.onnx")


def export_chunked_onnx(model_in: str = MODEL_IN_PATH,
                        transform_out: str = None,
                        model_out: str = MODEL_OUT_PATH,
                        chunk_size: int = 10,
                        **kwargs) -> None:
    """Decomposes a fitted Scikit-Learn pipeline and exports a chunked ONNX model.

    Args:
        model_in: Path to the input serialized joblib pipeline.
        transform_out: Target path for the extracted Yeo-Johnson target transform parameters.
        model_out: Target path for the output ONNX model binary.
        chunk_size: Number of trees per TreeEnsembleRegressor chunk node.
    """
    target_transform_path = transform_out or kwargs.get("unscaler_out") or kwargs.get("lambda_out") or TRANSFORM_OUT_PATH
    print(f"[INFO] [ONNXExport] Loading trained pipeline from: {model_in}")
    edge_pipeline = joblib.load(model_in)

    # 1. Extract target unscaling parameters (lambda, mean, scale)
    y_transformer = edge_pipeline.named_steps['rf_model_with_y_scaler'].transformer_
    y_lambda = float(y_transformer.lambdas_[0])
    y_mean = float(y_transformer._scaler.mean_[0])
    y_scale = float(y_transformer._scaler.scale_[0])

    transform_params = {
        "y_lambda": y_lambda,
        "y_mean": y_mean,
        "y_scale": y_scale
    }

    os.makedirs(os.path.dirname(target_transform_path), exist_ok=True)
    with open(target_transform_path, "w") as f:
        json.dump(transform_params, f, indent=4)
    print(f"[INFO] [ONNXExport] Saved target transform parameters (lambda={y_lambda:.5f}, mean={y_mean:.5f}, scale={y_scale:.5f}) to: {target_transform_path}")

    # 2. Extract feature scaler and underlying random forest regressor
    x_scaler = edge_pipeline.named_steps['x_scaler']
    rf = edge_pipeline.named_steps['rf_model_with_y_scaler'].regressor_

    total_trees = rf.n_estimators
    n_chunks = total_trees // chunk_size
    print(f"[INFO] [ONNXExport] Splitting {total_trees} trees into {n_chunks} chunks of {chunk_size} trees...")

    # 3. Convert feature scaler to base ONNX graph
    scaler_pipeline = Pipeline([('x_scaler', x_scaler)])
    scaler_onnx = convert_sklearn(
        scaler_pipeline,
        initial_types=[('float_input', FloatTensorType([None, 4]))],
        target_opset=15
    )

    graph_nodes = list(scaler_onnx.graph.node)
    initializers = list(scaler_onnx.graph.initializer)
    scaled_output_name = scaler_onnx.graph.output[0].name

    # 4. Convert each chunk of trees with scaled target weights
    sub_outputs = []
    scale = float(chunk_size) / float(total_trees)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size

        rf_chunk = RandomForestRegressor(n_estimators=chunk_size)
        rf_chunk.estimators_ = rf.estimators_[start:end]
        rf_chunk.n_outputs_ = rf.n_outputs_
        rf_chunk.n_features_in_ = rf.n_features_in_

        chunk_onnx = convert_sklearn(
            rf_chunk,
            initial_types=[('chunk_input', FloatTensorType([None, 4]))],
            target_opset=15
        )

        tree_node = [n for n in chunk_onnx.graph.node if 'Tree' in n.op_type][0]
        chunk_node_name = f"TreeEnsemble_chunk_{chunk_idx}"
        chunk_out_name = f"tree_out_{chunk_idx}"

        new_attrs = []
        for attr in tree_node.attribute:
            if attr.name == "target_weights":
                scaled_weights = [w * scale for w in attr.floats]
                new_attrs.append(helper.make_attribute("target_weights", scaled_weights))
            else:
                new_attrs.append(attr)

        tree_node_new = helper.make_node(
            "TreeEnsembleRegressor",
            inputs=[scaled_output_name],
            outputs=[chunk_out_name],
            name=chunk_node_name,
            domain="ai.onnx.ml",
            **{a.name: helper.get_attribute_value(a) for a in new_attrs}
        )
        graph_nodes.append(tree_node_new)
        sub_outputs.append(chunk_out_name)
        print(f"[INFO] [ONNXExport]   -> Chunk {chunk_idx + 1}/{n_chunks} converted.")

    # 5. Aggregate sub-outputs via an ONNX Sum node
    sum_node = helper.make_node(
        "Sum",
        inputs=sub_outputs,
        outputs=["predicted_transformed"],
        name="Sum_All_Chunks"
    )
    graph_nodes.append(sum_node)

    # 6. Assemble and validate final ONNX graph
    final_graph = helper.make_graph(
        nodes=graph_nodes,
        name="Chlorophyll_RandomForest_V2_Ensemble",
        inputs=scaler_onnx.graph.input,
        outputs=[helper.make_tensor_value_info("predicted_transformed", TensorProto.FLOAT, [None, 1])],
        initializer=initializers
    )

    final_model = helper.make_model(
        final_graph,
        producer_name="chlorophyll-soft-sensor",
        opset_imports=scaler_onnx.opset_import
    )

    onnx.checker.check_model(final_model)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    with open(model_out, "wb") as f:
        f.write(final_model.SerializeToString())

    size_mb = os.path.getsize(model_out) / (1024 * 1024)
    print(f"[INFO] [ONNXExport] Export completed: {model_out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    export_chunked_onnx()
