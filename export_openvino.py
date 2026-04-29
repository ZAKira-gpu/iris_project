import os
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

def build_beheaded_model(input_shape=(70, 70, 3)):
    """
    Creates a DenseNet-201 model that stops at the GlobalAveragePooling2D layer.
    """
    print(f"Building beheaded DenseNet-201 model...")
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def export_to_savedmodel(model, export_path='exported_model'):
    """
    Saves the model in TensorFlow SavedModel format.
    This is the required input for OpenVINO Model Optimizer.
    """
    print(f"Saving model to {export_path}...")
    model.save(export_path)
    print("Model saved successfully.")

def print_openvino_instructions(export_path='exported_model'):
    """
    Prints instructions on how to convert the SavedModel to OpenVINO IR format.
    """
    instructions = f"""
================================================================================
OpenVINO Export Instructions
================================================================================

1. Install OpenVINO Development Tools:
   pip install openvino-dev

2. Convert the SavedModel to OpenVINO IR format (.xml and .bin):
   mo --saved_model_dir {export_path} --input_shape [1,70,70,3] --data_type FP16 --output_dir openvino_ir

3. After conversion, you can load the model in OpenVINO:
   
   from openvino.runtime import Core
   core = Core()
   model = core.read_model("openvino_ir/saved_model.xml")
   compiled_model = core.compile_model(model, "CPU")
   
   # Inference
   results = compiled_model([input_data])[compiled_model.output(0)]

================================================================================
"""
    print(instructions)

if __name__ == "__main__":
    # Ensure we use the same architecture as the real-time script
    model = build_beheaded_model()
    
    export_path = 'densenet201_beheaded'
    export_to_savedmodel(model, export_path)
    
    print_openvino_instructions(export_path)
