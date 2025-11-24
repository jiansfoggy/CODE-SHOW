import torch
from ultralytics import YOLO
import coremltools as ct
import os

def convert_yolo_to_coreml():
    """
    Downloads the yolov9-c.pt model and converts it to the Core ML format (.mlpackage).
    """
    print("Loading YOLOv9-c model...")
    # Load the YOLOv9 model
    # This will download yolov9-c.pt if it's not present
    model = YOLO('yolov9-c.pt')

    # Define the export arguments
    export_args = {
        'format': 'coreml',
        'imgsz': 640,
        'half': True,
        'int8': False,
        'nms': True,
    }

    print("Exporting model to Core ML format...")
    # Export the model to CoreML format
    # The output will be yolov9-c.mlpackage
    model_path = model.export(**export_args)
    
    print(f"Model successfully converted and saved at: {model_path}")

    # It's good practice to move the converted model to a known location
    output_dir = "converted_models"
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, os.path.basename(model_path))

    if os.path.exists(final_path):
        # In case of name collision, remove old directory
        import shutil
        if os.path.isdir(final_path):
            shutil.rmtree(final_path)
        else:
            os.remove(final_path)

    os.rename(model_path, final_path)
    
    print(f"Model moved to: {final_path}")


if __name__ == '__main__':
    convert_yolo_to_coreml()
