import argparse
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("YOLOONNXExporter")

def main():
    parser = argparse.ArgumentParser(description="Export YOLO Model to ONNX format optimized for Jetson Nano")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Path to YOLO .pt model file")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version (12 is highly recommended for older TensorRT)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.error(f"Model file not found at: {args.model}")
        return

    logger.info(f"Loading Ultralytics YOLO to export {args.model} to ONNX format...")
    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
        
        logger.info(f"Exporting {args.model} to ONNX (dynamic=True, opset={args.opset})...")
        # Run export
        output_path = model.export(
            format="onnx",
            dynamic=True,
            opset=args.opset
        )
        logger.info(f"Successfully exported to: {output_path}")
        logger.info("Next Step: Copy the .onnx file to your Jetson Nano and run:")
        logger.info(f"  /usr/src/tensorrt/bin/trtexec --onnx={os.path.basename(output_path)} --saveEngine={os.path.splitext(os.path.basename(output_path))[0]}.engine --fp16")
    except Exception as e:
        logger.error(f"Failed to export YOLO model: {e}")

if __name__ == "__main__":
    main()
