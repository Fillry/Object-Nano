#!/bin/bash
# ==============================================================================
# ITS Edge AI Setup Script for NVIDIA Jetson Nano
# ==============================================================================

set -e

# ANSI Color Codes
GREEN='\033[0;32m'
NC='\033[0m' # No Color
YELLOW='\033[1;33m'
RED='\033[0;31m'

echo -e "${GREEN}======================================================"
echo -e "   Starting ITS Pipeline Setup on NVIDIA Jetson Nano   "
echo -e "======================================================${NC}"

# 1. Setup CUDA environment in bashrc if not present
echo -e "\n${YELLOW}[Step 1] Checking CUDA environment...${NC}"
if ! grep -q "cuda/bin" ~/.bashrc; then
    echo "Adding CUDA paths to ~/.bashrc..."
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo -e "${GREEN}CUDA paths added to ~/.bashrc. Please run 'source ~/.bashrc' after this script completes.${NC}"
else
    echo "CUDA paths already configured in ~/.bashrc."
fi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Check nvcc compiler
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Warning: nvcc compiler not found! Make sure CUDA is installed at /usr/local/cuda${NC}"
else
    nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}')
    echo -e "${GREEN}Found CUDA compiler (nvcc) version: $nvcc_version${NC}"
fi

# 2. Setup Virtual Environment
echo -e "\n${YELLOW}[Step 2] Setting up Python Virtual Environment...${NC}"
if [ ! -d "env_its" ]; then
    echo "Creating env_its virtual environment with system packages enabled..."
    python3 -m venv --system-site-packages env_its
    echo -e "${GREEN}env_its created successfully.${NC}"
else
    echo "env_its directory already exists. Skipping creation."
fi

# Activate virtual environment
source env_its/bin/activate

# 3. Inform user about PyTorch (requires manual step due to platform dependencies)
echo -e "\n${YELLOW}[Step 3] Verifying PyTorch installation...${NC}"
if python3 -c "import torch; print('PyTorch Version:', torch.__version__, 'CUDA Available:', torch.cuda.is_available())" 2>/dev/null; then
    echo -e "${GREEN}PyTorch with CUDA support is already installed!${NC}"
else
    echo -e "${RED}PyTorch with CUDA support is NOT installed!${NC}"
    echo -e "Since Jetson Nano requires official NVIDIA wheels, please run these commands manually:"
    echo -e "--------------------------------------------------------------------------------"
    echo -e "  wget https://nvidia.box.com/shared/static/fjup34sb5gq2yiyx9u2f9uxf3n31t175.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
    echo -e "  pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
    echo -e "--------------------------------------------------------------------------------"
fi

# 4. Install PyCUDA
echo -e "\n${YELLOW}[Step 4] Installing PyCUDA (for TensorRT)...${NC}"
if python3 -c "import pycuda" 2>/dev/null; then
    echo -e "${GREEN}PyCUDA is already installed.${NC}"
else
    echo "Installing PyCUDA from source... This may take a few minutes."
    pip3 install pycuda --user
fi

# 5. Install Dependencies (Avoiding OpenCV Overwrite)
echo -e "\n${YELLOW}[Step 5] Installing Python package dependencies...${NC}"
pip3 install pyyaml pandas tqdm scipy matplotlib

# Install ultralytics with --no-deps to prevent overwriting the system opencv-python
echo "Installing Ultralytics (without dependencies to protect custom OpenCV)..."
pip3 install --no-deps ultralytics

# 6. Verify OpenCV with GStreamer & CUDA
echo -e "\n${YELLOW}[Step 6] Verifying OpenCV backend support...${NC}"
python3 -c "
import cv2
print('OpenCV Version:', cv2.__version__)
print('GStreamer Support:', cv2.getBuildInformation().find('GStreamer') != -1)
print('CUDA Support:', cv2.getBuildInformation().find('CUDA') != -1)
"

# 7. Provide instructions for TensorRT export
echo -e "\n${YELLOW}[Step 7] Exporting Models to TensorRT...${NC}"
echo -e "To export YOLOv26n to ONNX, run:"
echo -e "  ${GREEN}yolo export model=yolo26n.pt format=onnx dynamic=True opset=12${NC}"
echo -e "To compile the ONNX model to TensorRT (.engine) on the Jetson Nano GPU, run:"
echo -e "  ${GREEN}/usr/src/tensorrt/bin/trtexec --onnx=yolo26n.onnx --saveEngine=yolo26n.engine --fp16${NC}"
echo -e "To compile TinyViT ONNX model to TensorRT (.engine), run:"
echo -e "  ${GREEN}/usr/src/tensorrt/bin/trtexec --onnx=models/tinyVIT_ThaiCar_ONNX/brand_classifier.onnx --saveEngine=models/tinyVIT_ThaiCar_ONNX/brand_classifier.engine --fp16${NC}"

echo -e "\n${GREEN}======================================================"
echo -e "                  Setup Completed!                    "
echo -e "======================================================${NC}"
echo -e "Please run the following commands to start using the system:"
echo -e "  1. ${YELLOW}source ~/.bashrc${NC} (If this is your first setup)"
echo -e "  2. ${YELLOW}source env_its/bin/activate${NC}"
echo -e "  3. Configure ${YELLOW}configs/model_config.yaml${NC} to use .engine files and device: 'cuda'"
echo -e "  4. Configure ${YELLOW}configs/camera_config.yaml${NC} to enable hardware decoding (use_jetson_hw_dec: true)"
echo -e "  5. Run: ${YELLOW}python3 main.py --headless${NC}"
