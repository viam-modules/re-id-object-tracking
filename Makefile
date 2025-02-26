.PHONY: setup clean pyinstaller clean-pyinstaller onnxruntime-gpu-wheel pytorch-wheel torchvision-wheel

MODULE_DIR=$(shell pwd)
BUILD=$(MODULE_DIR)/build

VENV_DIR=$(BUILD)/.venv
PYTHON=$(VENV_DIR)/bin/python

PYTORCH_WHEEL=torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
PYTORCH_WHEEL_URL=https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/$(PYTORCH_WHEEL)

TORCHVISION_REPO=https://github.com/pytorch/vision 
TORCHVISION_WHEEL=torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
TORCHVISION_VERSION=0.20.0

ONNXRUNTIME_WHEEL=onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
ONNXRUNTIME_WHEEL_URL=https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/0c4/18beb3326027d/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl#sha256=0c418beb3326027d83acc283372ae42ebe9df12f71c3a8c2e9743a4e323443a4

REQUIREMENTS=requirements.txt

PYINSTALLER_WORKPATH=$(BUILD)/pyinstaller_build
PYINSTALLER_DISTPATH=$(BUILD)/pyinstaller_dist
	
$(VENV_DIR):
	@echo "Building python venv"
	sudo apt install python3.10-venv
	sudo apt install python3-pip
	python3 -m venv $(VENV_DIR)
	
$(BUILD)/$(ONNXRUNTIME_WHEEL):
	wget $(ONNXRUNTIME_WHEEL_URL) -O $(BUILD)/$(ONNXRUNTIME_WHEEL)

onnxruntime-gpu-wheel: $(BUILD)/$(ONNXRUNTIME_WHEEL)

$(BUILD)/$(PYTORCH_WHEEL):
	@echo "Making $(BUILD)/$(PYTORCH_WHEEL)"
	wget  -P $(BUILD) $(PYTORCH_WHEEL_URL)

pytorch-wheel: $(BUILD)/$(PYTORCH_WHEEL)

$(BUILD)/$(TORCHVISION_WHEEL): $(VENV_DIR) $(BUILD)/$(PYTORCH_WHEEL)
	@echo "Installing dependencies for TorchVision"
	bin/first_run.sh
	bin/install_cusparselt.sh

	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install wheel
	$(PYTHON) -m pip install 'numpy<2' $(BUILD)/$(PYTORCH_WHEEL)

	@echo "Cloning Torchvision"
	git clone --branch v${TORCHVISION_VERSION} --recursive --depth=1 $(TORCHVISION_REPO) $(BUILD)/torchvision

	@echo "Building torchvision wheel"
	cd $(BUILD)/torchvision && $(PYTHON) setup.py --verbose bdist_wheel --dist-dir ../

torchvision-wheel: $(BUILD)/$(TORCHVISION_WHEEL)

$(PYINSTALLER_DISTPATH)/main: $(BUILD)/$(TORCHVISION_WHEEL) $(BUILD)/$(ONNXRUNTIME_WHEEL)
	@echo " Building pyinstaller executable"
	$(PYTHON) -m pip install -r $(REQUIREMENTS)
	$(PYTHON) -m pip install 'numpy<2' $(BUILD)/$(PYTORCH_WHEEL)
	$(PYTHON) -m pip install $(BUILD)/$(TORCHVISION_WHEEL)
	$(PYTHON) -m pip install $(BUILD)/$(ONNXRUNTIME_WHEEL)
	$(PYTHON) -m PyInstaller --workpath "$(PYINSTALLER_WORKPATH)" --distpath "$(PYINSTALLER_DISTPATH)" main.spec



pyinstaller: $(PYINSTALLER_DISTPATH)/main

clean-pyinstaller:
	rm -rf $(PYINSTALLER_DISTPATH) $(PYINSTALLER_WORKPATH)

setup: torchvision-wheel onnxruntime-gpu-wheel

clean:
	rm -rf $(BUILD) cuda-keyring_1.1-1_all.deb

