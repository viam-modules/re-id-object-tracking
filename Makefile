.PHONY: setup clean pyinstaller clean-pyinstaller onnxruntime-gpu-wheel pytorch-wheel torchvision-wheel

MODULE_DIR=$(shell pwd)
BUILD=$(MODULE_DIR)/build

VENV_DIR=$(BUILD)/.venv
PYTHON=$(VENV_DIR)/bin/python

PYTORCH_WHEEL=torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
PYTORCH_WHEEL_URL=https://storage.googleapis.com/packages.viam.com/tools/$(PYTORCH_WHEEL)

TORCHVISION_REPO=https://github.com/pytorch/vision 
TORCHVISION_WHEEL=torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
TORCHVISION_VERSION=0.20.0

ONNXRUNTIME_WHEEL=onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
ONNXRUNTIME_WHEEL_URL=https://storage.googleapis.com/packages.viam.com/tools/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl

PYINSTALLER_WORKPATH=$(BUILD)/pyinstaller_build
PYINSTALLER_DISTPATH=$(BUILD)/pyinstaller_dist

JP6_REQUIREMENTS=requirements_jp6.txt
REQUIREMENTS=requirements.txt

# Detect JetPack 6.x (R36)
IS_JP6 := $(shell if [ -f /etc/nv_tegra_release ] && [ "$$(cat /etc/nv_tegra_release | head -1 | cut -f 2 -d ' ' | cut -f 1 -d ',')" = "R36" ]; then echo "1"; else echo "0"; fi)
	
$(VENV_DIR):
	@echo "Building python venv"
	@if [ "$(IS_JP6)" = "1" ]; then \
		sudo apt install python3.10-venv; \
		sudo apt install python3-pip; \
	fi
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

setup: $(VENV_DIR)
	@if [ "$(IS_JP6)" = "1" ]; then \
		echo "Detected JetPack 6.x - Installing requirements for JP6"; \
		$(MAKE) torchvision-wheel onnxruntime-gpu-wheel; \
		$(PYTHON) -m pip install -r $(JP6_REQUIREMENTS); \
		$(PYTHON) -m pip install 'numpy<2' $(BUILD)/$(PYTORCH_WHEEL); \
		$(PYTHON) -m pip install $(BUILD)/$(TORCHVISION_WHEEL); \
		$(PYTHON) -m pip install $(BUILD)/$(ONNXRUNTIME_WHEEL); \
	else \
		echo "Installing requirements"; \
		source $(VENV_DIR)/bin/activate && pip install -r $(REQUIREMENTS); \
	fi

pyinstaller: $(PYINSTALLER_DISTPATH)/main

$(PYINSTALLER_DISTPATH)/main: setup
	$(PYTHON) -m PyInstaller --workpath "$(PYINSTALLER_WORKPATH)" --distpath "$(PYINSTALLER_DISTPATH)" main.spec

module.tar.gz: $(PYINSTALLER_DISTPATH)/main
	cp $(PYINSTALLER_DISTPATH)/main ./
	@if [ "$(IS_JP6)" = "1" ]; then \
		cp $(MODULE_DIR)/bin/first_run.sh ./; \
		tar -czvf module.tar.gz main meta.json first_run.sh; \
		rm -f first_run.sh; \
	else \
		tar -czvf module.tar.gz main meta.json; \
	fi
	rm -f main

clean-pyinstaller:
	rm -rf $(PYINSTALLER_DISTPATH) $(PYINSTALLER_WORKPATH)
	
clean:
	rm -rf $(BUILD) cuda-keyring_1.1-1_all.deb

