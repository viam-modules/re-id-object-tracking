.PHONY: pyinstaller clean-pyinstaller

# Default values for workpath and distpath
PYINSTALLER_WORKPATH = ./pyinstaller_build
PYINSTALLER_DISTPATH = ./pyinstaller_dist

# The 'pyinstaller' target runs the build/build_pyinstaller.sh script
pyinstaller:
	WORKPATH=$(PYINSTALLER_WORKPATH) DISTPATH=$(PYINSTALLER_DISTPATH) ./build/build_pyinstaller.sh

# The 'clean-pyinstaller' target removes the workpath and distpath directories
clean-pyinstaller:
	@echo "Cleaning PyInstaller workpath: $(PYINSTALLER_WORKPATH)"
	@echo "Cleaning PyInstaller distpath: $(PYINSTALLER_DISTPATH)"
	rm -rf $(PYINSTALLER_WORKPATH) $(PYINSTALLER_DISTPATH)

build-torchvision-wheel:
	./build/build_torchvision_wheel.sh

setup:
	sudo apt-get install cmake python3-pip libopenblas-base libopenmpi-dev libomp-dev libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
# 	This install cuDnn	
	bin/first_run.sh
# 	This also pip install torch and torchvision
	$(MAKE) build-torchvision-wheel
	./install_requirements.sh

