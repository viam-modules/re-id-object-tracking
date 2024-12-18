.PHONY: pyinstaller clean-pyinstaller

# Default values for workpath and distpath
WORKPATH ?= ./pyinstaller_build      # Default build output directory
DISTPATH ?= ./pyinstaller_dist       # Default distribution output directory

# The 'pyinstaller' target runs the build/build_pyinstaller.sh script
pyinstaller:
	WORKPATH=$(WORKPATH) DISTPATH=$(DISTPATH) ./build/build_pyinstaller.sh

# The 'clean-pyinstaller' target removes the workpath and distpath directories
clean-pyinstaller:
	@echo "Cleaning PyInstaller workpath: $(WORKPATH)"
	@echo "Cleaning PyInstaller distpath: $(DISTPATH)"
	rm -rf $(WORKPATH) $(DISTPATH)