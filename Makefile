# Define variables to extract the package name and version from pyproject.toml
PACKAGE_INFO = $(shell python scripts/get_project_info.py)
PACKAGE_NAME = $(shell echo $(PACKAGE_INFO) | awk '{print $$1}')
PACKAGE_VERSION = $(shell echo $(PACKAGE_INFO) | awk '{print $$2}')
WHEEL_FILE = dist/$(PACKAGE_NAME)-$(PACKAGE_VERSION)-py3-none-any.whl

.PHONY: build uninstall install build_and_reinstall fmt

# Target to build the package using poetry
build:
	poetry build

# Target to uninstall the package using pip
uninstall:
	pip uninstall -y $(PACKAGE_NAME)

# Target to install the package using pip
install:
	pip install -q $(WHEEL_FILE)

# Target to run all steps: build, uninstall, and install
build_and_reinstall: build uninstall install

# Sort imports and format python files
fmt:
	isort --profile black .
	black .
