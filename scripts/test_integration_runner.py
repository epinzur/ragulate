import pytest
import sys

def main():
    sys.exit(pytest.main(["tests/integration_tests"]))
