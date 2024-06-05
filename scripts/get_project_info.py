import toml


def get_package_info():
    with open("pyproject.toml", "r") as f:
        pyproject = toml.load(f)
        package_name = pyproject["tool"]["poetry"]["name"]
        package_version = pyproject["tool"]["poetry"]["version"]
        return package_name, package_version


if __name__ == "__main__":
    name, version = get_package_info()
    print(f"{name} {version}")
