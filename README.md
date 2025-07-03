# <auto-title>PYTHON PACKAGE TEMPLATE</auto-title>

Template for creating Python packages. Put your short project description here.

To configure this package and create an Azure Pipeline, see [this Notion page](https://www.notion.so/cortihome/Creating-a-new-GitHub-repository-with-CI-pipeline-9241fb356ead448b941a9d4cfa4daf73).

## Installation

### For development purposes

```bash
make install
```

## Run tests

```bash
make test
make pre-commit
```

### As a dependency

Add the following line to your `pyproject.toml` file:

```toml
python-package-template = { git = "ssh://git@github.com/corticph/python-package-template.git", tag="vX.Y.Z"}
```
