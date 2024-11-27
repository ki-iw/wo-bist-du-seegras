# ZUG-seegras

This is a template repository for a Birds on Mars project.

## Getting started with your project

This project was created from the [Birds on Mars Developer Template](https://github.com/birds-on-mars/dev-template) and comes with batteries included:
- Poetry for dependency managment
- Formatting, linting
- GitHub Actions run tests and test compatibility under different versions of Python
- Makefile commands to automatically format and lint your code


To get started, let's create a Conda environment, and install dependencies into this environment.

1. Change into the newly created project folder `ZUG-seegras`, and initialize the Git repo
    ```bash
    $ git init -b main
    ```
1. Create a new Conda environment:
    ```bash
    $ conda env create --file environment.yml
    ```
1. Activate the new environment!
    ```bash
    $ conda activate bom-ZUG-seegras
    ```
1. Install the project dependencies into the newly created Conda environment.
    ```bash
    $ make install
    ```
1. Lastly, run the project
    ```bash
    $ python -m zug_seegras
    ```


### Format on save
Formatting Python code when saving a file is *highly* recommended to make sure the files are properly formatted before any commit. To enable this, install the "Black Formatter" extension in VS Code either within the UI or by executing:
```bash
code --install-extension ms-python.black-formatter
```

In case the `code` executable cannot be found, install it via the UI by bringing up the Command Palette with `Shift+Cmd+P` and search for `Shell command: Install "code" command in PATH` and re-run the previous command. You can always run `make format` to format the entire project (altough you should then never need to do this).

To activate black in PyCharm, [follow the description here](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea).


You are now ready to start development on your project with your first commit! The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

There see all available commands for convenience, type
```bash
make
```



## Developement
Some tasks need to be done repeatedly.

### Adding dependencies
```bash
$ poetry add [package-name]
```
In case this requirement is only needed for development, add `-G dev` to mark it as needed for developement only by adding it to the "dev" group. When installing the project later some place, you can ommit them:
```bash
$ poetry install --without dev
```
You can create your own custom groups to bundle optional packages.
### Updating dependencies
Update all or specified packages within the [version constraints](https://python-poetry.org/docs/dependency-specification/) defined in `pyproject.toml`
```bash
$ poetry update [package-name(s)]
```

### Commiting to Git
Before doing so, run tests and optionally format your code (you should have "Format on save" activated by now anyway, so this should not be required):
```bash
$ make test
$ make format
```

When commiting to a branch, the pre-commit hooks will not be executed. It's still good practice to regularly check your code to make sure the code passes the pre-commit hooks.

```bash
$ make check
```

To run formatting, linting and tests run altogether run:
```bash
$ make dev
```

Then commit to git the way you always do!


## Run your code
Activate the shell with
```bash
$ poetry shell
```

### Run package
The entrypoint for your module is `zug_seegras/__main__.py` which is excecuted by running
```bash
$ python -m zug_seegras
```
### Run CLI
Additionally, a command line interface exists that can be called via
```bash
$ zug_seegras_cli
```

## Install
To install the project without the development dependencies, run
`$ poetry install --without=dev`

## Feedback and issues
Should you encounter any problems or suggestions with this template, please [open an issue here](https://github.com/birds-on-mars/dev-template/issues)