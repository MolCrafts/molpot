# Getting start

In this section we will guide you through all the feature of `molpot` and help you explore the potential of this package.

## Installation

Since the most of feature are still under development, we recommend you to install the package with editable mode, thus you can easily debug code:

``` python
git clone https://github.com/MolCrafts/molpot.git
git checkout dev
cd molpot
pip install -e .
```
All the changes you made in the code will take effect immediately except `op` module.
Type `molpot` in your terminal to check if the installation is successful. 

### devcontainer

If you are using `vscode`, you can use the `devcontainer` to develop the code. Open the project in `vscode`, and click the `Reopen in Container` button in prompt, then `devcontainer` will automatically set up the environment for you.

We don't provide dockefile, but you can still build image with `devcontainer-cli`.

### 