
## Installing virtual environments and working with them

### On snellius


#### Installing the venv

```bash
source requirements/snel_modules_2023.sh

python -m venv .venv
source .venv/bin/activate
pip install -r requirements/snellius.txt
```


#### Running them

On regular and OSSC Snellius, the batch scripts should activate the virtual environment with


```bash
source requirements/load_venv.sh
```

On the CBS RA: Open a powershell prompt and then run:

```console
C:\mambaforge\envs\9424\Scripts\jupyter-notebook.exe --notebook-dir="H:/"
# or
C:\mambaforge\envs\9424\python.exe \path\to\some\script.py
```


### On other machines
For other cases (ie, on local laptops or on a github action), the virtual environment is used in the regular way.

#### Installing the venv


```bash
pyenv install 3.11.3 # might be necessary
pyenv local 3.11.3 # or other ways to get the right python version
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/regular.txt
```

#### Using the venv

```
source .venv/bin/activate
```


### For developers

#### For Snellius/OSSC/regular
The file `requirements/source.txt` was carefully put together so that all models can be run with the same dependencies.
The virtual environment on the OSSC is installed by SURF.


**Workflow for new requirements and venvs**
1. Put together the requirements on snellius.
    ```bash
    module load YEAR
    module load PYTHON
    pip install additional-packages
    ```
2. Export and translate
    ```bash
    pip freeze > requirements/source.txt
    python requirements/translate.py
    ```
3. Give `requirements/source.txt` to SURF so they can install a new environment on the OSSC.

Note. If you install a package with `pip install pkg_name`, it's suggested to add this package to the list `PKG_NO_PIN` in `requirements/translate.py`. This will make it easier to install the venv on other machines. The environment on snellius will still be equivalent to the environment on the OSSC because `requirements/source.txt` records the exact version you installed on Snellius and this will be used to install the virtual environment on the OSSC.

#### Sending diffs to SURF

If you only add new packages, you can send SURF the diff between the old and the new requirements file. I used this code to achieve this:
```bash
git diff <commit1> <commit2> -- <file_path> | grep -E "^(\+|-)" | grep -Ev "^(\+\+|\-\-)" | sed 's/^[+-]//' > diff_output.txt
```

For bigger changes it's better to create a new virtual environment with a complete requirements file.


#### For CBS
To create the same virtual environment, follow the steps here *on a Windows machine*: https://github.com/sodascience/cbs_python. As input requirements.txt use `requirements/regular.txt`.

Make sure to test the created requirements file. Any lines with `--find-links` will have to be manually added to the requirements file. Note that the `--find-links` created some problems when importing via CBS; their package mirror Nexus at first did not seem to find the relevant PyTorch installation. In a second try, it worked.

The current environment on the CBS RA is created with `requirements/cbs_ra.txt`.
