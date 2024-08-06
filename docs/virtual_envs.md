
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

On regular and OSSC snellius, the batch scripts should activate the virtual environment with 


```bash
source requirements/load_venv.sh
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