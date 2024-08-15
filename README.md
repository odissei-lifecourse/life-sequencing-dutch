# life-sequencing-dutch

Main code for the project [Modeling life outcomes](https://research-software-directory.org/projects/modeling-life-outcomes).


### For developers

#### Running tests 

Requires the virtual environment installed.

```bash
source .venv/bin/activate 
python -m pytest -v
```

#### Writing new tests 

Each module has its own directory for tests. Import any code from repository relative to the project root. 
It is recommended to use [pytest](https://docs.pytest.org/en/stable/) for writing unit tests. An example can be found in `pop2vec/fake_data/tests/test_synthetic_utils.py`. 