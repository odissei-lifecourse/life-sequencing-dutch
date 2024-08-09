# pop2vec

Embedding population registry data. This is the main code for the project [Modeling life outcomes](https://research-software-directory.org/projects/modeling-life-outcomes). It consists of the following modules:
- `llm`: compute token and person embeddings with a sequence modeling approach. This code was forked from [life2vec](https://github.com/SocialComplexityLab/)
- `graph`: compute node embeddings from population network data.
- `evaluation`: code for evaluating the embeddings on downstream prediction tasks.
- `fake_data`: create fake data for code development

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

#### Linting and code quality checks with `pre-commit`

Before a commit is stored (with `git commit -m "some message"`), `pre-commit` runs, on all the files staged for a commit, as specified in `.pre-commit-config.yaml` . `pre-commit` will
- automatically fix certain things, such as line endings and import ordering. If this happens, it will be necessary to re-stage the files for committing.
- complain about other errors that cannot be fixed automatically. for instance
  ```bash
  src/evaluation/report_utils.py:1528:5: S101 Use of `assert` detected
  ```
  - what `S101` means can be found in the ruff docs: https://docs.astral.sh/ruff/rules/assert/

Sometimes it makes sense to turn off checks for a specific line. This can be done with

```python
assert not_recommended # noqa: S101
```
