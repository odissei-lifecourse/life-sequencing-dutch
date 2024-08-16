# pop2vec

Embedding population registry data. This is the main code for the project [Modeling life outcomes](https://research-software-directory.org/projects/modeling-life-outcomes). It consists of the following modules:
- `llm`: compute token and person embeddings with a sequence modeling approach. This code was forked from [life2vec](https://github.com/SocialComplexityLab/)
- `graph`: compute node embeddings from population network data.
- `evaluation`: code for evaluating the embeddings on downstream prediction tasks.
- `fake_data`: create fake data for code development

Main code for the project [Modeling life outcomes](https://research-software-directory.org/projects/modeling-life-outcomes).

### Installation

Your setup depends on which machine you work. See `docs/virutal_envs.md` for instructions.

## For developers

### Tests

#### Running tests

Requires the virtual environment installed.

```bash
source .venv/bin/activate
python -m pytest -v
```

#### Writing new tests

Each module has its own directory for tests. Import any code from repository relative to the project root.
It is recommended to use [pytest](https://docs.pytest.org/en/stable/) for writing unit tests. An example can be found in `pop2vec/fake_data/tests/test_synthetic_utils.py`.

### Linting and code quality checks with `ruff` and `pre-commit`

`ruff` is a linter and code formatter. One way it can be used is directly from the command line, for instance with `ruff check`.

For two reasons, however, it's recommended to use `ruff` via `pre-commit`. First, we have a lot of files that do not conform to code style conventions, and `ruff check` will give an overwhelming number of errors.
Second, in the future, we will set up a github action to automatically run `ruff` on newly committed files.

Running `ruff` via `pre-commit` works as follows. First, you need to install the pre-commit hook with

```bash
source .venv/bin/activate
pre-commit install
```

Now, pre-commit will run everytime before you commit something via `git commit -m "some message"`. `pre-commit` runs, on all the files staged for a commit, as specified in `.pre-commit-config.yaml` . `pre-commit` will
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
