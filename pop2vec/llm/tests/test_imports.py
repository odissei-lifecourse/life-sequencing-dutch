
import pytest 




def test_imports_src():
    from pop2vec.llm.src import (
        callbacks,
        prepare_data,
        strategy, # delete?
        test,
        train,
        # tune ## requires ray package
        utils,
        # val # TODO: FAILED pop2vec/llm/tests/test_imports.py::test_imports - ValueError: resolver 'last_ckpt' is already registered
    )

def test_imports_new_code():
    from pop2vec.llm.src.new_code import (
        explore,
        infer_embedding,
        # check_pipeline_output, # FileNotFoundError
        # create_dummy_data, # CustomDataset not defined anymore
        create_life_seq_jsons,
        custom_vocab,
        data_io_utilities,
    )