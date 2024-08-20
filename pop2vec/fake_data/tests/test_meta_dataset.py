import json
import tempfile
from pop2vec.fake_data.meta_dataset import MetaDataSet

json_content = {
        "path": "/some/path/with//a/file.txt",
        "shape": [10, 2],
        "columns_with_dtypes": {
            "col_a": "int64",
            "col_b": "object" },
        "total_nobs": 150,
        "nobs_sumstat": 10,
        "has_pii_columns": ["col_c", "col_d"]
        }

def test_from_json():
    with tempfile.NamedTemporaryFile(mode="w") as temp_file:
        json.dump(json_content, temp_file)
        temp_file.flush()

        data_set = MetaDataSet.from_json(temp_file.name)

    for key, read_value in vars(data_set).items():
        if key == "path":
            assert read_value == "/some/path/with/a/file.txt"
        else:
            assert read_value == json_content[key]

