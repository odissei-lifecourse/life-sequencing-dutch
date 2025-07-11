import glob
import pyarrow.parquet as pq


def merge_parquets(source_dir, output_path):
    files = glob.glob(f"{source_dir}/**/*sequences.parquet", recursive=True)
    if not files:
        raise FileNotFoundError("No input parquet files under source")
    first_pf = pq.ParquetFile(files[0])
    writer = pq.ParquetWriter(
        output_path,
        first_pf.schema_arrow,
        version=first_pf.metadata.format_version,
        compression=None
    )

    for path in files:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg_table = pf.read_row_group(rg_idx)
            writer.write_table(rg_table, row_group_size=rg_table.num_rows)
    writer.close()
    print(f"Merged {len(files)} parquet files into {output_path}")