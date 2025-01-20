import csv
import random
import sys
from pathlib import Path


def write_ids_to_csv(ids, output_path):
    """Writes the IDs to a CSV file with a header.

    Args:
        ids: A list of IDs to write.
        output_path: The path to the output CSV file.
    """
    with Path(output_path).open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["RINPERSOON"])
        for id_number in ids:
            writer.writerow([id_number])


def main():
    """Main function to generate IDs and write them to a CSV file."""
    len_argv = 2
    if len(sys.argv) != len_argv:
        msg = "Wrong number of arguments. Give 1 argument with the output path."
        raise RuntimeError(msg)

    output_path = sys.argv[1]
    num_ids = 27_275_012
    max_id = 999_999_999

    ids = random.sample(range(1, max_id + 1), num_ids)
    write_ids_to_csv(ids, output_path)


if __name__ == "__main__":
    main()
