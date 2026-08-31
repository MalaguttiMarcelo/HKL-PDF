from pathlib import Path

from pdf_fitting.io_handler import read_input_file


def test_example_input_can_be_read():
    input_path = Path("examples/fe_conventional.inp")

    if not input_path.exists():
        return

    config = read_input_file(
        str(input_path)
    )

    assert "structure_file" in config
    assert "gr_data_file" in config
    assert "initial" in config
    assert "refinable" in config