def test_import_pdf_calculator():
    from pdf_fitting.models.gr_model import PDFCalculator

    assert PDFCalculator is not None


def test_import_fit_engine():
    from pdf_fitting.fit_engine import FitEngine

    assert FitEngine is not None


def test_import_io_handler():
    from pdf_fitting.io_handler import read_input_file

    assert callable(read_input_file)