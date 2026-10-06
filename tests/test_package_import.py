def test_package_import_and_version() -> None:
    import nervenuc

    assert nervenuc.__version__ == "0.1.0"
