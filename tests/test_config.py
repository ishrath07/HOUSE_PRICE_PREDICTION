from house_price import config, data


def test_paths_are_inside_project():
    assert config.RAW_DIR.parent == config.DATA_DIR
    assert config.MODELS_DIR.parent == config.PROJECT_ROOT


def test_ensure_dirs_creates_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    config.ensure_dirs()
    assert (tmp_path / "models").is_dir()


def test_split_is_reproducible_and_excludes_target():
    import pandas as pd

    df = pd.DataFrame(
        {
            config.ID_COLUMN: range(100),
            "LotArea": range(100),
            config.TARGET: [float(i) + 1000 for i in range(100)],
        }
    )
    a = data.split_data(df)
    b = data.split_data(df)
    assert a[0].index.equals(b[0].index)
    assert config.TARGET not in a[0].columns
    assert len(a[1]) == int(100 * config.TEST_SIZE)


def test_load_raw_missing_file_gives_helpful_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="Kaggle"):
        data.load_raw(tmp_path / "nope.csv")