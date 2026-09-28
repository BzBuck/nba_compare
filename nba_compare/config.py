"""
Central place for pointing nba_compare at the basketball encyclopedia's
parquet files, so they stay a single source of truth instead of being
copied/duplicated here.

The default source is the Hugging Face dataset
https://huggingface.co/datasets/BBuckz/basketball-encyclopedia -- each file
is downloaded once and served out of the local Hugging Face cache
(~/.cache/huggingface by default) on every run after that. Nothing needs to
sit next to this project on disk, so the app runs the same on a laptop and
on a deployed host.

Two optional environment overrides:

  NBA_COMPARE_DATA_DIR    read the parquet files straight out of this local
                          folder instead of the Hub -- for working offline,
                          or testing a rebuild of the data before it's
                          pushed. Point it at the NBA Encyclopedia project's
                          data/ folder to get the old behaviour back.
  NBA_COMPARE_HF_REVISION branch, tag, or commit sha to pull instead of
                          "main" -- pin it to a sha if you ever need a run
                          to be reproducible against one version of the data.

HF_TOKEN / HF_HOME are read by huggingface_hub itself; a token is only
needed if the dataset is ever made private.
"""
from __future__ import annotations

import os
from pathlib import Path

HF_REPO_ID = "BBuckz/basketball-encyclopedia"
HF_REPO_TYPE = "dataset"
# The NBA files live under nba/ in the dataset (wnba/ is the other league).
HF_PATH_PREFIX = "nba"
DEFAULT_HF_REVISION = "main"

LOCAL_DIR_ENV = "NBA_COMPARE_DATA_DIR"
REVISION_ENV = "NBA_COMPARE_HF_REVISION"

# Logical dataset key -> filename, shared by both sources: the local
# override folder is expected to use the same names the Hub repo does.
FILENAMES = {
    "regular": "nba_gamelogs.parquet",
    "playoffs": "nba_playoffs_gamelogs.parquet",
    "team_regular": "nba_team_gamelogs.parquet",
    "team_playoffs": "nba_team_playoffs_gamelogs.parquet",
}


def local_data_dir() -> Path | None:
    """The NBA_COMPARE_DATA_DIR override, or None when it isn't set."""
    raw = os.environ.get(LOCAL_DIR_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


def hf_revision() -> str:
    return os.environ.get(REVISION_ENV, "").strip() or DEFAULT_HF_REVISION


def resolve(key: str) -> str:
    """
    Local filesystem path for one dataset, downloading it from the Hub on
    first use if it isn't cached yet. Call this lazily -- it's a network
    round trip the first time, and the point of NBADataStore's per-file
    caching is that a session only ever pays for the files it touches.
    """
    try:
        filename = FILENAMES[key]
    except KeyError:
        raise KeyError(
            f"unknown dataset {key!r}; expected one of {sorted(FILENAMES)}"
        ) from None

    override = local_data_dir()
    if override is not None:
        path = override / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{LOCAL_DIR_ENV} is set to {override}, but {filename} isn't "
                f"there. Unset {LOCAL_DIR_ENV} to pull from "
                f"https://huggingface.co/datasets/{HF_REPO_ID} instead."
            )
        return str(path)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to load the data from the Hub "
            "(pip install -r requirements.txt), or set "
            f"{LOCAL_DIR_ENV} to a folder holding the parquet files."
        ) from None

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        revision=hf_revision(),
        filename=f"{HF_PATH_PREFIX}/{filename}",
    )


def describe_source() -> str:
    """One-line human-readable description of where data is coming from."""
    override = local_data_dir()
    if override is not None:
        return f"local folder {override}"
    return f"hf://datasets/{HF_REPO_ID}@{hf_revision()}/{HF_PATH_PREFIX}"
