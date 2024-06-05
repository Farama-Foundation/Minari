import pathlib

import pytest
from mktestdocs import check_md_file


def test_readme():
    fpath = pathlib.Path("README.md")
    check_md_file(fpath=fpath, lang="bash")
    check_md_file(fpath=fpath)


@pytest.mark.parametrize("fpath", [*pathlib.Path("docs").glob("**/*.md")], ids=str)
def test_markdown(fpath):
    check_md_file(fpath=fpath, lang="bash")
    check_md_file(fpath=fpath)
