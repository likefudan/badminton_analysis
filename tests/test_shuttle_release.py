"""Fine-tuned TrackNet from a GitHub Release: download once, check sha256, unpack safely."""

import hashlib
import zipfile

import pytest

from badminton import shuttle as S

NAME = "tracknetv3_tracknet_fp16_b2.mlpackage"


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def release(tmp_path, monkeypatch):
    src = tmp_path / "server"
    src.mkdir()

    def publish(members, sha=None):
        digest = make_zip(src / "m.zip", members)
        monkeypatch.setitem(S.TRACKNET_RELEASES, "ft-test", {"file": "m.zip", "sha256": sha or digest, "size": 1})
        return f"file://{src}/{{file}}"

    return publish


def test_downloads_checks_and_caches(tmp_path, release):
    url = release({f"{NAME}/Manifest.json": "{}", f"{NAME}/Data/w.bin": b"\x00" * 10})
    cache = tmp_path / "cache"
    path = S.release_model("ft-test", cache, url)
    assert path == cache / "ft-test" / NAME and (path / "Data" / "w.bin").stat().st_size == 10
    (tmp_path / "server" / "m.zip").unlink()  # second use: from the cache, no download
    assert S.release_model("ft-test", cache, url) == path


def test_wrong_checksum_leaves_nothing_behind(tmp_path, release):
    url = release({f"{NAME}/Manifest.json": "{}"}, sha="0" * 64)
    cache = tmp_path / "cache"
    with pytest.raises(SystemExit, match="校验"):
        S.release_model("ft-test", cache, url)
    assert not (cache / "ft-test").exists() and list(cache.iterdir()) == []


def test_zip_with_files_outside_the_model_is_refused(tmp_path, release):
    url = release({f"{NAME}/Manifest.json": "{}", "../evil.txt": "x"})
    with pytest.raises(SystemExit, match="意外"):
        S.release_model("ft-test", tmp_path / "cache", url)
    assert not (tmp_path / "evil.txt").exists()


def test_choice_follows_the_environment(monkeypatch):
    monkeypatch.delenv("BADMINTON_TRACKNET", raising=False)
    assert S.tracknet_choice() == S.TRACKNET_DEFAULT
    monkeypatch.setenv("BADMINTON_TRACKNET", "released")
    assert S.tracknet_choice() == "released"
    monkeypatch.setenv("BADMINTON_TRACKNET", "no-such-model")
    with pytest.raises(SystemExit):
        S.tracknet_choice()


def test_default_is_a_published_release():
    rel = S.TRACKNET_RELEASES[S.TRACKNET_DEFAULT]
    assert len(rel["sha256"]) == 64 and rel["file"].endswith(".mlpackage.zip")
