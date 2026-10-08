"""No secret may be hard-coded or committed (spec §38, rule 13)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.core.config import Settings

REPO = Path(__file__).resolve().parents[2]
SECRET_PATTERNS = re.compile(
    r"AIz[a][0-9A-Za-z_\-]{35}"  # Google / Gemini API key
    r"|sk-an[t]-[0-9A-Za-z_\-]{20,}"  # Anthropic API key
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)


def test_settings_never_print_secret_values():
    s = Settings(_env_file=None, gemini_api_key="fake-gemini-key-123", jwt_secret="jwt-secret-value",
                 database_url="postgresql://u:db-password@h:5432/d")
    text = repr(s) + str(s.model_dump())
    for value in ("fake-gemini-key-123", "jwt-secret-value", "db-password"):
        assert value not in text


def test_production_refuses_to_start_without_jwt_secret():
    with pytest.raises(ValueError):
        Settings(_env_file=None, environment="production")


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_no_secret_patterns_in_tracked_or_staged_files():
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                           cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    hits = []
    for rel in files:
        path = REPO / rel
        if path.suffix in {".csv", ".xlsx", ".png", ".pdf"} or not path.is_file():
            continue
        if SECRET_PATTERNS.search(path.read_text(errors="ignore")):
            hits.append(rel)
    assert hits == []


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_env_file_is_gitignored_but_template_is_not():
    def ignored(name):
        return subprocess.run(["git", "check-ignore", "-q", name], cwd=REPO).returncode == 0

    assert ignored(".env") and not ignored(".env.example")
