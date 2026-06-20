#!/usr/bin/env python3

# Description
###############################################################################
'''
Regression tests for OCDocker.Initialise config parsing behavior.
'''

# Imports
###############################################################################
import pytest

import OCDocker.Initialise as ocinit

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##


def test_parse_config_preserves_hash_inside_value(tmp_path):
    cfg = tmp_path / "OCDocker.cfg"
    cfg.write_text(
        "USER = abc#123\n"
        "vina_exhaustiveness = 5\n"
        "plants_cluster_structures = 3\n"
    )

    parsed = ocinit._parse_config_file(str(cfg))

    assert parsed["USER"] == "abc#123"


def test_parse_config_supports_inline_comments_without_corrupting_hash_values(tmp_path):
    cfg = tmp_path / "OCDocker.cfg"
    cfg.write_text(
        "USER = alice # inline comment\n"
        "vina_scoring_functions = vina#,vinardo\n"
        "vina_exhaustiveness = 5\n"
        "plants_cluster_structures = 3\n"
    )

    parsed = ocinit._parse_config_file(str(cfg))

    assert parsed["USER"] == "alice"
    assert parsed["vina_scoring_functions"] == ["vina#", "vinardo"]


def test_parse_config_db_backend_defaults_to_postgresql(tmp_path):
    cfg = tmp_path / "OCDocker.cfg"
    cfg.write_text(
        "HOST = localhost\n"
        "USER = user\n"
        "DATABASE = ocdocker\n"
        "OPTIMIZEDB = optimization\n"
    )

    parsed = ocinit._parse_config_file(str(cfg))

    assert parsed["DB_BACKEND"] == "postgresql"
    assert parsed["PORT"] is None


def test_parse_yml_config_supports_lists_and_typed_values(tmp_path):
    cfg = tmp_path / "OCDocker.yml"
    cfg.write_text(
        "DB_BACKEND: sqlite\n"
        "PORT: 3307\n"
        "vina_exhaustiveness: 9\n"
        "vina_scoring_functions:\n"
        "  - vina\n"
        "  - vinardo\n"
        "reference_column_order:\n"
        "  - name\n"
        "  - receptor\n",
        encoding="utf-8",
    )

    parsed = ocinit._parse_config_file(str(cfg))

    assert parsed["DB_BACKEND"] == "sqlite"
    assert parsed["PORT"] == 3307
    assert parsed["vina_exhaustiveness"] == 9
    assert parsed["vina_scoring_functions"] == ["vina", "vinardo"]
    assert parsed["reference_column_order"] == ["name", "receptor"]


def test_parse_yml_config_rejects_non_mapping_top_level(tmp_path):
    cfg = tmp_path / "broken.yml"
    cfg.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        ocinit._parse_config_file(str(cfg))

    assert exc.value.code == 2


def test_parse_yml_config_coerces_bool_and_scalar_values(tmp_path):
    cfg = tmp_path / "coerce.yml"
    cfg.write_text(
        "smina_minimize: true\n"
        "smina_custom_scoring: false\n"
        "vina_exhaustiveness: true\n"
        "oddt_scoring_functions: 123\n",
        encoding="utf-8",
    )

    parsed = ocinit._parse_config_file(str(cfg))

    assert parsed["smina_minimize"] == "yes"
    assert parsed["smina_custom_scoring"] == "no"
    # bool is invalid for int conversion and should fall back to default
    assert parsed["vina_exhaustiveness"] == 5
    assert parsed["oddt_scoring_functions"] == ["123"]


def test_resolve_config_file_path_uses_explicit_existing_path(tmp_path):
    cfg = tmp_path / "custom.yml"
    cfg.write_text("DB_BACKEND: sqlite\n", encoding="utf-8")

    resolved = ocinit._resolve_config_file_path(str(cfg), include_package_locations=False)

    assert resolved == str(cfg.resolve())


def test_resolve_config_file_path_falls_back_to_local_yml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "OCDocker.yml"
    cfg.write_text("DB_BACKEND: sqlite\n", encoding="utf-8")

    resolved = ocinit._resolve_config_file_path("missing.cfg", include_package_locations=False)

    assert resolved == str(cfg.resolve())


def test_resolve_config_file_path_reports_missing_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError) as exc:
        ocinit._resolve_config_file_path("missing.cfg", include_package_locations=False)

    message = str(exc.value)
    assert "Requested: missing.cfg" in message
    assert "Searched:" in message


def test_normalize_db_backend_aliases():
    assert ocinit._normalize_db_backend("postgres") == "postgresql"
    assert ocinit._normalize_db_backend("postgresql") == "postgresql"
    assert ocinit._normalize_db_backend("mysql") == "mysql"
    assert ocinit._normalize_db_backend("mariadb") == "mysql"
    assert ocinit._normalize_db_backend("sqlite") == "sqlite"
    assert ocinit._normalize_db_backend("nope") is None
