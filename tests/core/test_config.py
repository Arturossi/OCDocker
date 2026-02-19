#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCDocker.Config helpers and singleton behavior.
'''

# Imports
###############################################################################
import sys
import types
import pytest

import OCDocker.Config as occfg
import OCDocker.Error as ocerror

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

def _install_fake_initialise(monkeypatch, parse_fn, resolve_fn=None):
    fake_init = types.ModuleType("OCDocker.Initialise")
    fake_init._parse_config_file = parse_fn
    if resolve_fn is not None:
        fake_init._resolve_config_file_path = resolve_fn
    monkeypatch.setitem(sys.modules, "OCDocker.Initialise", fake_init)


## Public ##

@pytest.fixture(autouse=True)
def _reset_global_config():
    occfg.reset_config()
    yield
    occfg.reset_config()


@pytest.mark.order(109)
def test_get_exhaustiveness_handles_int_and_fallback():
    assert occfg._get_exhaustiveness({"x": 7}, "x", 5) == 7
    assert occfg._get_exhaustiveness({"x": "9"}, "x", 5) == 9
    assert occfg._get_exhaustiveness({"x": "abc"}, "x", 5) == "abc"
    assert occfg._get_exhaustiveness({"x": None}, "x", 5) == "None"


@pytest.mark.order(110)
def test_from_config_file_uses_env_path_then_fallback_local(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "OCDocker.cfg").write_text("[dummy]\n", encoding="utf-8")
    monkeypatch.setenv("OCDOCKER_CONFIG", "missing-from-env.cfg")

    seen = {}

    def _parse(path):
        seen["path"] = path
        return {"vina_exhaustiveness": "11", "oddt_models_dir": "/models"}

    _install_fake_initialise(monkeypatch, _parse)
    cfg = occfg.OCDockerConfig.from_config_file("")

    assert seen["path"] == str((tmp_path / "OCDocker.cfg").resolve())
    assert cfg.vina.exhaustiveness == 11
    assert cfg.oddt_models_dir == "/models"
    assert cfg.database.backend == "postgresql"


@pytest.mark.order(111)
def test_from_config_file_uses_absolute_path_from_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "env.cfg").write_text("[dummy]\n", encoding="utf-8")
    monkeypatch.setenv("OCDOCKER_CONFIG", "env.cfg")

    seen = {}

    def _parse(path):
        seen["path"] = path
        return {"vina_exhaustiveness": "7"}

    _install_fake_initialise(monkeypatch, _parse)
    occfg.OCDockerConfig.from_config_file("")
    assert seen["path"] == str((tmp_path / "env.cfg").resolve())


@pytest.mark.order(111)
def test_from_config_file_falls_back_to_local_yml_when_cfg_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "OCDocker.yml").write_text("dummy: true\n", encoding="utf-8")
    monkeypatch.setenv("OCDOCKER_CONFIG", "missing-from-env.cfg")

    seen = {}

    def _parse(path):
        seen["path"] = path
        return {"vina_exhaustiveness": "13"}

    _install_fake_initialise(monkeypatch, _parse)
    occfg.OCDockerConfig.from_config_file("")

    assert seen["path"] == str((tmp_path / "OCDocker.yml").resolve())


@pytest.mark.order(112)
def test_from_config_file_missing_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OCDOCKER_CONFIG", raising=False)
    _install_fake_initialise(monkeypatch, lambda _path: {"x": 1})

    with pytest.raises(FileNotFoundError):
        occfg.OCDockerConfig.from_config_file("")


@pytest.mark.order(112)
def test_from_config_file_uses_resolver_helper_when_available(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OCDOCKER_CONFIG", raising=False)

    seen = {}
    resolved = str((tmp_path / "resolved-by-helper.cfg").resolve())

    def _parse(path):
        seen["parse_path"] = path
        return {"vina_exhaustiveness": "6"}

    def _resolve(requested, include_package_locations=True):
        seen["requested"] = requested
        seen["include_package_locations"] = include_package_locations
        return resolved

    _install_fake_initialise(monkeypatch, _parse, _resolve)
    cfg = occfg.OCDockerConfig.from_config_file("")

    assert seen["requested"] == ""
    assert seen["include_package_locations"] is False
    assert seen["parse_path"] == resolved
    assert cfg.vina.exhaustiveness == 6


@pytest.mark.order(112)
def test_from_config_file_bubbles_resolver_not_found(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def _resolve(_requested, include_package_locations=True):
        raise FileNotFoundError("resolver-miss")

    _install_fake_initialise(monkeypatch, lambda _path: {"x": 1}, _resolve)

    with pytest.raises(FileNotFoundError, match="resolver-miss"):
        occfg.OCDockerConfig.from_config_file("")


@pytest.mark.order(113)
def test_from_config_file_empty_parse_raises(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.cfg"
    cfg_file.write_text("[dummy]\n", encoding="utf-8")
    _install_fake_initialise(monkeypatch, lambda _path: {})

    with pytest.raises(ValueError):
        occfg.OCDockerConfig.from_config_file(str(cfg_file))


@pytest.mark.order(114)
def test_from_dict_applies_sections_and_direct_fields():
    cfg = occfg.OCDockerConfig.from_dict(
        {
            "vina": {"executable": "vina-x", "exhaustiveness": 6},
            "smina": {"executable": "smina-x"},
            "gnina": {"executable": "gnina-x"},
            "plants": {"executable": "plants-x", "cluster_structures": 4},
            "database": {"host": "localhost", "port": 3307},
            "tools": {"obabel": "obabel-x"},
            "paths": {"ocdb_path": "/tmp/ocdb"},
            "output_level": ocerror.ReportLevel.ERROR,
            "multiprocess": False,
            "overwrite": True,
            "tmp_dir": "/tmp/work",
            "oddt_models_dir": "/tmp/models",
        }
    )

    assert cfg.vina.executable == "vina-x"
    assert cfg.vina.exhaustiveness == 6
    assert cfg.smina.executable == "smina-x"
    assert cfg.gnina.executable == "gnina-x"
    assert cfg.plants.cluster_structures == 4
    assert cfg.database.host == "localhost"
    assert cfg.database.port == 3307
    assert cfg.tools.obabel == "obabel-x"
    assert cfg.paths.ocdb_path == "/tmp/ocdb"
    assert cfg.output_level == ocerror.ReportLevel.ERROR
    assert cfg.multiprocess is False
    assert cfg.overwrite is True
    assert cfg.tmp_dir == "/tmp/work"
    assert cfg.oddt_models_dir == "/tmp/models"


@pytest.mark.order(115)
def test_singleton_set_get_and_reset():
    first = occfg.get_config()
    assert isinstance(first, occfg.OCDockerConfig)

    custom = occfg.OCDockerConfig.from_dict({"overwrite": True})
    occfg.set_config(custom)
    assert occfg.get_config() is custom

    occfg.reset_config()
    assert occfg.get_config() is not custom
