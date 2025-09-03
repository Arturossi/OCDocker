import json
import os
from pathlib import Path


def test_vina_log_and_rescoring_parsing(tmp_path, monkeypatch):
    # Avoid heavy Initialise auto-bootstrap on import
    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    log = tmp_path / "vina.log"
    # Header first, then rows (reverse reader stops at header when it reaches it)
    log.write_text("""-----+ header
1 -7.50 0 0
2 -6.20 0 0
""")
    data = basevina.read_vina_log(str(log))
    assert set(data.keys()) == {1, 2}
    assert data[1][basevina.vina_scoring] == "-7.50"  # type: ignore

    best_only = basevina.read_vina_log(str(log), onlyBest=True)
    assert set(best_only.keys()) == {1}

    resc = tmp_path / "rescore.log"
    resc.write_text("Estimated Free Energy of Binding    -7.23 (kcal/mol)\n")
    val = basevina.read_vina_rescoring_log(str(resc))
    assert val == -7.23

    out_json = tmp_path / "digest.json"
    rc = basevina.generate_vina_digest(str(out_json), str(log), overwrite=True, digestFormat="json")
    assert rc == 0 and out_json.exists()
    j = json.loads(out_json.read_text())
    # Top level contains base keys and pose keys as strings
    assert "vina_affinity" in j
    assert "1" in j and j["1"][basevina.vina_scoring] == "-7.50"  # type: ignore


def test_smina_log_and_rescoring_parsing(tmp_path, monkeypatch):
    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    log = tmp_path / "smina.log"
    log.write_text("""-----+ header
1 -8.00 0 0
3 -6.75 0 0
""")
    data = basevina.read_smina_log(str(log))
    assert set(data.keys()) == {1, 3}
    assert data[3][basevina.smina_scoring] == "-6.75"  # type: ignore

    resc = tmp_path / "rescore_smina.log"
    resc.write_text("Affinity: -6.71 (kcal/mol)\n")
    val = basevina.read_smina_rescoring_log(str(resc))
    assert val == -6.71

    out_json = tmp_path / "digest_smina.json"
    rc = basevina.generate_smina_digest(str(out_json), str(log), overwrite=True, digestFormat="json")
    assert rc == 0 and out_json.exists()
    j = json.loads(out_json.read_text())
    assert "smina_affinity" in j
    assert "1" in j
