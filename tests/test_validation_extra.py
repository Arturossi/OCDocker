import pytest

import OCDocker.Toolbox.Validation as ocvalidation


@pytest.mark.order(95)
def test_validate_obabel_extension_and_digest_format(tmp_path):
    # Supported ext
    assert ocvalidation.validate_obabel_extension("foo.mol2") == "mol2"
    # Unsupported ext returns an int error code
    bad = ocvalidation.validate_obabel_extension("foo.zzz")
    assert isinstance(bad, int) and bad != 0

    # Digest format validation
    assert ocvalidation.validate_digest_extension(str(tmp_path/"x.json"), "json") is True
    # Unknown format: tries to infer from path
    assert ocvalidation.validate_digest_extension(str(tmp_path/"x.json"), "foobar") is True
