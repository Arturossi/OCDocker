#!/usr/bin/env python3

# Description
###############################################################################
'''
Extra tests for Toolbox.Validation helpers.
'''

# Imports
###############################################################################
import pytest

import OCDocker.Toolbox.Validation as ocvalidation

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

## Public ##

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
