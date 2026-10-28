Scripts
=======

Standalone command-line tools in ``scripts/`` for one-off data curation and
provisioning tasks. Unlike the packages under ``OCDocker/``, these are not
part of the importable ``OCDocker`` package and are run directly with
``python scripts/<name>.py``.

LIT-PCBA External Validation Subset
-----------------------------------

Builds a leakage-checked, compute-tractable LIT-PCBA subset for external
blind evaluation of OCDocker/OCScore: deduplicates LIT-PCBA candidate
receptors against the local PDBbind/DUDEz archives via mmseqs2 sequence
search, picks the best-resolution surviving structure per target, and
subsamples each target's inactives.

.. literalinclude:: ../../scripts/litpcba_validation_subset.py
   :language: python
   :caption: LIT-PCBA validation subset builder

This script demonstrates:

* CA-trace sequence extraction directly from PDB/mol2 atom records (no external structure parser)
* Receptor near-duplicate detection against PDBbind/DUDEz via ``mmseqs2 search``
* Representative-receptor selection by crystallographic resolution (cached, RCSB Data API fallback)
* Deterministic, seeded inactive subsampling with a floor/cap/ratio rule
* Manifest generation (``manifest.json``/``.csv``, ``dropped_targets.tsv``, ``excluded_structures.tsv``) for full provenance

See ``docs/litpcba_validation_subset.md`` in the repository root for the full
methodology, threshold rationale, and the resulting 13-target subset table.

LIT-PCBA Raw Archive Builder
----------------------------

Converts the output of ``litpcba_validation_subset.py`` into the raw archive
layout OCDocker's ``Prepare``/``Dock`` pipeline expects for a new archive
type (see :doc:`OCDocker.DB.LITPCBA`), mirroring how the DUDEz/PDBbind
archives are laid out under ``config.paths.ocdb_path``.

.. literalinclude:: ../../scripts/litpcba_build_archive.py
   :language: python
   :caption: LIT-PCBA raw archive builder

This script demonstrates:

* Converting a mol2 receptor/reference ligand to PDB via ``obabel`` for OCDockerPipeline's box-generation step
* Splitting a whitespace-delimited SMILES list into OCDocker's pre-organized per-compound subfolder layout
* Skipping already-built target outputs unless ``--overwrite`` is passed
