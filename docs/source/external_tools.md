# External Tools

OCDocker can call external docking and preparation tools. The exact requirements
depend on the workflow you run.

## System packages

Ubuntu/Debian baseline:

```bash
sudo apt-get install openbabel libopenbabel-dev swig cmake g++
```

DSSP, when needed:

```bash
sudo apt-get install dssp
```

## Tool summary

| Tool | Used for | Notes |
| --- | --- | --- |
| OpenBabel | molecular conversion and preparation support | required by many docking workflows |
| MGLTools | Vina/Smina PDBQT preparation | configure `pythonsh`, `prepare_ligand`, and `prepare_receptor` |
| AutoDock Vina | docking/scoring | configure `vina` if not on `PATH` |
| Smina | docking/scoring | configure `smina` if not on `PATH` |
| Gnina | CNN-assisted docking/scoring | OCDocker expects a compatible Gnina binary |
| PLANTS/SPORES | PLANTS docking and preparation | configure `plants` and `spores` |
| ODDT | optional rescoring models | install through the relevant Python dependency stack |

## Gnina

OCDocker is configured for the Gnina CUDA 12.8 build. Ensure the NVIDIA driver and
runtime are compatible with that binary.

```bash
mkdir -p gnina
wget -O gnina/gnina.1.3.2.cuda12.8 \
  https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2.cuda12.8
chmod +x gnina/gnina.1.3.2.cuda12.8
sudo install -m 0755 gnina/gnina.1.3.2.cuda12.8 /usr/bin/gnina
gnina --version
```

## AutoDock Vina

```bash
mkdir -p vina
wget -O vina/vina \
  https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64
chmod +x vina/vina
sudo install -m 0755 vina/vina /usr/bin/vina
```

## MGLTools

Download MGLTools from the official Scripps distribution, extract it, and run its
installer. Then point `pythonsh`, `prepare_ligand`, and `prepare_receptor` in
`OCDocker.cfg` to the installed paths.

The preparation scripts are usually under:

```text
<installation_dir>/mgltools/MGLToolsPckgs/AutoDockTools
```

If MGLTools reports NumPy or Python path issues, check that your conda/system
Python paths are not shadowing MGLTools' bundled Python.

## GPU checks

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

If PyTorch cannot see the GPU, verify the driver, torch CUDA build, active conda
environment, and CUDA runtime compatibility.
