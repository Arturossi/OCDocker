import pytest
import shutil

from pathlib import Path

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr

import OCDocker.Docking.Smina as ocsmina
import OCDocker.Toolbox.Conversion as occonversion


@pytest.fixture
def smina_inputs():
    base = Path("test_files/test_ptn1")

    pre_output_dir = base / "compounds/ligands/ligand"

    output_dir = pre_output_dir / "sminaFiles"

    pre_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "smina_out.pdbqt"

    receptor_file = base / "receptor.pdb"
    ligand_file = pre_output_dir / "ligand.smi"
    converted_ligand_file = pre_output_dir / "ligand.mol2"
    box_file = pre_output_dir / "boxes/box0.pdb"

    config_file = output_dir / "smina_config.txt"
    config_file.write_text("receptor = prepared_receptor.pdb\nligand = prepared_ligand.pdb\n")

    prepared_receptor_path = base / "prepared_receptor.pdbqt"
    prepared_ligand_path = pre_output_dir / "prepared_ligand.pdbqt"
    smina_log = output_dir / "smina.log"
    
    # If there are already smina folders, remove them
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    receptor = ocr.Receptor(structure=str(receptor_file), name="test_rec")
    ligand = ocl.Ligand(molecule=str(ligand_file), name="test_lig")
    
    return {
        "config": str(config_file),
        "box": str(box_file),
        "receptor": receptor,
        "receptor_file": str(receptor_file),
        "receptor_path": str(prepared_receptor_path),
        "ligand": ligand,
        "ligand_file": str(ligand_file),
        "ligand_path": str(prepared_ligand_path),
        "converted_ligand_file": str(converted_ligand_file),
        "output_dir": output_dir,
        "output": str(output_file),
        "smina_log": str(smina_log)
    }

@pytest.mark.order(1)
def test_smina_instantiation(smina_inputs):
    '''
    Test Smina class can be instantiated with all required real inputs.
    '''
    smina_instance = ocsmina.Smina(
        configPath=smina_inputs["config"],
        boxFile=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        preparedReceptorPath=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        preparedLigandPath=smina_inputs["ligand_path"],
        sminaLog=smina_inputs["smina_log"],
        outputSmina=smina_inputs["output"],
        name="test"
    )
    assert isinstance(smina_instance, ocsmina.Smina)

@pytest.mark.order(2)
def test_convert_smi_to_mol2(smina_inputs):
    '''
    Test explicit call to convert .smi to .mol2 using Conversion.py routine.
    '''

    result = occonversion.convertMols(
        input_file=str(smina_inputs["ligand_file"]),
        output_file=str(smina_inputs["converted_ligand_file"]),
        overwrite=True
    )

    assert result == 0 or result is True
    assert Path(smina_inputs["converted_ligand_file"]).exists(), "Failed to generate .mol2 from .smi"

@pytest.mark.order(3)
def test_run_prepare_ligand(smina_inputs):
    '''
    Run ligand preparation and check that it produces expected files.
    '''

    out = Path(smina_inputs["ligand_path"]).parent
    out.mkdir(parents=True, exist_ok=True)

    output_ligand = out / "prep_ligand.pdbqt"

    result = ocsmina.run_prepare_ligand(
        inputLigandPath=smina_inputs["converted_ligand_file"],
        preparedLigand=str(output_ligand)
    )

    assert result is True or isinstance(result, int)
    assert output_ligand.exists(), "Prepared ligand file not found"

@pytest.mark.order(4)
def test_run_prepare_receptor(smina_inputs):
    '''
    Run receptor preparation and check that it produces expected files.
    '''
    out = smina_inputs["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    output_receptor = out / "prep_receptor.pdbqt"

    result = ocsmina.run_prepare_receptor(
        inputReceptorPath=smina_inputs["receptor_file"],
        preparedReceptor=str(output_receptor)
    )

    assert result is True or isinstance(result, int)
    assert output_receptor.exists(), "Prepared receptor file not found"

@pytest.mark.order(5)
def test_run_smina(smina_inputs):
    '''
    Run Smina docking using real prepared ligand and receptor.
    '''
    
    smina_inputs["output_dir"].mkdir(parents=True, exist_ok=True)

    result = ocsmina.run_smina(
        config=smina_inputs["config"],
        preparedLigand=smina_inputs["converted_ligand_file"],
        outputSmina=smina_inputs["output"],
        sminaLog=smina_inputs["smina_log"],
        logPath=smina_inputs["smina_log"]
    )

    assert result is True or isinstance(result, int)
    assert Path(smina_inputs["output"]).exists(), "Docking output file not created"
