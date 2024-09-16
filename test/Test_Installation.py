import unittest
import os
import OCDocker.Docking.Vina as ocvina
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsdclust
import OCDocker.Rescoring.ODDT as ocoddt
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr

import pandas as pd

class TestOCDockerPipeline(unittest.TestCase):
    """ Test the OCDocker pipeline. """
    
    def setUp(self):
        '''
        Setup the base paths and check if required files exist before running the tests.
        '''

        self.basePath = os.path.dirname(os.path.abspath(__file__))
        self.ptn = "test_ptn1"
        self.lig = "ligand"
        self.baseProtPath = f"{self.basePath}/{self.ptn}"
        self.baseLigPath = f"{self.baseProtPath}/compounds/ligands"
        self.baseDecPath = f"{self.baseProtPath}/compounds/decoys"
        self.baseCanPath = f"{self.baseProtPath}/compounds/candidates"

        # Ensure the required files exist
        self.required_files = [
            f"{self.baseProtPath}/receptor.pdb",
            f"{self.baseLigPath}/{self.lig}/ligand.smi",
            f"{self.baseLigPath}/{self.lig}/vinaFiles/conf_vina.txt",
            f"{self.baseLigPath}/{self.lig}/boxes/box0.pdb"
        ]
        for file in self.required_files:
            self.assertTrue(os.path.isfile(file), f"Required file {file} not found!")

    def test_create_ligand_object(self):
        '''
        Create a ligand object.
        '''

        ligand = ocl.Ligand(f"{self.baseLigPath}/{self.lig}/ligand.smi", name=f"{self.lig}")

        # Assert that the ligand object is created
        self.assertIsNotNone(ligand, "Failed to create ligand object.")

        self.ligand = ligand
    
    def test_create_receptor_object(self):
        '''
        Create a receptor object.
        '''

        receptor = ocr.Receptor(f"{self.baseProtPath}/receptor.pdb", relativeASAcutoff=0.7, name=f"{self.ptn}")

        # Assert that the receptor object is created
        self.assertIsNotNone(receptor, "Failed to create receptor object.")

        self.receptor = receptor

    def test_create_vina_object(self):
        '''
        Create a Vina object.
        '''

        # Assert that the ligand and receptor objects are not None
        if not hasattr(self, 'ligand'):
            self.fail("Ligand object is None.")
        
        if not hasattr(self, 'receptor'):
            self.fail("Receptor object is None.")

        vina = ocvina.Vina(
            f"{self.baseLigPath}/{self.lig}/vinaFiles/conf_vina.txt",
            f"{self.baseLigPath}/{self.lig}/boxes/box0.pdb",
            self.receptor, # type: ignore
            f"{self.baseProtPath}/prepared_receptor.pdbqt",
            self.ligand, # type: ignore
            f"{self.baseLigPath}/{self.lig}/prepared_ligand.pdbqt",
            f"{self.baseLigPath}/{self.lig}/vinaFiles/{self.lig}.log",
            f"{self.baseLigPath}/{self.lig}/vinaFiles/{self.lig}.pdbqt",
            name=f"Vina {self.ptn}-{self.lig}"
        )

        # Assert that the Vina object is created
        self.assertIsNotNone(vina, "Failed to create Vina object.")

        self.vina = vina

    def test_create_smina_object(self):
        '''
        Create a Smina object.
        '''

        # Assert that the ligand and receptor objects are not None
        if not hasattr(self, 'ligand'):
            self.fail("Ligand object is None.")
        
        if not hasattr(self, 'receptor'):
            self.fail("Receptor object is None.")

        smina = ocsmina.Smina(
            f"{self.baseLigPath}/{self.lig}/sminaFiles/conf_smina.txt",
            f"{self.baseLigPath}/{self.lig}/boxes/box0.pdb",
            self.receptor, # type: ignore
            f"{self.baseProtPath}/prepared_receptor.pdbqt",
            self.ligand, # type: ignore
            f"{self.baseLigPath}/{self.lig}/prepared_ligand.pdbqt",
            f"{self.baseLigPath}/{self.lig}/sminaFiles/{self.lig}.log",
            f"{self.baseLigPath}/{self.lig}/sminaFiles/{self.lig}.pdbqt",
            name=f"Smina {self.ptn}-{self.lig}"
        )

        # Assert that the Smina object is created
        self.assertIsNotNone(smina, "Failed to create Smina object.")

        self.smina = smina

    def test_create_plants_object(self):
        '''
        Create a PLANTS object.
        '''

        # Assert that the ligand and receptor objects are not None
        if not hasattr(self, 'ligand'):
            self.fail("Ligand object is None.")
        
        if not hasattr(self, 'receptor'):
            self.fail("Receptor object is None.")

        plants = ocplants.PLANTS(
            f"{self.baseLigPath}/{self.lig}/plantsFiles/conf_plants.txt",
            f"{self.baseLigPath}/{self.lig}/boxes/box0.pdb",
            self.receptor, # type: ignore
            f"{self.baseProtPath}/prepared_receptor.mol2",
            self.ligand, # type: ignore
            f"{self.baseLigPath}/{self.lig}/prepared_ligand.mol2",
            f"{self.baseLigPath}/{self.lig}/plantsFiles/{self.lig}.log",
            f"{self.baseLigPath}/{self.lig}/plantsFiles",
            name=f"PLANTS {self.ptn}-{self.lig}"
        )

        # Assert that the PLANTS object is created
        self.assertIsNotNone(plants, "Failed to create PLANTS object.")

        self.plants = plants

    def test_vina_docking(self):
        '''
        Test Vina docking preparation, docking, and rescoring.
        '''

        # Assert that the vina object is not None
        if not hasattr(self, 'vina'):
            self.fail("Vina object is None.")

        # Test preparation
        self.vina.run_prepare_receptor()
        self.vina.run_prepare_ligand()

        # Test docking
        self.vina.run_docking()

        # Test pose splitting
        self.vina.split_poses(f"{self.baseLigPath}/{self.lig}/vinaFiles", logFile="")

        # Test rescoring
        self.vina.run_rescore(f"{self.baseLigPath}/{self.lig}/vinaFiles", skipDefaultScoring=True)

        # Ensure docking and rescoring logs are readable
        docking_result = self.vina.read_log()
        rescore_result = self.vina.read_rescore_logs(f"{self.baseLigPath}/{self.lig}/vinaFiles")
        self.assertTrue(docking_result, "Vina docking log is empty.")
        self.assertTrue(rescore_result, "Vina rescoring log is empty.")

    def test_smina_docking(self):
        '''
        Test Smina docking preparation, docking, and rescoring.
        '''

        # Assert that the smina object is not None
        if not hasattr(self, 'smina'):
            self.fail("Smina object is None.")

        # Test preparation
        self.smina.run_prepare_receptor()
        self.smina.run_prepare_ligand()

        # Test docking
        self.smina.run_docking()

        # Test pose splitting
        self.smina.split_poses(f"{self.baseLigPath}/{self.lig}/sminaFiles", logFile="")

        # Test rescoring
        self.smina.run_rescore(f"{self.baseLigPath}/{self.lig}/sminaFiles", skipDefaultScoring=True)

        # Ensure docking and rescoring logs are readable
        docking_result = self.smina.read_log()
        rescore_result = self.smina.read_rescore_logs(f"{self.baseLigPath}/{self.lig}/sminaFiles")
        self.assertTrue(docking_result, "Smina docking log is empty.")
        self.assertTrue(rescore_result, "Smina rescoring log is empty.")
    
    def test_plants_docking(self):
        '''
        Test PLANTS docking, clustering, rescoring, and exporting.
        '''

        # Assert that the plants object is not None
        if not hasattr(self, 'plants'):
            self.fail("PLANTS object is None.")

        # Test preparation steps
        self.plants.run_prepare_receptor()
        self.plants.run_prepare_ligand()

        # Test docking
        self.plants.run_docking()

        # Test docking results
        docking_result = self.plants.read_log(onlyBest=False)
        self.assertTrue(docking_result, "PLANTS docking log is empty.")

        # Get docking poses
        plantsPoses = self.plants.get_docked_poses()

        # Write the pose list file for rescoring
        pose_list = self.plants.write_pose_list()

        # Assert that the pose list is not None
        self.assertTrue(pose_list, "Failed to write pose list for PLANTS.")

        # Test rescoring
        self.plants.run_rescore(pose_list, logFile="", overwrite=False) # type: ignore

        # Test rescoring results
        rescore_result = self.plants.read_rescore_logs(onlyBest=False)
        self.assertTrue(rescore_result, "PLANTS rescoring log is empty.")

        # Save docking poses for clustering
        self.docking_poses = {
            'vina': self.vina.get_docked_poses(),
            'plants': plantsPoses
        }
        print(self.docking_poses)

    def test_clustering(self):
        '''
        Test clustering of the docking poses.
        '''

        # Assert that the docking poses are not None
        if not hasattr(self, 'docking_poses'):
            self.fail("Docking poses are None.")

        vinaPoses = self.docking_poses['vina']
        plantsPoses = self.docking_poses['plants']

        # Test clustering
        rmsdMatrix = ocmolproc.get_rmsd_matrix(vinaPoses + plantsPoses)
        clusters = ocrmsdclust.cluster_rmsd(rmsdMatrix, algorithm='agglomerativeClustering')

        # Ensure that clusters is np.ndarray
        self.assertTrue(clusters, "Failed to cluster poses.")

        # Test medoids extraction
        medoids = ocrmsdclust.get_medoids(rmsdMatrix, clusters, onlyBiggest=True)  # type: ignore
        self.assertTrue(medoids, "Failed to get medoids.")

        # Save medoids for rescoring
        self.medoids = medoids

    def test_oddt_rescoring(self):
        '''
        Test rescoring using ODDT.
        '''

        if not hasattr(self, 'medoids'):
            self.fail("Medoids not prepared; ensure test_clustering is run first.")

        # Assuming medoids contains the paths or data required for rescoring
        dockedPoses = self.medoids

        df = ocoddt.run_oddt(f"{self.baseProtPath}/prepared_receptor.pdb", dockedPoses, self.lig, f"{self.baseLigPath}/{self.lig}")

        # Assert that df is a dataframe
        self.assertTrue(isinstance(df, pd.DataFrame), "ODDT rescoring did not return a dataframe.")

        # If you want a dict, you can convert with this function
        dt = ocoddt.df_to_dict(df) # type: ignore
        self.assertIsNotNone(dt, "ODDT rescoring did not return any results.")

if __name__ == '__main__':
    unittest.main()
