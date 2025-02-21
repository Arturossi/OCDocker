import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pymysql
import unittest
import pandas as pd
from urllib.parse import quote_plus

import OCDocker.Docking.Vina as ocvina
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Rescoring.ODDT as ocoddt
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsdclust
import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr

from OCDocker.Initialise import *


class TestOCDockerPipeline(unittest.TestCase):
    """ Test the OCDocker pipeline. """

    @classmethod
    def setUpClass(cls):
        """
        Setup base paths and database connection.
        """
        cls.basePath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../test_files"))

        cls.ptn = "test_ptn1"
        cls.lig = "ligand"
        cls.baseProtPath = os.path.join(cls.basePath, cls.ptn)
        cls.baseLigPath = os.path.join(cls.baseProtPath, "compounds", "ligands")

        # Setup storage for objects but don't create them yet
        cls.ligand = None
        cls.receptor = None
        cls.vina = None
        cls.smina = None
        cls.plants = None
        cls.docking_objects = {}
        cls.medoids = None

        # Database connection
        cls.db_config = {
            "host": "192.168.101.2",
            "user": "ocdocker",
            "password": "@Kp3sRv9t@",
            "database": "optimization",
            "port": 3306
        }

        try:
            connection = pymysql.connect(**cls.db_config)
            connection.close()
        except pymysql.MySQLError as e:
            raise unittest.SkipTest(f"Skipping tests. Unable to connect to the database: {e}")

    def test_01_database_connection(self):
        """ Test the database connection. """
        try:
            connection = pymysql.connect(**self.__class__.db_config)
            connection.close()
        except pymysql.MySQLError as e:
            self.fail(f"Database connection failed: {e}")

    def test_02_create_ligand_object(self):
        """ Create and store the ligand object. """
        cls = self.__class__
        cls.ligand = ocl.Ligand(f"{cls.baseLigPath}/{cls.lig}/ligand.smi", name=cls.lig)
        self.assertIsNotNone(cls.ligand, "Failed to create ligand object.")

    def test_03_create_receptor_object(self):
        """ Create and store the receptor object. """
        cls = self.__class__
        cls.receptor = ocr.Receptor(f"{cls.baseProtPath}/receptor.pdb", relativeASAcutoff=0.7, name=cls.ptn)
        self.assertIsNotNone(cls.receptor, "Failed to create receptor object.")

    def test_04_create_vina_object(self):
        """ Create a Vina object. """
        cls = self.__class__
        if cls.ligand is None or cls.receptor is None:
            self.fail("Ligand or receptor not created. Ensure `test_02_create_ligand_object` and `test_03_create_receptor_object` run first.")

        cls.vina = ocvina.Vina(
            f"{cls.baseLigPath}/{cls.lig}/vinaFiles/conf_vina.txt",
            f"{cls.baseLigPath}/{cls.lig}/boxes/box0.pdb",
            cls.receptor,
            f"{cls.baseProtPath}/prepared_receptor.pdbqt",
            cls.ligand,
            f"{cls.baseLigPath}/{cls.lig}/prepared_ligand.pdbqt",
            f"{cls.baseLigPath}/{cls.lig}/vinaFiles/{cls.lig}.log",
            f"{cls.baseLigPath}/{cls.lig}/vinaFiles/{cls.lig}.pdbqt",
            name=f"Vina {cls.ptn}-{cls.lig}"
        )

        self.assertIsNotNone(cls.vina, "Failed to create Vina object.")
    
    def test_05_create_smina_object(self):
        """ Create a Smina object. """
        cls = self.__class__
        if cls.ligand is None or cls.receptor is None:
            self.fail("Ligand or receptor not created. Ensure previous tests run first.")

        cls.smina = ocsmina.Smina(
            f"{cls.baseLigPath}/{cls.lig}/sminaFiles/conf_smina.txt",
            f"{cls.baseLigPath}/{cls.lig}/boxes/box0.pdb",
            cls.receptor,
            f"{cls.baseProtPath}/prepared_receptor.pdbqt",
            cls.ligand,
            f"{cls.baseLigPath}/{cls.lig}/prepared_ligand.pdbqt",
            f"{cls.baseLigPath}/{cls.lig}/sminaFiles/{cls.lig}.log",
            f"{cls.baseLigPath}/{cls.lig}/sminaFiles/{cls.lig}.pdbqt",
            name=f"Smina {cls.ptn}-{cls.lig}"
        )

        self.assertIsNotNone(cls.smina, "Failed to create Smina object.")
    
    def test_06_create_plants_object(self):
        """ Create a PLANTS object. """
        cls = self.__class__
        if cls.ligand is None or cls.receptor is None:
            self.fail("Ligand or receptor not created. Ensure previous tests run first.")

        cls.plants = ocplants.PLANTS(
            f"{cls.baseLigPath}/{cls.lig}/plantsFiles/conf_plants.txt",
            f"{cls.baseLigPath}/{cls.lig}/boxes/box0.pdb",
            cls.receptor,
            f"{cls.baseProtPath}/prepared_receptor.mol2",
            cls.ligand,
            f"{cls.baseLigPath}/{cls.lig}/prepared_ligand.mol2",
            f"{cls.baseLigPath}/{cls.lig}/plantsFiles/{cls.lig}.log",
            f"{cls.baseLigPath}/{cls.lig}/plantsFiles",
            name=f"PLANTS {cls.ptn}-{cls.lig}"
        )

        self.assertIsNotNone(cls.plants, "Failed to create PLANTS object.")

    def test_07_vina_docking(self):
        """ Test Vina docking. """
        cls = self.__class__
        if cls.vina is None:
            self.fail("Vina object not created. Ensure `test_04_create_vina_object` runs first.")

        cls.vina.run_prepare_receptor()
        cls.vina.run_prepare_ligand()
        cls.vina.run_docking()
        cls.vina.split_poses(f"{cls.baseLigPath}/{cls.lig}/vinaFiles", logFile="")
        cls.vina.run_rescore(f"{cls.baseLigPath}/{cls.lig}/vinaFiles", skipDefaultScoring=True, overwrite=True)

        docking_result = cls.vina.read_log()
        rescore_result = cls.vina.read_rescore_logs(f"{cls.baseLigPath}/{cls.lig}/vinaFiles")

        self.assertTrue(docking_result, "Vina docking log is empty.")
        self.assertTrue(rescore_result, "Vina rescoring log is empty.")

    def test_08_smina_docking(self):
        """ Test Smina docking. """
        cls = self.__class__
        if cls.smina is None:
            self.fail("smina object not created. Ensure `test_04_create_smina_object` runs first.")

        cls.smina.run_prepare_receptor()
        cls.smina.run_prepare_ligand()
        cls.smina.run_docking()
        cls.smina.split_poses(f"{cls.baseLigPath}/{cls.lig}/sminaFiles", logFile="")
        cls.smina.run_rescore(f"{cls.baseLigPath}/{cls.lig}/sminaFiles", skipDefaultScoring=True, overwrite=True)

        docking_result = cls.smina.read_log()
        rescore_result = cls.smina.read_rescore_logs(f"{cls.baseLigPath}/{cls.lig}/sminaFiles")

        self.assertTrue(docking_result, "Smina docking log is empty.")
        self.assertTrue(rescore_result, "Smina rescoring log is empty.")
    
    def test_09_plants_docking(self):
        """ Test Plants docking. """
        cls = self.__class__
        if cls.plants is None:
            self.fail("Plants object not created. Ensure `test_04_create_smina_object` runs first.")

        cls.plants.run_prepare_receptor()
        cls.plants.run_prepare_ligand()
        cls.plants.run_docking()
        cls.plants.split_poses(f"{cls.baseLigPath}/{cls.lig}/plantsFiles", logFile="")
        cls.plants.run_rescore(f"{cls.baseLigPath}/{cls.lig}/plantsFiles", skipDefaultScoring=True)

        docking_result = cls.plants.read_log()
        rescore_result = cls.plants.read_rescore_logs(f"{cls.baseLigPath}/{cls.lig}/plantsFiles")

        self.assertTrue(docking_result, "Plants docking log is empty.")
        self.assertTrue(rescore_result, "Plants rescoring log is empty.")

    def test_10_docking_files_created(self):
        """ Ensure docking directories exist. """
        cls = self.__class__
        required_folders = [
            os.path.join(cls.baseLigPath, cls.lig, "vinaFiles"),
            os.path.join(cls.baseLigPath, cls.lig, "sminaFiles"),
            os.path.join(cls.baseLigPath, cls.lig, "plantsFiles")
        ]

        for folder in required_folders:
            self.assertTrue(os.path.exists(folder), f"Missing required docking folder: {folder}")

    def test_11_clustering(self):
        """ Test clustering. """
        cls = self.__class__
        if cls.vina is None:
            self.fail("Vina object not created. Ensure `test_04_create_vina_object` runs first.")

        vina_poses = cls.vina.get_docked_poses()
        self.assertTrue(vina_poses, "No Vina docking poses found.")

        rmsd_matrix = ocmolproc.get_rmsd_matrix(vina_poses)
        clusters = ocrmsdclust.cluster_rmsd(rmsd_matrix, algorithm="agglomerativeClustering")
        self.assertTrue(isinstance(clusters, list) and len(clusters) > 0, "Failed to cluster docking poses.")

        cls.medoids = ocrmsdclust.get_medoids(rmsd_matrix, clusters, onlyBiggest=True)
        self.assertTrue(cls.medoids, "Failed to extract medoids.")

    def test_12_oddt_rescoring(self):
        """ Test rescoring using ODDT. """
        cls = self.__class__
        if cls.medoids is None:
            self.fail("Medoids not prepared. Ensure `test_07_clustering` runs first.")

        df = ocoddt.run_oddt(
            f"{cls.baseProtPath}/prepared_receptor.pdbqt",
            cls.medoids,
            cls.lig,
            f"{cls.baseLigPath}/{cls.lig}"
        )

        self.assertTrue(isinstance(df, pd.DataFrame), "ODDT rescoring did not return a dataframe.")

if __name__ == '__main__':
    suite = unittest.TestSuite()

    # Explicitly enforce test order
    suite.addTest(TestOCDockerPipeline("test_01_database_connection"))
    suite.addTest(TestOCDockerPipeline("test_02_create_ligand_object"))
    suite.addTest(TestOCDockerPipeline("test_03_create_receptor_object"))
    suite.addTest(TestOCDockerPipeline("test_04_create_vina_object"))
    suite.addTest(TestOCDockerPipeline("test_05_create_smina_object"))
    suite.addTest(TestOCDockerPipeline("test_06_create_plants_object"))
    suite.addTest(TestOCDockerPipeline("test_07_vina_docking"))
    suite.addTest(TestOCDockerPipeline("test_08_smina_docking"))
    suite.addTest(TestOCDockerPipeline("test_09_plants_docking"))
    suite.addTest(TestOCDockerPipeline("test_10_docking_files_created"))
    suite.addTest(TestOCDockerPipeline("test_11_clustering"))
    suite.addTest(TestOCDockerPipeline("test_12_oddt_rescoring"))

    runner = unittest.TextTestRunner()
    runner.run(suite)
