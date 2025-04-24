
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)  # Set RDKit logger to ERROR level, ignoring warnings


import os
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from tqdm import tqdm
from sklearn.model_selection import ShuffleSplit
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import sys
import types
from typing import *
from collections import defaultdict
import abc



# Add dataset path (adjust as needed for your environment)
sys.path.append('/kaggle/input/bdmh-dataset')

# Import original machine_learning_models.py classes
from machine_learning_models import Dataset, MLModel, Model_Evaluation, set_global_determinism

# Fingerprints implementation (copied from fingerprints.py)
def construct_check_mol_list(smiles_list: List[str]) -> List[Chem.Mol]:
    mol_obj_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    if None in mol_obj_list:
        invalid_smiles = []
        for smiles, mol_obj in zip(smiles_list, mol_obj_list):
            if not mol_obj:
                invalid_smiles.append(smiles)
        invalid_smiles = "\n".join(invalid_smiles)
        raise ValueError(f"Following smiles are not valid:\n {invalid_smiles}")
    return mol_obj_list

class Fingerprint(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def n_bits(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, mol_obj_list: List[Chem.Mol]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def fit_transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        raise NotImplementedError

    def fit_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        self.fit(mol_obj_list)

    def fit_transform_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.fit_transform(mol_obj_list)

    def transform_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.transform(mol_obj_list)

class _MorganFingerprint(Fingerprint):
    def __init__(self, radius: int = 2, use_features=False):
        super().__init__()
        self._n_bits = None
        self._use_features = use_features
        if isinstance(radius, int) and radius >= 0:
            self._radius = radius
        else:
            raise ValueError(f"Number of bits has to be a positive integer! (Received: {radius})")

    def __len__(self):
        return self.n_bits

    @property
    def n_bits(self) -> int:
        if self._n_bits is None:
            raise ValueError("Number of bits is undetermined!")
        return self._n_bits

    @property
    def radius(self):
        return self._radius

    @property
    def use_features(self) -> bool:
        return self._use_features

    @abc.abstractmethod
    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict:
        raise NotImplementedError

class FoldedMorganFingerprint(_MorganFingerprint):
    def __init__(self, n_bits=2048, radius: int = 2, use_features=False):
        super().__init__(radius=radius, use_features=use_features)
        if isinstance(n_bits, int) and n_bits >= 0:
            self._n_bits = n_bits
        else:
            raise ValueError(f"Number of bits has to be a positive integer! (Received: {n_bits})")

    def fit(self, mol_obj_list: List[Chem.Mol]) -> None:
        pass

    def transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        fingerprints = []
        for mol in mol_obj_list:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, useFeatures=self._use_features,
                                                       nBits=self._n_bits)
            fingerprints.append(sparse.csr_matrix(fp))
        return sparse.vstack(fingerprints)

    def fit_transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        return self.transform(mol_obj_list)

    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict:
        bi = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(mol_obj, self.radius, useFeatures=self._use_features, bitInfo=bi,
                                                  nBits=self._n_bits)
        return bi

def ECFP4(smiles_list: List[str]) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
    mols = construct_check_mol_list(smiles_list)
    return [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]

# Mock ml_utils with implementations aligned with the original
def tanimoto_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
    intersection = matrix_a.dot(matrix_b.transpose()).toarray()
    norm_1 = np.array(matrix_a.multiply(matrix_a).sum(axis=1))
    norm_2 = np.array(matrix_b.multiply(matrix_b).sum(axis=1))
    union = norm_1 + norm_2.T - intersection
    return intersection / union

def maxminpicker(fp_list, fraction, seed=None):
    from rdkit.SimDivFilters import MaxMinPicker
    picker = MaxMinPicker()
    n_select = int(fraction * len(fp_list))
    def dist_func(i, j):
        return 1 - DataStructs.TanimotoSimilarity(fp_list[i], fp_list[j])
    if seed is not None:
        np.random.seed(seed)
    picked = picker.LazyPick(dist_func, len(fp_list), n_select, seed=seed)
    return list(picked)

def create_directory(path: str, verbose: bool = False):
    os.makedirs(path, exist_ok=True)
    if verbose:
        print(f"Created directory: {path}")
    return path

# Mock ml_utils module
ml_utils_module = types.ModuleType('ml_utils')
sys.modules['ml_utils'] = ml_utils_module
ml_utils_module.tanimoto_from_sparse = tanimoto_from_sparse
ml_utils_module.maxminpicker = maxminpicker
ml_utils_module.create_directory = create_directory

# Define paths
results_path_reg = '/kaggle/working/regression_results/regular/'
os.makedirs(results_path_reg, exist_ok=True)

# Model Parameters
model_list = ['kNN']  # Only train kNN model
cv_folds = 2  # Match your code
opt_metric = "neg_mean_absolute_error"
data_order = ['regular', 'y_rand']  # Include y-randomization
compound_sets = ['Complete set', 'Random set', 'Diverse set']
compound_sets_size = 0.25

# Load Data
regression_db = pd.read_csv("/kaggle/input/bdmh-dataset/chembl_30_IC50_10_tids_1000_CPDs.csv")
regression_tids = regression_db.chembl_tid.unique()[:10]

# Initialize DataFrames
performance_train_df = pd.DataFrame()
performance_test_df = pd.DataFrame()
predictions_test_df = pd.DataFrame()
parameter_resume = []

# Generate Molecular Fingerprints
morgan_radius2 = FoldedMorganFingerprint(radius=2, n_bits=2048)
morgan_radius2.fit_smiles(regression_db.nonstereo_aromatic_smiles.tolist())

# Training Loop
for data_ord in data_order:
    for target in tqdm(regression_tids, desc=f"Processing targets ({data_ord})"):
        for approach in compound_sets:
            for i in range(3):
                print(f'Training on {target} - {approach} - {data_ord}')
                regression_db_tid = regression_db.loc[regression_db.chembl_tid == target]
                fp_matrix = morgan_radius2.transform_smiles(regression_db_tid.nonstereo_aromatic_smiles.tolist())
                potency = regression_db_tid.pPot.values.copy()
                
                # Y-randomization
                if data_ord == "y_rand":
                    np.random.shuffle(potency)
                
                # Create dataset
                dataset = Dataset(features=fp_matrix, labels=potency)
                dataset.add_instance("target", regression_db_tid.chembl_tid.values)
                dataset.add_instance("smiles", regression_db_tid.nonstereo_aromatic_smiles.values)

                # Subset for Random or Diverse sets
                if approach == 'Diverse set':
                    fp_bit_vec = ECFP4(regression_db_tid.nonstereo_aromatic_smiles.tolist())
                    mol_idx = maxminpicker(fp_bit_vec, compound_sets_size, seed=i+1)
                    dataset = dataset[mol_idx]
                elif approach == 'Random set':
                    np.random.seed(i+1)
                    mol_idx = np.random.choice(range(dataset.features.shape[0]), 
                                              size=int(compound_sets_size * dataset.features.shape[0]), 
                                              replace=False)
                    dataset = dataset[mol_idx]

                # Data splitting with ShuffleSplit
                data_splitter = ShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=20021997)
                for trial, (train_idx, test_idx) in enumerate(data_splitter.split(dataset.features)):
                    training_set = dataset[train_idx]
                    test_set = dataset[test_idx]
                    set_global_determinism(seed=trial)
                    print(f"Trial {trial} - Training set features shape: {training_set.features.shape}, type: {type(training_set.features)}")

                    for model in model_list:
                        print(f'Training {model}')
                        model_fpath = create_directory(f"/kaggle/working/trained_models/{model}/", verbose=False)
                        ml_model = MLModel(training_set, model, reg_class="regression", parameters='grid', cv_fold=cv_folds, random_seed=trial)
                        model_fpath += f"{target}_{trial}.sav"
                        pickle.dump(ml_model, open(model_fpath, 'wb'))

                        # Save best parameters
                        opt_parameters_dict = {'model': model, 'trial': trial, 'Target ID': target, 'Approach': approach}
                        for param, value in ml_model.best_params.items():
                            opt_parameters_dict[param] = value
                        parameter_resume.append(opt_parameters_dict)

                        # Evaluate training performance
                        model_eval_train = Model_Evaluation(ml_model, training_set, training_set.labels, model_id=model)
                        performance_train = model_eval_train.pred_performance
                        performance_train["trial"] = trial
                        performance_train["Approach"] = approach
                        performance_train["Approach_trial"] = i
                        performance_train["data_order"] = data_ord
                        performance_train_df = pd.concat([performance_train_df, performance_train], ignore_index=True)

                        # Evaluate test performance
                        model_eval_test = Model_Evaluation(ml_model, test_set, test_set.labels, model_id=model)
                        performance_test = model_eval_test.pred_performance
                        performance_test["trial"] = trial
                        performance_test["Approach"] = approach
                        performance_test["Approach_trial"] = i
                        performance_test["data_order"] = data_ord
                        performance_test_df = pd.concat([performance_test_df, performance_test], ignore_index=True)

                        # Save test predictions
                        predictions_test = model_eval_test.predictions
                        predictions_test["trial"] = trial
                        predictions_test["Approach"] = approach
                        predictions_test["Approach_trial"] = i
                        predictions_test["data_order"] = data_ord
                        predictions_test_df = pd.concat([predictions_test_df, predictions_test], ignore_index=True)

                        # Print test MAE for monitoring
                        mae = performance_test['Value'][performance_test['Metric'] == 'MAE'].values[0]
                        print(f"Test MAE for {model} on {target}, approach {approach}, trial {trial}: {mae}")

                if approach == 'Complete set':
                    break

    # Save results
    parameter_df = pd.DataFrame(parameter_resume)
    result_path = create_directory(results_path_reg)
    performance_train_df.to_csv(os.path.join(result_path, 'performance_train.csv'), index=False)
    performance_test_df.to_csv(os.path.join(result_path, 'performance_test.csv'), index=False)
    parameter_df.to_csv(os.path.join(result_path, 'model_best_parameters.csv'), index=False)
    predictions_test_df.to_csv(os.path.join(result_path, 'predictions_test.csv'), index=False)
