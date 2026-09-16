
import os
import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors,
    Lipinski,
    Crippen,
    QED,
    rdMolDescriptors,
    rdFingerprintGenerator,
)

from rdkit.ML.Descriptors import MoleculeDescriptors

from mordred import Calculator, descriptors


# ============================================================
# FINAL CATEGORY THRESHOLDS
# ============================================================

VERY_GOOD_CUTOFF = -6.45
GOOD_CUTOFF = -5.70
MEDIUM_CUTOFF = -4.945


def docking_category(score):

    score = float(score)

    if score < VERY_GOOD_CUTOFF:
        return "VERY_GOOD"

    elif score < GOOD_CUTOFF:
        return "GOOD"

    elif score < MEDIUM_CUTOFF:
        return "MEDIUM"

    else:
        return "POOR"


# ============================================================
# MAIN PREDICTOR CLASS
# ============================================================

class PLS6WebPredictor:

    def __init__(self, model_dir):

        self.model_dir = model_dir

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model_file = os.path.join(
            model_dir,
            "OpenPLS_RDKit_Mordred_PLS6.joblib"
        )

        prep_file = os.path.join(
            model_dir,
            "preprocessing_parameters.joblib"
        )

        feature_file = os.path.join(
            model_dir,
            "final_931_feature_manifest.csv"
        )

        train_file = os.path.join(
            model_dir,
            "training_reference_607.csv"
        )

        fp_file = os.path.join(
            model_dir,
            "training_reference_morgan2048.npz"
        )

        required_files = [
            model_file,
            prep_file,
            feature_file,
            train_file,
            fp_file
        ]

        for path in required_files:

            if not os.path.exists(path):

                raise FileNotFoundError(
                    f"Required model asset missing: {path}"
                )

        self.model = joblib.load(
            model_file
        )

        self.prep = joblib.load(
            prep_file
        )

        self.feature_manifest = pd.read_csv(
            feature_file
        )

        self.train_ref = pd.read_csv(
            train_file
        )

        self.train_morgan = sparse.load_npz(
            fp_file
        )


        # ----------------------------------------------------
        # RDKit descriptor calculator
        #
        # IMPORTANT:
        # use exact descriptor names saved at training time
        # ----------------------------------------------------

        self.rdkit_raw_names = list(
            self.prep[
                "rdkit_raw_descriptor_names"
            ]
        )

        self.rdkit_calculator = (
            MoleculeDescriptors
            .MolecularDescriptorCalculator(
                self.rdkit_raw_names
            )
        )


        # ----------------------------------------------------
        # Mordred calculator
        # ----------------------------------------------------

        self.mordred_calculator = Calculator(
            descriptors,
            ignore_3D=True
        )


        # ----------------------------------------------------
        # Morgan generator
        # ----------------------------------------------------

        self.morgan_generator = (
            rdFingerprintGenerator
            .GetMorganGenerator(
                radius=2,
                fpSize=2048
            )
        )


        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        if self.model.n_components != 6:

            raise ValueError(
                "Loaded model is not PLS6."
            )

        if len(
            self.feature_manifest
        ) != 931:

            raise ValueError(
                "Final feature manifest does not contain 931 features."
            )


    # ========================================================
    # MOLECULE PARSING
    # ========================================================

    def _parse_smiles(self, smiles):

        if smiles is None:
            raise ValueError(
                "SMILES is empty."
            )

        smiles = str(
            smiles
        ).strip()

        if not smiles:
            raise ValueError(
                "SMILES is empty."
            )

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:

            raise ValueError(
                "Invalid SMILES."
            )

        canonical = (
            Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True
            )
        )

        return mol, canonical


    # ========================================================
    # RDKIT DESCRIPTORS
    # ========================================================

    def _rdkit_features(self, mol):

        values = (
            self.rdkit_calculator
            .CalcDescriptors(
                mol
            )
        )

        x = np.asarray(
            values,
            dtype=np.float64
        )

        x[
            np.isinf(x)
        ] = np.nan


        # --------------------------------------------
        # Apply exact train-time mask
        # --------------------------------------------

        keep = np.asarray(
            self.prep[
                "rdkit_keep_mask"
            ],
            dtype=bool
        )

        x = x[
            keep
        ]


        # --------------------------------------------
        # Impute using TRAINING medians
        # --------------------------------------------

        medians = np.asarray(
            self.prep[
                "rdkit_train_medians"
            ],
            dtype=float
        )

        if len(x) != len(medians):

            raise ValueError(
                "RDKit preprocessing dimension mismatch."
            )

        missing = np.isnan(
            x
        )

        x[
            missing
        ] = medians[
            missing
        ]

        return x


    # ========================================================
    # MORDRED DESCRIPTORS
    # ========================================================

    def _mordred_features(self, mol):

        # Calculate one row while preserving Mordred column order
        result = self.mordred_calculator.pandas(
            [mol],
            quiet=True
        )

        # Convert Error / Missing objects -> NaN
        numeric = result.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Reorder against exact saved raw descriptor names
        raw_names = list(
            self.prep[
                "mordred_raw_descriptor_names"
            ]
        )

        current_names = [
            str(c)
            for c in numeric.columns
        ]

        if current_names != raw_names:

            # safer than silently accepting version/order mismatch
            name_to_col = {
                str(c): c
                for c in numeric.columns
            }

            missing_names = [
                name
                for name in raw_names
                if name not in name_to_col
            ]

            if missing_names:

                raise ValueError(
                    "Mordred descriptor version mismatch. "
                    f"Missing descriptors: {missing_names[:10]}"
                )

            numeric = numeric[
                [
                    name_to_col[name]
                    for name in raw_names
                ]
            ]


        x = (
            numeric
            .iloc[0]
            .to_numpy(
                dtype=np.float64
            )
        )

        x[
            np.isinf(x)
        ] = np.nan


        # --------------------------------------------
        # Step 1: training missingness filter
        # --------------------------------------------

        keep_missing = np.asarray(
            self.prep[
                "mordred_missingness_keep_mask"
            ],
            dtype=bool
        )

        x = x[
            keep_missing
        ]


        # --------------------------------------------
        # Step 2: valid median mask
        # --------------------------------------------

        valid_median = np.asarray(
            self.prep[
                "mordred_valid_median_mask"
            ],
            dtype=bool
        )

        x = x[
            valid_median
        ]


        # --------------------------------------------
        # Step 3: imputation
        # --------------------------------------------

        medians = np.asarray(
            self.prep[
                "mordred_train_medians_before_variance_filter"
            ],
            dtype=float
        )

        if len(x) != len(medians):

            raise ValueError(
                "Mordred imputation dimension mismatch."
            )

        missing = np.isnan(
            x
        )

        x[
            missing
        ] = medians[
            missing
        ]


        # --------------------------------------------
        # Step 4: variance mask
        # --------------------------------------------

        keep_var = np.asarray(
            self.prep[
                "mordred_variance_keep_mask"
            ],
            dtype=bool
        )

        x = x[
            keep_var
        ]


        return x


    # ========================================================
    # BUILD FINAL 931 FEATURES
    # ========================================================

    def _build_final_features(self, mol):

        rdkit_x = self._rdkit_features(
            mol
        )

        mordred_x = self._mordred_features(
            mol
        )

        combined = np.concatenate(
            [
                rdkit_x,
                mordred_x
            ]
        )


        final_keep = np.asarray(
            self.prep[
                "combined_correlation_keep_mask"
            ],
            dtype=bool
        )


        if len(
            combined
        ) != len(
            final_keep
        ):

            raise ValueError(
                "Combined descriptor dimension mismatch."
            )


        final_x = combined[
            final_keep
        ]


        if len(
            final_x
        ) != 931:

            raise ValueError(
                f"Expected 931 final descriptors, "
                f"got {len(final_x)}."
            )


        if np.isnan(
            final_x
        ).any():

            raise ValueError(
                "NaN remains in final descriptor vector."
            )


        if np.isinf(
            final_x
        ).any():

            raise ValueError(
                "Inf remains in final descriptor vector."
            )


        return final_x


    # ========================================================
    # DRUG PROPERTIES
    # ========================================================

    def _drug_properties(self, mol):

        mw = float(
            Descriptors.MolWt(
                mol
            )
        )

        logp = float(
            Crippen.MolLogP(
                mol
            )
        )

        tpsa = float(
            rdMolDescriptors.CalcTPSA(
                mol
            )
        )

        hbd = int(
            Lipinski.NumHDonors(
                mol
            )
        )

        hba = int(
            Lipinski.NumHAcceptors(
                mol
            )
        )

        rotatable = int(
            Lipinski.NumRotatableBonds(
                mol
            )
        )

        rings = int(
            rdMolDescriptors.CalcNumRings(
                mol
            )
        )

        heavy_atoms = int(
            mol.GetNumHeavyAtoms()
        )

        frac_csp3 = float(
            rdMolDescriptors.CalcFractionCSP3(
                mol
            )
        )

        qed = float(
            QED.qed(
                mol
            )
        )


        # --------------------------------------------
        # Lipinski Rule of Five
        # --------------------------------------------

        violations = []

        if mw > 500:
            violations.append(
                "MW > 500"
            )

        if logp > 5:
            violations.append(
                "cLogP > 5"
            )

        if hbd > 5:
            violations.append(
                "HBD > 5"
            )

        if hba > 10:
            violations.append(
                "HBA > 10"
            )


        lipinski_pass = (
            len(
                violations
            ) == 0
        )


        # --------------------------------------------
        # Simple drug-likeness filter
        #
        # Transparent heuristic for webtool.
        # NOT a clinical drug-success probability.
        # --------------------------------------------

        druglike_pass = (

            150 <= mw <= 600

            and -1 <= logp <= 6

            and tpsa <= 140

            and hbd <= 5

            and hba <= 10

            and rotatable <= 15
        )


        return {

            "MW":
                round(mw, 3),

            "cLogP":
                round(logp, 3),

            "TPSA":
                round(tpsa, 3),

            "HBD":
                hbd,

            "HBA":
                hba,

            "rotatable_bonds":
                rotatable,

            "ring_count":
                rings,

            "heavy_atom_count":
                heavy_atoms,

            "fraction_CSP3":
                round(
                    frac_csp3,
                    4
                ),

            "QED":
                round(
                    qed,
                    4
                ),

            "Lipinski_pass":
                lipinski_pass,

            "Lipinski_violations":
                violations,

            "druglikeness_filter":
                (
                    "PASS"
                    if druglike_pass
                    else "FAIL"
                )
        }


    # ========================================================
    # NEAREST TRAINING ANALOGUE
    # ========================================================

    def _nearest_training_analogue(
        self,
        mol
    ):

        query_fp = (
            self.morgan_generator
            .GetFingerprint(
                mol
            )
        )


        # Convert sparse reference rows to RDKit bit vectors
        # once per call; 607 molecules is small enough.
        best_similarity = -1.0
        best_index = -1


        for i in range(
            self.train_morgan.shape[0]
        ):

            row = self.train_morgan.getrow(
                i
            )

            ref_fp = DataStructs.ExplicitBitVect(
                2048
            )

            for bit in row.indices:

                ref_fp.SetBit(
                    int(bit)
                )


            sim = (
                DataStructs
                .TanimotoSimilarity(
                    query_fp,
                    ref_fp
                )
            )


            if sim > best_similarity:

                best_similarity = sim
                best_index = i


        ref = (
            self.train_ref
            .iloc[
                best_index
            ]
        )


        return {

            "nearest_training_similarity":
                round(
                    float(
                        best_similarity
                    ),
                    4
                ),

            "nearest_training_smiles":
                str(
                    ref[
                        "smiles"
                    ]
                ),

            "nearest_training_docking_score":
                round(
                    float(
                        ref[
                            "docking_score"
                        ]
                    ),
                    4
                ),

            "nearest_training_reference_id":
                (
                    str(
                        ref[
                            "reference_id"
                        ]
                    )
                    if "reference_id"
                    in ref.index
                    else None
                )
        }


    # ========================================================
    # SINGLE SMILES PREDICTION
    # ========================================================

    def predict(
        self,
        smiles
    ):

        mol, canonical = (
            self._parse_smiles(
                smiles
            )
        )


        # --------------------------------------------
        # Build exact 931-feature vector
        # --------------------------------------------

        final_x = (
            self._build_final_features(
                mol
            )
        )


        X = final_x.reshape(
            1,
            -1
        )


        # --------------------------------------------
        # PLS prediction
        # --------------------------------------------

        score = float(
            self.model
            .predict(
                X
            )
            .ravel()[0]
        )


        category = (
            docking_category(
                score
            )
        )


        # --------------------------------------------
        # Additional calculations
        # --------------------------------------------

        properties = (
            self._drug_properties(
                mol
            )
        )


        nearest = (
            self._nearest_training_analogue(
                mol
            )
        )


        # --------------------------------------------
        # Final result
        # --------------------------------------------

        result = {

            "input_smiles":
                str(smiles),

            "canonical_smiles":
                canonical,

            "valid_smiles":
                True,

            "predicted_docking_score":
                round(
                    score,
                    4
                ),

            "predicted_category":
                category,

            **properties,

            **nearest,

            "model_name":
                "OpenPLS_RDKit_Mordred_PLS6",

            "model_test_R2":
                0.5592,

            "model_test_RMSE":
                0.6328,

            "model_test_MAE":
                0.5055,

            "model_test_Spearman":
                0.6643,

            "prediction_note":
                (
                    "QSAR estimate from molecular structure. "
                    "This is not an experimentally measured "
                    "binding affinity or docking calculation."
                )
        }


        return result


    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    def predict_batch(
        self,
        smiles_list
    ):

        results = []


        for smi in smiles_list:

            try:

                result = self.predict(
                    smi
                )

            except Exception as e:

                result = {

                    "input_smiles":
                        str(smi),

                    "valid_smiles":
                        False,

                    "error":
                        str(e)
                }


            results.append(
                result
            )


        return pd.DataFrame(
            results
        )
