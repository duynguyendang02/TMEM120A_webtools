"""Streamlit prototype for molecular docking-score prediction.

Run locally with: streamlit run app.py
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, FilterCatalog, Lipinski, QED, rdMolDescriptors
from streamlit_ketcher import st_ketcher

from predictor_ood import get_model_directory, predict_docking_score

st.set_page_config(page_title="TMEM120A Candidate Predictor", layout="wide")


@st.cache_resource
def alert_catalogs():
    """Build PAINS and Brenk substructure-alert catalogues once per app session."""
    # Use RDKit's public module API; it works across the RDKit builds supplied
    # by Google Colab and conda/pip environments.
    pains_params = FilterCatalog.FilterCatalogParams()
    for catalog in (
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C,
    ):
        pains_params.AddCatalog(catalog)
    brenk_params = FilterCatalog.FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(pains_params), FilterCatalog.FilterCatalog(brenk_params)


def make_3d_block(mol: Chem.Mol) -> str:
    """Generate a 3D conformer in the browser-friendly SDF format."""
    mol_3d = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol_3d, params)
    if status != 0:
        raise ValueError("Could not generate a 3D conformer for this molecule.")
    AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)
    return Chem.MolToMolBlock(mol_3d)


def render_3d(mol_block: str, height: int = 680) -> None:
    """Render an interactive 3D molecular viewer using 3Dmol.js."""
    escaped = mol_block.replace("`", "\\`").replace("${", "\\${")
    html = f"""
    <div id="viewer" style="height:{height - 20}px;width:100%;position:relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      const viewer = $3Dmol.createViewer('viewer', {{backgroundColor: '#0f172a'}});
      viewer.addModel(`{escaped}`, 'sdf');
      viewer.setStyle({{}}, {{stick: {{radius: 0.16}}, sphere: {{scale: 0.28}}}});
      viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.18, color: 'white'}});
      viewer.zoomTo(); viewer.render(); viewer.zoom(1.15, 0);
    </script>
    """
    components.html(html, height=height)


def properties(mol: Chem.Mol) -> dict[str, float | int | str]:
    return {
        "Molecular formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight": Descriptors.MolWt(mol),
        "cLogP": Crippen.MolLogP(mol),
        "TPSA (Å²)": rdMolDescriptors.CalcTPSA(mol),
        "H-bond donors": Lipinski.NumHDonors(mol),
        "H-bond acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable bonds": Lipinski.NumRotatableBonds(mol),
        "Rings": Lipinski.RingCount(mol),
        "Heavy atoms": mol.GetNumHeavyAtoms(),
        "Molar refractivity": Crippen.MolMR(mol),
        "Fraction Csp³": rdMolDescriptors.CalcFractionCSP3(mol),
        "QED": QED.qed(mol),
    }


def filter_results(mol: Chem.Mol, p: dict[str, float | int | str]) -> list[dict[str, str]]:
    """Return common medicinal-chemistry screens and structural alerts."""
    def result(name: str, passed: bool, criteria: str, note: str = "") -> dict[str, str]:
        return {
            "Filter": name,
            "Result": "✓ PASS" if passed else "⚠ FLAG",
            "Criteria": criteria,
            "Details": note or ("Within guideline" if passed else "Outside guideline"),
        }

    lipinski_violations = sum((
        p["Molecular weight"] > 500,
        p["cLogP"] > 5,
        p["H-bond donors"] > 5,
        p["H-bond acceptors"] > 10,
    ))
    pains_catalog, brenk_catalog = alert_catalogs()
    pains = [match.GetDescription() for match in pains_catalog.GetMatches(mol)]
    brenk = [match.GetDescription() for match in brenk_catalog.GetMatches(mol)]

    return [
        result("Lipinski Ro5", lipinski_violations <= 1, "≤1 violation: MW ≤500; cLogP ≤5; HBD ≤5; HBA ≤10", f"{lipinski_violations} violation(s)"),
        result("Veber", p["Rotatable bonds"] <= 10 and p["TPSA (Å²)"] <= 140, "Rotatable bonds ≤10; TPSA ≤140 Å²"),
        result("Egan", p["cLogP"] <= 5.88 and p["TPSA (Å²)"] <= 131.6, "cLogP ≤5.88; TPSA ≤131.6 Å²"),
        result("Ghose", 160 <= p["Molecular weight"] <= 480 and -0.4 <= p["cLogP"] <= 5.6 and 40 <= p["Molar refractivity"] <= 130 and 20 <= p["Heavy atoms"] <= 70, "MW 160–480; cLogP −0.4–5.6; MR 40–130; atoms 20–70"),
        result("Muegge", 200 <= p["Molecular weight"] <= 600 and -2 <= p["cLogP"] <= 5 and p["TPSA (Å²)"] <= 150 and p["H-bond donors"] <= 5 and p["H-bond acceptors"] <= 10 and p["Rotatable bonds"] <= 15 and p["Rings"] <= 7, "MW 200–600; cLogP −2–5; TPSA ≤150; HBD ≤5; HBA ≤10; RB ≤15; rings ≤7"),
        result("PAINS alerts", not pains, "No PAINS substructure alerts", "; ".join(pains) if pains else "No PAINS alert"),
        result("Brenk alerts", not brenk, "No Brenk structural alerts", "; ".join(brenk) if brenk else "No Brenk alert"),
    ]


def property_table(p: dict[str, float | int | str]) -> pd.DataFrame:
    """Convert scalar property dictionary into rows for st.dataframe."""
    rows = []
    for name, value in p.items():
        value_text = f"{value:.2f}" if isinstance(value, float) else str(value)
        rows.append({"Property": name, "Value": value_text})
    return pd.DataFrame(rows)


def export_table(
    smiles: str,
    target_name: str,
    prediction,
    p: dict[str, float | int | str],
    filters: list[dict[str, str]],
) -> pd.DataFrame:
    """Create one flat, spreadsheet-friendly row with all reported results."""
    row: dict[str, str | float | int] = {
        "Input SMILES": smiles,
        "Protein target": target_name,
        "Predicted docking score (kcal/mol)": round(prediction.score, 3) if prediction.score is not None else "",
        "Predicted category": prediction.category if prediction.in_domain else "",
        "Model applicability": "IN_DOMAIN" if prediction.in_domain else "OUT_OF_DOMAIN",
        "Descriptor max |z|": round(prediction.max_abs_z, 3),
        "OOD guard threshold": round(prediction.ood_threshold, 3),
        "Nearest training Tanimoto": prediction.nearest_training_similarity,
        "Nearest training SMILES": prediction.nearest_training_smiles,
        "Nearest training docking score (kcal/mol)": prediction.nearest_training_docking_score,
        "Model": prediction.model_name,
    }
    row.update({f"Property: {name}": value for name, value in p.items()})
    for item in filters:
        row[f"Filter: {item['Filter']}"] = item["Result"]
        row[f"Details: {item['Filter']}"] = item["Details"]
    return pd.DataFrame([row])


def use_drawn_smiles(drawn_smiles: str) -> None:
    """Copy a valid structure-editor result into the main SMILES input."""
    st.session_state["smiles_input"] = drawn_smiles


st.title("TMEM120A Candidate Predictor")
st.caption("An AI-powered molecular screening tool for TMEM120A")

target_name = "TMEM120A"

EXAMPLES = {
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Lidocaine": "CCN(CC)CC(=O)Nc1c(C)cccc1C",
    "Propofol": "CC(C)c1cccc(C(C)C)c1O",
}

with st.sidebar:
    st.markdown("**Example molecule**")
    example_name = st.selectbox("Choose an example", list(EXAMPLES), label_visibility="collapsed")
    if st.button("Load example", use_container_width=True):
        st.session_state["smiles_input"] = EXAMPLES[example_name]
        st.rerun()
    st.code(EXAMPLES[example_name], language=None)

smiles_tab, draw_tab = st.tabs(["Enter SMILES", "Draw molecule"])

with smiles_tab:
    smiles = st.text_area(
        "Input SMILES",
        value="CC(=O)Nc1ccc(O)cc1",
        height=100,
        placeholder="e.g. CC(=O)Nc1ccc(O)cc1",
        key="smiles_input",
    )

with draw_tab:
    st.caption("Draw or edit a molecule below, then transfer its generated SMILES into the predictor.")
    drawn_smiles = st_ketcher(st.session_state.get("smiles_input", "CC(=O)Nc1ccc(O)cc1"))
    if drawn_smiles:
        if Chem.MolFromSmiles(drawn_smiles) is None:
            st.warning("The drawn structure does not yet produce a valid SMILES. Please complete the structure.")
        else:
            st.code(drawn_smiles, language=None)
            st.button(
                "Use drawn structure for prediction",
                type="primary",
                on_click=use_drawn_smiles,
                args=(drawn_smiles,),
            )
    else:
        st.info("Finish drawing a molecule to generate its SMILES.")

if st.button("Analyze molecule", type="primary", use_container_width=False):
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        st.error("Invalid SMILES. Please check the molecular notation and try again.")
        st.stop()

    try:
        mol_block = make_3d_block(mol)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    props = properties(mol)
    try:
        prediction = predict_docking_score(mol, target_name)
    except Exception as exc:
        st.error("The PLS6 model backend could not run.")
        st.exception(exc)
        st.stop()

    filters = filter_results(mol, props)
    lipinski = next(item for item in filters if item["Filter"] == "Lipinski Ro5")

    st.success("Molecular analysis complete")
    st.subheader("TMEM120A prediction")
    score_col, category_col = st.columns(2)
    if prediction.in_domain:
        score_col.metric("Predicted docking score", f"{prediction.score:.2f} kcal/mol", help="QSAR-estimated docking score. More negative values correspond to the stronger-score categories used to train this project.")
        category_col.metric("Predicted category", prediction.category)
    else:
        score_col.metric("Predicted docking score", "Not reported", help="The query is far outside the descriptor space represented by the training set, so the raw PLS extrapolation is suppressed.")
        category_col.metric("Model applicability", "OUT OF DOMAIN")
        st.warning(
            "Outside model applicability range. This molecule lies substantially outside the descriptor space represented by the 607-compound training set, so a docking-score prediction is not reported."
        )

    st.markdown("**Drug-likeness summary**")
    qed_col, filter_col = st.columns(2)
    qed_col.metric("QED", f"{props['QED']:.2f}", help="Quantitative Estimate of Drug-likeness, scaled from 0 to 1.")
    filter_col.metric("Lipinski rule-of-five", lipinski["Result"].upper(), help=lipinski["Details"])

    viewer_col, profile_col = st.columns([1.35, 1], gap="large")
    with viewer_col:
        st.subheader("3D structure")
        render_3d(mol_block, height=520)

    with profile_col:
        st.subheader("Molecular profile")
        mol_2d = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol_2d)
        st.image(Draw.MolToImage(mol_2d, size=(600, 330)), use_container_width=True)
        st.markdown("**Physicochemical properties**")
        st.dataframe(property_table(props), use_container_width=True, hide_index=True, height=390)

    st.subheader("Drug-likeness and structural-alert filters")
    filter_df = pd.DataFrame(filters)
    st.dataframe(filter_df, use_container_width=True, hide_index=True, column_config={
        "Result": st.column_config.TextColumn("Result", width="small"),
        "Criteria": st.column_config.TextColumn("Criteria", width="large"),
        "Details": st.column_config.TextColumn("Details", width="medium"),
    })
    st.caption("These are medicinal-chemistry screening heuristics. A flag does not prove a compound is unsuitable; it indicates an aspect worth reviewing.")

    st.subheader("Structural context")
    analogue_mol = Chem.MolFromSmiles(prediction.nearest_training_smiles)
    analogue_img_col, analogue_info_col = st.columns([1, 1.35], gap="large")
    with analogue_img_col:
        if analogue_mol is not None:
            analogue_2d = Chem.Mol(analogue_mol)
            AllChem.Compute2DCoords(analogue_2d)
            st.image(Draw.MolToImage(analogue_2d, size=(520, 300)), use_container_width=True)
    with analogue_info_col:
        st.markdown("**Nearest training analogue**")
        st.metric("Tanimoto similarity", f"{prediction.nearest_training_similarity:.3f}")
        st.metric("Training docking score", f"{prediction.nearest_training_docking_score:.3f} kcal/mol")
        st.code(prediction.nearest_training_smiles, language=None)
        st.caption("Similarity is provided as structural context and is not a calibrated prediction-confidence score.")

    export_df = export_table(smiles.strip(), target_name, prediction, props, filters)
    st.download_button(
        "Download results as CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="docking_prediction_results.csv",
        mime="text/csv",
    )

    with st.expander("About this prediction model"):
        st.markdown("**Canonical SMILES**")
        st.code(prediction.canonical_smiles, language=None)
        st.markdown(
            "**Model architecture:** RDKit + Mordred 2D descriptors → 931 retained features → PLSRegression (6 latent components)."
        )
        st.markdown(f"**Technical applicability guard:** descriptor-space max |z| = {prediction.max_abs_z:.2f}; threshold = {prediction.ood_threshold:.2f}.")
        st.markdown(
            f"**Independent test:** R² {prediction.model_test_r2:.4f} · RMSE {prediction.model_test_rmse:.4f} kcal/mol · "
            f"MAE {prediction.model_test_mae:.4f} kcal/mol · Spearman {prediction.model_test_spearman:.4f}."
        )

st.caption("Predictions are intended for virtual screening and candidate prioritization and should be followed by molecular docking and experimental validation.")

st.divider()
st.markdown(
    "<div style='text-align:center; color:#8b95a7; padding:0.4rem 0 1rem;'>"
    "© Made by Duy · CBMC Lab · Pharmacy Department · Seoul National University"
    "</div>",
    unsafe_allow_html=True,
)
