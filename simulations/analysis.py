"""
Runs analysis on stored trajectories.
"""

import argparse
import toml
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mdtraj as md
import numpy as np
import MDAnalysis as mda
import prolif as plf


def plot_state_equilibration(out_dir: Path) -> None:
    """plot equilibration variables

    Parameters
    ----------
    out_dir : Path
        output directory
    """
    equil_df = pd.read_csv(out_dir / "setup_data.csv")
    equil_df_m = equil_df.melt(id_vars='#"Time (ps)"')
    equil_df_m = equil_df_m.groupby("variable", as_index=False).rolling(50).mean()
    with sns.plotting_context("talk"):
        g = sns.relplot(
            equil_df_m,
            x='#"Time (ps)"',
            y="value",
            col="variable",
            kind="line",
            facet_kws={"sharey": False, "sharex": True},
        )

        plt.savefig(out_dir / "setup_variables.png")
        plt.close()


def plot_rmsd(out_dir, prefix=None):
    """
    plot the rmsd of the ligand and the protein
    Parameters
    ----------
    out_dir : Path
        path to output directory
    prefix : prefix, optional
        Optional prefix to denote a simulation stage, e.g., relaxation / production etc., by default None
    """
    print("Calculating RMSD")

    if prefix is None:
        traj_path = "trajectory.dcd"
        out_path = "rmsd.png"
    else:
        traj_path = f"{prefix}_trajectory.dcd"
        out_path = f"{prefix}_rmsd.png"

    traj_path = out_dir / traj_path
    top_path = out_dir / "complex.cif"
    with (out_dir / "config_simulation.toml").open("r") as f:
        config = toml.load(f)
    dt = float(config["traj_record_resolution_ps"]) / 1000

    out_path = out_dir / out_path
    traj = md.load(str(traj_path), top=str(top_path))

    # reimage molcules
    mols = traj.top.find_molecules()
    mol_size = [len(x) for x in mols]
    biggest_mol = mols[np.argmax(mol_size)]
    traj.center_coordinates()
    traj.image_molecules(inplace=True, anchor_molecules=[biggest_mol])
    traj.make_molecules_whole(inplace=True)
    traj.superpose(traj)

    # Save solute only
    ligand_ix = traj.top.select("(resn UNK)")
    lig_traj = traj.atom_slice(ligand_ix)

    traj_rmsd = md.rmsd(traj, traj, atom_indices=traj.top.select("name CA")) * 10
    lig_rmsd = (
        md.rmsd(lig_traj, lig_traj, atom_indices=lig_traj.top.select("mass > 2")) * 10
    )
    with sns.plotting_context("talk"):
        fig, axes = plt.subplots(2)
        rmsds = [traj_rmsd, lig_rmsd]
        trajs = [traj, lig_traj]
        labels = ["Protein Ca", "Ligand heavy atoms"]
        for i, (x, y) in enumerate(zip(trajs, rmsds)):
            axes[i].plot(x.time * dt, y)
            axes[i].set_xlabel("Time (ns)")
            axes[i].set_ylabel("RMSD (Ang)")
            axes[i].set_title(labels[i])

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


def load_solute(res_dir: Path) -> mda.Universe:
    """Loads trajectory.dcd with topology complex.cif and saves solute.dcd and solute.pdb.
    Returns solute.pdb and solute.dcd as an MDAnalysis universe object

    Parameters
    ----------
    res_dir : Path
        Path of results created from simualation functions

    Returns
    -------
    mda.Universe
        Just the solute (protein, ligand) of the simulation
    """
    traj = md.load(str(res_dir / "trajectory.dcd"), top=str(res_dir / "complex.cif"))
    print("loaded traj with ", traj.n_frames, " frames")
    mols = traj.top.find_molecules()
    mol_size = [len(x) for x in mols]
    biggest_mol = mols[np.argmax(mol_size)]
    traj.center_coordinates()
    traj.image_molecules(inplace=True, anchor_molecules=[biggest_mol])
    traj.make_molecules_whole(inplace=True)
    traj.superpose(traj)

    traj = traj.atom_slice(traj.top.select("protein or resn UNK"))
    traj.save_dcd(str(res_dir / "solute.dcd"))
    traj[0].save_pdb(str(res_dir / "solute.pdb"))
    u = mda.Universe(
        str(res_dir / "solute.pdb"),
        str(res_dir / "solute.dcd"),
        vdwradii=dict(Cl=1.75),
        guess_bonds=True,
    )
    return u


def extract_protein_ligand_chains(
    complex: mda.Universe,
) -> Tuple[List[mda.AtomGroup], mda.AtomGroup]:
    """searparates out the protein chains and the ligand chain

    Parameters
    ----------
    complex : mda.Universe
        the protein/ligand trajectory

    Returns
    -------
    Tuple[List[mda.AtomGroup], mda.AtomGroup]
        the list of protein chains and then ligand chain
    """
    chain_ids = []
    for seg in complex.segments:
        if len(seg.residues) > 1:
            chain_ids.append(seg.segid)
            print(
                f"Identified chain {seg.segid} with {len(seg.residues)} residues as a protein chain"
            )
        elif (len(seg.residues) == 1) and (seg.residues[0].resname == "UNK"):
            ligand_id = seg.segid
        else:
            print(f"This segment has a weird number of residues: {len(seg.residues)}")
    protein_chains = [complex.select_atoms(f"segid {x}") for x in chain_ids]
    ligand_chain = complex.select_atoms(f"segid {ligand_id}")
    return protein_chains, ligand_chain


def chain_ligand_fingerprint(
    complex: mda.Universe, chain: mda.AtomGroup, ligand: mda.AtomGroup
) -> plf.Fingerprint:
    """Creates a interaction fingeprint for a given chain / ligand combination

    Parameters
    ----------
    complex : mda.Universe
        the protein ligand trajectory object
    chain : mda.AtomGroup
        the atoms comprising the chain of interest
    ligand : mda.AtomGroup
        the atoms of the ligand

    Returns
    -------
    plf.Fingerprint
        Interaction fingerprint
    """
    fp = plf.Fingerprint(parameters=None, count=False)
    fp.run(
        complex.trajectory,
        ligand,
        chain,
    )

    return fp


def fp_to_df(fp: plf.Fingerprint) -> pd.DataFrame:
    """Creates a dataframe of the interaction fingerprint and tidies the column names

    Parameters
    ----------
    fp : plf.Fingerprint
        interaction fingerprint of a chain/ligand combination

    Returns
    -------
    pd.DataFrame
        a dataframe version of the input
    """
    df = fp.to_dataframe()
    df = df.reset_index().melt(id_vars=[("Frame", "", "")])
    df.rename(columns={("Frame", "", ""): "frame"}, inplace=True)
    return df


def fp_figure(fp: plf.Fingerprint) -> plt.Axes:
    """Creates barcode figure from fingerprint

    Parameters
    ----------
    fp : plf.Fingerprint
        Interaction fingerprint

    Returns
    -------
    plt.Axes
        Matplotlib axes object for plotting
    """
    return fp.plot_barcode()


def fingerprint_analysis(config: Dict[str, Any]) -> pd.DataFrame:
    """Runs a fingerprint analysis on a single simulation result

    Parameters
    ----------
    config : Dict[str, Any]
        The configuration object used to run the simulation

    Returns
    -------
    pd.DataFrame
        Contains all the fingerprints for all the chains.
    """
    res_dir = Path(config["results_dir"])
    solute = load_solute(res_dir)
    protein_chains, ligand = extract_protein_ligand_chains(solute)
    protein_fp = []
    for chain in protein_chains:
        chain_fp = chain_ligand_fingerprint(solute, chain, ligand)
        if chain_fp.to_dataframe().shape[1] > 0:
            ax = fp_figure(chain_fp)
            plt.savefig(
                str(res_dir / f"fingerprint_{chain[0].segid}.png")
            )  # chain is an atom group.  segid sits on each atom.
            chain_fp.to_pickle(str(res_dir / f"fingerprint_{chain[0].segid}.pkl"))
    protein_fp.append(chain_fp)
    protein_fp = pd.concat(protein_fp)
    return protein_fp


def com_chains(config: Dict[str, Any]) -> None:
    """Create and save a plot of the difference between the center of mass of all the chains.
        Saves the plot to 'com_distances.png' in the results directory specified in the config.

    Parameters
    ----------
    config : Dict[str, Any]
        The config object used to run the simulation.
    """
    res_dir = Path(config["results_dir"])
    dt = float(config["traj_record_resolution_ps"]) / 1000
    # u = mda.Universe(str(res_dir/'solute.dcd'), str(res_dir/'solute.pdb'))
    traj = md.load(str(res_dir / "solute.dcd"), top=str(res_dir / "solute.pdb"))
    chains = [
        traj.atom_slice(traj.top.select(f"chainid {i} and mass > 2"))
        for i in range(traj.top.n_chains)
    ]
    chain_lengths = [chain.top.n_atoms for chain in chains]
    coms = [md.compute_center_of_mass(chain) for chain in chains]
    dfs = []
    n_chains = len(chains)
    for i in range(n_chains - 1):
        for j in range(i + 1, n_chains):
            label = f"d({i},{j}) ({chain_lengths[i]} atoms, {chain_lengths[j]} atoms)"
            distance = (np.sqrt(((coms[i] - coms[j]) ** 2).sum(axis=1))) * 10
            time = np.arange(distance.shape[0]) * dt
            df = pd.DataFrame(data={"distance_Ang": distance, "time_ns": time})
            df["label"] = label
            dfs.append(df)

    df = pd.concat(dfs)
    with sns.plotting_context("paper"):
        sns.relplot(data=df, x="time_ns", y="distance_Ang", row="label", kind="line")
        plt.savefig(res_dir / "com_distances.png")


def ligand_analysis():
    """Entry point for analysis of a single protein-ligand simulation.
    Creates a fingerprint and a plot showing the distances between the chains.
    """
    args = get_args_single_ligand()
    with (Path(args.results_dir) / "config_simulation.toml").open("r") as f:
        config = toml.load(f)

    fingerprint_analysis(config)
    com_chains(config)


def collate_fps(files: List[Path]) -> pd.DataFrame:
    """Collates all the fingperints from different ligands and puts them in the same dataframe

    Parameters
    ----------
    files : List[Path]
        List of all the paths to the fingerprints

    Returns
    -------
    pd.DataFrame
        Dataframe containing all the ligands. Labelled according to the directory name.
    """

    series_fp = []
    for file in files:
        print(f"Collating FP from {file}")
        label = file.parents[0].name
        fp = plf.Fingerprint.from_pickle(str(file))
        df = fp_to_df(fp)
        df["ligand_id"] = label
        series_fp.append(df)
    series_fp = pd.concat(series_fp)
    series_fp.reset_index(inplace=True, drop=True)
    return series_fp


def summarize_fp(series_fp: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe containing the fingerprints from several ligands and calculates the proprotion
    of frames a interaction is observed for each ligand.

    Parameters
    ----------
    series_fp : pd.DataFrame
        output from `collate_fps`

    Returns
    -------
    pd.DataFrame
        each row is a different ligand, each column a interaction and the values are the proportion of
        frames in which the observation is observed.
    """
    summary = series_fp.groupby(
        ["ligand_id", "ligand", "protein", "interaction"], as_index=False
    ).mean()
    summary.drop(labels=["frame", "ligand"], axis=1, inplace=True)
    return summary


def plot_summary_fp(summary_fp: pd.DataFrame, out_dir: Path) -> None:
    """create a plot of the fingeprint across the whole series of ligands

    Parameters
    ----------
    summary_fp : pd.DataFrame
        the output of `summary_fp`
    out_dir : Path
        the location to put the image.
    """
    interactions = summary_fp["interaction"].unique()

    summary_wide = summary_fp.pivot(
        index="ligand_id", columns=["protein", "interaction"]
    )
    summary_wide = summary_wide.droplevel(level=0, axis=1).swaplevel(axis=1)
    summary_wide = summary_wide.fillna(0)
    summary_wide = summary_wide.sort_values(by=["ligand_id"])

    sns.set_style({"font.family": "monospace"})

    for interaction in interactions:
        with sns.plotting_context("paper"):
            df_toplot = summary_wide.loc[:, (interaction)]
            df_toplot = df_toplot[
                sorted(df_toplot.columns, key=lambda x: f"{x[-1]}{x[3:]}")
            ]
            ax = sns.heatmap(df_toplot.T, vmin=0, vmax=1, cmap="rocket_r")
            ax.tick_params(labelbottom=False, bottom=False, labeltop=True, top=True)
            _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.savefig(
                out_dir / f"{interaction}_fingerprints.png", bbox_inches="tight"
            )
        plt.close()


def series_analysis():
    """Entry point for running analysis on whole series of ligands."""
    args = get_args_multiple_ligands()
    out_dir = Path(args.results_dir) / "summary" / "fingerprints"
    out_dir.mkdir(exist_ok=True, parents=True)
    # collate fingerprints
    files = Path(args.results_dir).rglob("fingerprint*.pkl")
    series_fp = collate_fps(files)
    summary_fp = summarize_fp(series_fp)
    plot_summary_fp(summary_fp, out_dir)
    series_fp.to_json(out_dir / "series_fp.json")


def get_args_single_ligand():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="output directory containing docking results e.g., ./ligand_001",
    )
    return parser.parse_args()


def get_args_multiple_ligands():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help='output directory containing "ligand_001,002..." subfolders',
    )
    return parser.parse_args()
