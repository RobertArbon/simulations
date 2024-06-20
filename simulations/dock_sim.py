import argparse
from pathlib import Path
import time
import toml

from rdkit import Chem
from openff.toolkit import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmm.app import (
    ForceField,
    Modeller,
    PME,
    HBonds,
    Simulation,
    StateDataReporter,
    DCDReporter,
    PDBFile,
    PDBxFile,
)
from openmm import (
    XmlSerializer,
    LangevinMiddleIntegrator,
)
from openmm.unit import (
    nanometer,
    picosecond,
    kelvin,
    nanosecond,
    amu,
)

from .analysis import plot_rmsd, plot_state_equilibration


def simulate(config: argparse.Namespace, out_dir: Path):
    """Run simulation

    Parameters
    ----------
    config : argparse.Namespace
        CLI arguments
    out_dir : Path
        output directory
    """
    tic = time.time()
    topology_path = (
        out_dir / "complex.cif"
    )  # output for the new topology. Changing to CIF as it's better for large systems.
    system_path = out_dir / "system.xml"
    minimized_sim_path = out_dir / "minimized_simulation.xml"

    simualtion_traj_path = out_dir / "trajectory.dcd"
    simulation_data_path = out_dir / "data.csv"

    # create new topology delete old ligand, add new ligand.
    print("Creating new topology")
    pdb = PDBFile(str(config.complex_pdb))
    new_ligand = Molecule.from_rdkit(
        Chem.SDMolSupplier(config.new_ligand_sdf)[config.new_ligand_index]
    )
    model = Modeller(pdb.topology, pdb.positions)
    old_ligand = [
        res
        for res in pdb.topology.residues()
        if res.name.lower() == config.old_ligand_name.lower()
    ]
    assert (
        len(old_ligand) == 1
    ), f"More than 1 ligand found with resname {config.old_ligand_name.lower()}"
    model.delete(old_ligand)
    model.add(
        new_ligand.to_topology().to_openmm(), new_ligand.conformers[0].to_openmm()
    )
    topology = model.topology
    positions = model.positions
    # create new system and forcefield
    forcefield = ForceField(config.prt_forcefield, config.wat_forcefield)
    smirnoff = SMIRNOFFTemplateGenerator(
        molecules=new_ligand, forcefield=config.lig_forcefield
    )
    forcefield.registerTemplateGenerator(smirnoff.generator)

    with topology_path.open("w") as f:
        PDBxFile.writeFile(topology, positions, f)

    print("Creating forcefield (potentially very slow)")
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=PME,
        nonbondedCutoff=config.nb_cutoff_nm * nanometer,
        constraints=HBonds,
        rigidWater=bool(config.use_rigid_water),
    )

    print("Saving system output")
    with system_path.open("w") as f:
        output = XmlSerializer.serialize(system)
        f.write(output)

    # Create simulation
    delta_t_ps = config.timestep_ps * picosecond
    n_steps = int(config.traj_length_ns * nanosecond / delta_t_ps)
    integrator = LangevinMiddleIntegrator(
        config.temperature_K * kelvin, config.gamma_inv_ps / picosecond, delta_t_ps
    )
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    time_setup = (time.time() - tic) / 60
    print(f"Simulation creation time: {time_setup: 4.2f} mins")

    # Minimize and save
    print("Minimzing structure")
    simulation.minimizeEnergy()
    simulation.saveState(str(minimized_sim_path))

    # Add reporters
    simulation.reporters.append(
        StateDataReporter(
            file=str(simulation_data_path),
            reportInterval=int(
                config.traj_record_resolution_ps * picosecond / delta_t_ps
            ),
            time=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            totalSteps=n_steps,
            remainingTime=True,
        ),
    )
    simulation.reporters.append(
        DCDReporter(
            file=str(simualtion_traj_path),
            reportInterval=int(
                config.traj_record_resolution_ps * picosecond / delta_t_ps
            ),
        )
    )

    ## Add constraints
    if Path(config.constraint_file).exists():
        print("Adding constraints")
        with Path(config.constraint_file).open("r") as f:
            constraint_atoms = [int(x.strip()) for x in f.readlines()]
        print(f"\tConstraining {len(constraint_atoms)} atoms")

        for idx in constraint_atoms:
            system.setParticleMass(idx, 0 * amu)
    else:
        print("No constraint file found.")

    print(f"simulating for {n_steps} steps ({config.traj_length_ns} ns)")
    simulation.context.setVelocitiesToTemperature(config.temperature_K * kelvin)
    simulation.step(n_steps)

    with (out_dir / "config_simulation.toml").open("w") as f:
        toml.dump(vars(config), f)


def report(out_dir: Path) -> None:
    """Create report

    Parameters
    ----------
    out_dir : Path
        output directory
    """
    plot_state_equilibration(out_dir / "data.csv")
    plot_rmsd(out_dir, prefix=None)


def main(args: argparse.Namespace) -> None:
    """Main routine. Runs simulation and report

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments

    Raises
    ------
    Exception
        prints traceback if exception is raised.
    """
    out_dir = Path(args.results_dir)
    overwrite = bool(args.overwrite) or bool(args.report_only)
    if out_dir.exists() and (not overwrite):
        print(f"{out_dir} already exists")
        out_dir = Path(f"{args.results_dir}_{time.strftime('%Y_%m_%d_%H_%M_%S')}")
        print(f"creating new output directory {out_dir}")
        out_dir.mkdir(exist_ok=False, parents=False)
    else:
        out_dir.mkdir(exist_ok=True, parents=True)

    try:
        if not bool(args.report_only):
            simulate(args, out_dir)
        report(out_dir)

    except Exception as e:
        import traceback

        log = out_dir / "log.out"
        msg = f"{traceback.format_exc()}"
        with log.open("w") as f:
            f.write(msg)
        raise e


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--new_ligand_sdf", type=str, default="ligand.sdf", help="new ligand sdf file"
    )
    parser.add_argument(
        "--new_ligand_index",
        type=int,
        default=0,
        help="the index of the conformer in the sdf file.",
    )
    parser.add_argument(
        "--old_ligand_name",
        type=str,
        default="UNK",
        help="the residue name of the ligand to remove.",
    )
    parser.add_argument(
        "--complex_pdb",
        type=str,
        default="complex.pdb",
        help="complex (receptor + old ligand + water etc.)",
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results", help="Output directory."
    )
    parser.add_argument(
        "--report_only",
        type=bool,
        default=False,
        help="skip the simulation and run an analysis report only. overwrite set to True in this case.",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="overwrite the existing results_dir",
    )
    parser.add_argument(
        "--lig_forcefield",
        type=str,
        default="openff-2.1.0.offxml",
        help="Ligand forcefield",
    )
    parser.add_argument(
        "--wat_forcefield",
        type=str,
        default="amber/tip3p_standard.xml",
        help="Water forcefield",
    )
    parser.add_argument(
        "--prt_forcefield",
        type=str,
        default="amber/protein.ff14SB.xml",
        help="Protein forcefield",
    )
    parser.add_argument(
        "--nb_cutoff_nm", type=float, default=1.0, help="non-bonded cutoff in nm"
    )
    parser.add_argument(
        "--use_rigid_water",
        type=bool,
        default=True,
        help="whether to constrain water molecule angles",
    )
    parser.add_argument(
        "--constraint_file",
        type=str,
        default="constraints.txt",
        help="File containing new line separated atom indices to contrain",
    )
    parser.add_argument(
        "--temperature_K",
        type=float,
        default=310,
        help="final temperature of simulation in K",
    )
    parser.add_argument(
        "--gamma_inv_ps",
        type=float,
        default=1,
        help="friction coefficient for Langevin integrator in ps^-1",
    )
    parser.add_argument(
        "--timestep_ps", type=float, default=0.002, help="integration time step in ps"
    )
    parser.add_argument(
        "--traj_length_ns",
        type=float,
        default=1,
        help="simulation length in nanoseconds",
    )
    parser.add_argument(
        "--traj_record_resolution_ps",
        type=float,
        default=100,
        help="time resolution to record the trajectory)",
    )
    return parser.parse_args()


def run():
    args = get_args()
    main(args)
