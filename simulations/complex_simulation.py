import argparse
from pathlib import Path
import time
import toml

from openmm.app import (
    Simulation,
    StateDataReporter,
    DCDReporter,
    PDBFile,
)
from openmm import (
    XmlSerializer,
    LangevinMiddleIntegrator,
)
from openmm.unit import (
    picosecond,
    kelvin,
    nanosecond,
    amu,
)

from .analysis import plot_rmsd


def simulate(config: argparse.Namespace, out_dir: Path) -> None:
    """Main simualtion routine

    Parameters
    ----------
    config : argparse.Namespace
        CLI arguments
    out_dir : Path
        output directory

    Raises
    ------
    FileNotFoundError
        if system.xml isn't found
    FileNotFoundError
        if relaxed_system.xml isn't found
    """
    system_path = out_dir / "system.xml"
    topology_path = out_dir / "complex.pdb"

    relaxed_sim_path = out_dir / "relaxed_simulation.xml"

    simualtion_traj_path = out_dir / "trajectory.dcd"
    # simulation_top_path = out_dir / 'topology.pdb'
    simulation_data_path = out_dir / "data.csv"

    if not Path(system_path).exists():
        # TODO - allow new simulation to be created from new ligand
        raise FileNotFoundError(
            f"System definition not found. Make sure {system_path.name} and {topology_path.name} exist."
        )

    else:
        print("loading system")
        with system_path.open("r") as input:
            system = XmlSerializer.deserialize(input.read())

        pdb = PDBFile(str(topology_path))

    topology = pdb.getTopology()

    if not Path(relaxed_sim_path).exists():
        # TODO - allow new simulation to be created from new ligand
        raise FileNotFoundError(
            f"System state not found. Make sure {relaxed_sim_path.name} exists"
        )
    else:
        with relaxed_sim_path.open("r") as input:
            state = XmlSerializer.deserialize(input.read())

    delta_t_ps = config.timestep_ps * picosecond
    n_steps = int(config.traj_length_ns * nanosecond / delta_t_ps)

    # Create simulation
    integrator = LangevinMiddleIntegrator(
        config.temperature_K * kelvin, config.gamma_inv_ps / picosecond, delta_t_ps
    )
    simulation = Simulation(topology, system, integrator)

    simulation.context.setTime(state.getTime())
    simulation.context.setPositions(state.getPositions())
    simulation.context.setVelocities(state.getVelocities())
    simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
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
    simulation.step(n_steps)

    with (out_dir / "config_simulation.toml").open("w") as f:
        toml.dump(vars(config), f)


def report(out_dir: Path) -> None:
    """creates report in out_dir

    Parameters
    ----------
    out_dir : Path
        output directory
    """
    plot_rmsd(out_dir, prefix=None)


def main(args: argparse.Namespace) -> None:
    """Runs simulation and report

    Parameters
    ----------
    args : argparse.Namespace
        _description_
    """

    res_dir = Path(args.results_dir)
    ## This allows you to skip system setup
    if not (res_dir / "system.xml").exists():
        out_dir = Path(args.results_dir) / f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"
        out_dir.mkdir(exist_ok=True, parents=True)
    else:
        out_dir = res_dir

    simulate(args, out_dir)

    report(out_dir)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ligand_sdf", type=str, default="ligand.sdf", help="ligand sdf  file"
    )
    parser.add_argument(
        "--complex_pdb",
        type=str,
        default="complex.pdb",
        help="complex (receptor + ligand + water etc.)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Output dir. A timestamped subdir will be made if trajectory.dcd not found",
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
