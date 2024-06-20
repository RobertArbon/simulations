import argparse
from pathlib import Path
import time
import toml

from openff.toolkit import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmm.app import (
    ForceField,
    Modeller,
    PME,
    Simulation,
    StateDataReporter,
    DCDReporter,
    PDBFile,
)
from openmm import XmlSerializer, LangevinMiddleIntegrator, MonteCarloBarostat
from openmm.unit import (
    nanometer,
    picosecond,
    molar,
    kelvin,
    bar,
    nanosecond,
)

from .analysis import plot_state_equilibration, plot_rmsd


def simulate(config: argparse.Namespace, out_dir: Path):
    """Run simulation

    Parameters
    ----------
    config : argparse.Namespace
        CLI arguments
    out_dir : Path
        output directory
    """
    system_path = out_dir / "system.xml"
    minimized_sim_path = out_dir / "minimized_simulation.xml"
    heated_sim_path = out_dir / "heated_simulation.xml"
    relaxed_sim_path = out_dir / "relaxed_simulation.xml"

    setup_traj_path = out_dir / "setup_trajectory.dcd"
    setup_data_path = out_dir / "setup_data.csv"

    simualtion_traj_path = out_dir / "trajectory.dcd"
    simulation_top_path = out_dir / "topology.pdb"
    simulation_data_path = out_dir / "data.csv"

    print("Creating Ligand")
    # Generate ligand conformers (to start simulation)
    molecule = Molecule.from_smiles(config.smiles)
    molecule.generate_conformers()

    # Create system
    # Generate forcefield - warning this will be slow.
    print("Creating forcefield")
    forcefield = ForceField(config.wat_forcefield)
    smirnoff = SMIRNOFFTemplateGenerator(
        molecules=molecule, forcefield=config.lig_forcefield
    )
    forcefield.registerTemplateGenerator(smirnoff.generator)

    # Solvate the ligand
    print("Creating solvent box (could be slow for Sage FF)")
    model = Modeller(
        molecule.to_topology().to_openmm(), molecule.conformers[0].to_openmm()
    )
    model.addSolvent(
        forcefield,
        padding=config.padding_nm * nanometer,
        ionicStrength=config.ionic_strength_molar * molar,
    )
    topology = model.topology
    positions = model.positions

    system = forcefield.createSystem(
        topology, nonbondedMethod=PME, nonbondedCutoff=config.nb_cutoff_nm * nanometer
    )
    #  constraints=HBonds
    #  rigidWater=config.use_rigid_water)
    print("Saving output")
    with system_path.open("w") as f:
        output = XmlSerializer.serialize(system)
        f.write(output)

    with simulation_top_path.open("w") as f:
        PDBFile.writeFile(topology, positions, file=f)

    # Create simulation
    integrator = LangevinMiddleIntegrator(
        config.temperature_K * kelvin,
        config.gamma_inv_ps / picosecond,
        config.timestep_ps * picosecond,
    )
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    # Minize and save.
    print("Minimzing structure")
    simulation.minimizeEnergy()
    simulation.saveState(str(minimized_sim_path))

    delta_t_ps = config.timestep_ps * picosecond

    # Append reporters for set up
    simulation.reporters.append(
        StateDataReporter(
            file=str(setup_data_path),
            reportInterval=int(
                config.setup_record_resolution_ps * picosecond / delta_t_ps
            ),
            time=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            density=True,
        ),
    )
    simulation.reporters.append(
        DCDReporter(
            file=str(setup_traj_path),
            reportInterval=int(
                config.setup_record_resolution_ps * picosecond / delta_t_ps
            ),
        )
    )

    # Heat to target temperature
    Tmin = config.start_temperature_K
    Tmax = config.temperature_K
    print("Heating simulation")
    for i in range(config.n_heating_steps):
        T = Tmin + (i) * (Tmax - Tmin) / (config.n_heating_steps - 1)
        print(f"\tT = {T}K")
        simulation.integrator.setTemperature(T * kelvin)
        simulation.step(
            int(
                config.time_per_heat_ns * nanosecond / (config.timestep_ps * picosecond)
            )
        )

    simulation.saveState(str(heated_sim_path))
    # Relax system
    barostat = MonteCarloBarostat(
        config.pressure_bar * bar, config.temperature_K * kelvin, config.barostat_freq
    )

    system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)
    print("Equilibrating pressure")
    simulation.step(
        int(config.relax_time_ns * nanosecond / (config.timestep_ps * picosecond))
    )
    simulation.saveState(str(relaxed_sim_path))

    del simulation.reporters[:]

    # Append reporters for set up
    simulation.reporters.append(
        StateDataReporter(
            file=str(simulation_data_path),
            reportInterval=int(
                config.simulation_record_resolution_ps * picosecond / delta_t_ps
            ),
            time=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
        ),
    )
    simulation.reporters.append(
        DCDReporter(
            file=str(simualtion_traj_path),
            reportInterval=int(
                config.simulation_record_resolution_ps * picosecond / delta_t_ps
            ),
        )
    )

    # Run production
    print("Production simulation")
    # barostat.setFrequency(0)
    system.removeForce(index=system.getNumForces() - 1)
    simulation.context.reinitialize(preserveState=True)
    simulation.step(int(config.simulation_time_ns * nanosecond / delta_t_ps))


def report(out_dir: Path) -> None:
    """Create report of equilibration variables and rmsd.

    Parameters
    ----------
    out_dir : Path
        output directory
    """
    plot_state_equilibration(out_dir)
    plot_rmsd(out_dir, prefix=None)


def main(args: argparse.Namespace) -> None:
    """Main routine. runs simulation and report

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments
    """

    res_dir = Path(args.results_dir)
    ## This allows you to run analysis scripts without re-doing the simulation
    if not (res_dir / "trajectory.dcd").exists():
        do_sim = True
        out_dir = Path(args.results_dir) / f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"
        out_dir.mkdir(exist_ok=True, parents=True)
    else:
        do_sim = False
        out_dir = res_dir

    if do_sim:
        with (out_dir / "config.toml").open("w") as f:
            toml_string = toml.dumps(vars(args))
            print(f"Doing simulation with arguments:\n{toml_string}")
            toml.dump(vars(args), f)

        simulate(args, out_dir)

    report(out_dir)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles", type=str, default="CCO", help="SMILES string of molecule"
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
        "--padding_nm",
        type=float,
        default=1.5,
        help="the size of the water box from the edges of the molecule in nm.",
    )
    parser.add_argument(
        "--ionic_strength_molar",
        type=float,
        default=0.15,
        help="ionic strenght of the solvent in Molar units",
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
        "--start_temperature_K",
        type=float,
        default=100,
        help="starting temperature of simulation in K",
    )
    parser.add_argument(
        "--temperature_K",
        type=float,
        default=323,
        help="final temperature of simulation in K",
    )
    parser.add_argument(
        "--n_heating_steps", type=int, default=5, help="number of heating increments"
    )
    parser.add_argument(
        "--time_per_heat_ns",
        type=float,
        default=0.5,
        help="time spent at each temperature in ns",
    )
    parser.add_argument(
        "--gamma_inv_ps",
        type=float,
        default=0.5,
        help="friction coefficient for Langevin integrator in ps^-1",
    )
    parser.add_argument(
        "--timestep_ps", type=float, default=0.002, help="integration time step in ps"
    )
    parser.add_argument(
        "--relax_time_ns",
        type=float,
        default=1.0,
        help="time spent under constant pressure in ns",
    )
    parser.add_argument(
        "--pressure_bar",
        type=float,
        default=1.0,
        help="pressure of relaxation step in barr",
    )
    parser.add_argument(
        "--barostat_freq", type=int, default=25, help="frequency of barostat"
    )
    parser.add_argument(
        "--simulation_time_ns", type=float, default=1.0, help="Simulation time in ns"
    )
    parser.add_argument(
        "--setup_record_resolution_ps",
        type=float,
        default=1,
        help="time resolution to record the setup state variables (T, PE, V, P)",
    )
    parser.add_argument(
        "--simulation_record_resolution_ps",
        type=float,
        default=1,
        help="time resolution to record the simulation",
    )
    return parser.parse_args()


def run():
    args = get_args()
    main(args)
