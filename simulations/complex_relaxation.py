import argparse
from pathlib import Path
import time
import toml

from openff.toolkit import Molecule, Topology
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmm.app import (
    ForceField,
    PME,
    HBonds,
    Simulation,
    StateDataReporter,
    DCDReporter,
    PDBFile,
)
from openmm import (
    XmlSerializer,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    CustomExternalForce,
)
from openmm.unit import (
    nanometer,
    picosecond,
    kelvin,
    bar,
    nanosecond,
    kilojoules_per_mole,
)

from .analysis import plot_rmsd, plot_state_equilibration


def simulate(config: argparse.Namespace, out_dir: Path) -> None:
    """Main simulation routine

    Parameters
    ----------
    config : argparse.Namespace
        arguments
    out_dir : Path
        output directory
    """
    system_path = out_dir / "system.xml"
    topology_path = out_dir / "complex.pdb"

    minimized_sim_path = out_dir / "minimized_simulation.xml"
    heated_sim_path = out_dir / "heated_simulation.xml"
    relaxed_sim_path = out_dir / "relaxed_simulation.xml"

    setup_traj_path = out_dir / "setup_trajectory.dcd"
    setup_data_path = out_dir / "setup_data.csv"

    if not Path(system_path).exists():
        print("Creating Topology")
        # Generate ligand conformers (to start simulation)
        ligand = Molecule(config.ligand_sdf)
        complex = Topology.from_pdb(config.complex_pdb)
        complex.add_molecule(ligand)
        # Create system
        # Generate forcefield - warning this will be slow.
        print("Creating forcefield")
        forcefield = ForceField(config.prt_forcefield, config.wat_forcefield)
        smirnoff = SMIRNOFFTemplateGenerator(
            molecules=ligand, forcefield=config.lig_forcefield
        )
        forcefield.registerTemplateGenerator(smirnoff.generator)

        # Solvate the ligand
        print("Creating system (could be slow for large liands with Sage FF)")
        system = forcefield.createSystem(
            complex.to_openmm(),
            nonbondedMethod=PME,
            nonbondedCutoff=config.nb_cutoff_nm * nanometer,
            constraints=HBonds,
        )
        print("Saving output")
        with system_path.open("w") as f:
            output = XmlSerializer.serialize(system)
            f.write(output)

        with topology_path.open("w") as f:
            PDBFile.writeFile(
                complex.to_openmm(), complex.get_positions().to_openmm(), f
            )

        pdb = complex.to_openmm()
    else:
        print("loading system")
        with system_path.open("r") as input:
            system = XmlSerializer.deserialize(input.read())

        pdb = PDBFile(str(topology_path))

    topology = pdb.getTopology()
    positions = pdb.getPositions()

    # Create simulation

    integrator = LangevinMiddleIntegrator(
        config.start_temperature_K * kelvin,
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

    ## Add restraints
    print("Adding restraints")
    restraint_forces = [float(x) for x in config.restraint_forces]
    restraint = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    system.addForce(restraint)
    restraint.addGlobalParameter(
        "k", restraint_forces[0] * kilojoules_per_mole / nanometer
    )
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    for atom in topology.atoms():
        if (atom.residue.name != "HOH") and (
            atom.element.name not in ("hydrogen", "chlorine", "sodium")
        ):
            restraint.addParticle(atom.index, positions[atom.index])

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

    print("Equilibrating pressure while removing restraints")
    for force in restraint_forces:
        print(f"\tRestraint force = {force:4.2f} kJ/mol/nm")
        simulation.context.setParameter("k", force * kilojoules_per_mole / nanometer)
        simulation.step(
            int(config.relax_time_ns * nanosecond / (config.timestep_ps * picosecond))
        )

    simulation.saveState(str(relaxed_sim_path))

    with (out_dir / "config.toml").open("w") as f:
        # toml_string = toml.dumps(vars(config))
        toml.dump(vars(config), f)


def report(out_dir: Path):
    """Creates reporting figures

    Parameters
    ----------
    out_dir : Path
        directory to put the figures in.
    """
    plot_state_equilibration(out_dir)
    plot_rmsd(out_dir, prefix=None)


def main(args: argparse.Namespace) -> None:
    """Runs simulation and analysis report

    Parameters
    ----------
    args :
        Args from argparse.
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
        help="complex (receptor +  water etc.)",
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
        "--start_temperature_K",
        type=float,
        default=100,
        help="starting temperature of simulation in K",
    )
    parser.add_argument(
        "--temperature_K",
        type=float,
        default=310,
        help="final temperature of simulation in K",
    )
    parser.add_argument(
        "--n_heating_steps", type=int, default=5, help="number of heating increments"
    )
    parser.add_argument(
        "--time_per_heat_ns",
        type=float,
        default=1,
        help="time spent at each temperature in ns",
    )
    parser.add_argument(
        "--restraint_forces",
        nargs="+",
        help="A list of restraining forces in kJ/mol/nm (should decrease to 0)",
        default=[100, 10, 5, 0],
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
        "--setup_record_resolution_ps",
        type=float,
        default=1,
        help="time resolution to record the setup state variables (T, PE, V, P)",
    )
    return parser.parse_args()


def run():
    args = get_args()
    main(args)
