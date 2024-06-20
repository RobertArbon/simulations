import argparse
from pathlib import Path
import time
import toml

from openff.toolkit import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmm.app import ForceField, Simulation, StateDataReporter, PDBFile
from openmm import XmlSerializer, VerletIntegrator, LocalEnergyMinimizer


def simulate(config: argparse.Namespace, out_dir: Path) -> None:
    """Runs simulation

    Parameters
    ----------
    config : argparse.Namespace
        CLI arguments
    out_dir : Path
        output directory
    """
    system_path = out_dir / "system.xml"
    minimized_sim_path = out_dir / "minimized_simulation.xml"
    reporter_path = out_dir / "mimization_local.csv"

    pdb = PDBFile(
        config.complex_pdb
    )  # needed for restraints as well as setting up the system.
    if not Path(system_path).exists():
        print("Creating Ligand")
        # Generate ligand conformers (to start simulation)
        molecule = Molecule(config.ligand_sdf)
        print(molecule)
        # Create system
        # Generate forcefield - warning this will be slow.
        print("Creating forcefield")
        forcefield = ForceField(
            config.prt_forcefield,
            "amber/tip3p_standard.xml",
        )
        smirnoff = SMIRNOFFTemplateGenerator(
            molecules=molecule, forcefield=config.lig_forcefield
        )
        forcefield.registerTemplateGenerator(smirnoff.generator)

        # Solvate the ligand
        print("Creating system  (could be slow for Sage FF)")
        system = forcefield.createSystem(pdb.topology)

        print("Saving output")
        with system_path.open("w") as f:
            output = XmlSerializer.serialize(system)
            f.write(output)
    else:
        print("loading system")
        with system_path.open("r") as input:
            system = XmlSerializer.deserialize(input.read())

    # Create simulation
    integrator = VerletIntegrator()
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    # Minize and save.

    print("Minimzing structure")
    LocalEnergyMinimizer.minimize(
        simulation.context,
        tolerance=config.tolerance_local,
        maxIterations=config.max_iterations_local,
        reporter=StateDataReporter(
            str(reporter_path),
            reportInterval=1,
            step=True,
            potentionEnergy=True,
        ),
    )
    # simulation.minimizeEnergy()
    simulation.saveState(str(minimized_sim_path))


def main(args: argparse.Namespace) -> None:
    """Main routine - runs simulation

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments
    """
    out_dir = Path(args.results_dir) / f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    out_dir.mkdir(exist_ok=True, parents=True)

    with (out_dir / "config.toml").open("w") as f:
        toml_string = toml.dumps(vars(args))
        print(f"Doing simulation with arguments:\n{toml_string}")
        toml.dump(vars(args), f)

    simulate(args, out_dir)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ligand_sdf", type=str, default="ligand.sdf", help="ligand sdf  molecule"
    )
    parser.add_argument(
        "--complex_pdb",
        type=str,
        default="complex.pdb",
        help="complex (protein + ligand + water etc.)",
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
        "--prt_forcefield",
        type=str,
        default="amber/protein.ff14SB.xml",
        help="Protein forcefield",
    )
    parser.add_argument(
        "--tolerance_local",
        type=float,
        default=10,
        help="tolerance for LocalEnergyMinimizer",
    )
    parser.add_argument(
        "--max_iterations_local",
        type=int,
        default=0,
        help="max iterations.  If > 0 then minimziation stops regardless of tolerance",
    )

    return parser.parse_args()


def run():
    args = get_args()
    main(args)
