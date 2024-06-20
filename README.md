# WIP: molecular dynamics simulations

Code to run molecule dynamics simulations for various systems. 

## TODO

- [ ] Remove code duplication. 
- [ ] Split up simulation function 
- [ ] Make object oriented. 
- [ ] Write tests

## Installation

I'm using micromamba (alias =`mm`). You will not be able to install using Conda as it takes too long.

```
mm create -n simulations python openmm openff-toolkit openmmforcefields toml pandas seaborn matplotlib prolif mdanalysis -y
```

Once you've done that you can pip install this package: 

```
pip install (-e) . 
```

Use -e if you're going to make changes. 

## Usage
### 1. Simulation from docking results. 

This assumes you have done some docking from AutoDockVina and exported the docked pose to a sdf file. The system you give it (via a PDB) should be solvated and relaxed (via `pl_relaxation`) and contain any ligand you like e.g., the cognate ligand. Ideally this should be the same structure you do the docking into.  
The code will swap out the old ligand and put in the newly docked ligand and do minimization and then a simulation. 
```
dock_simulation --old_ligand_name ABC
                --complex_pdb complex.pdb
                --new_ligand_sdf results/ligand_001/docked_ligand.sdf
                --results_dir results/ligand_001 
```

You can then do fingerprint and rmsd analysis using. This will create some analysis charts in the results directory: 

```
ligand_analysis --results_dir results/ligand_001
```

Once you've done all the ligands in a series you can collate the results and compare the fingerprints across the whole series: 
```
series_analysis --results_dir results
```

### 2. Simulation of a ligand in solvent

You can simulate a ligand in a water box by: 
```
lig_simulation --smiles 'CCO' --results_dir ./results
```
A unique timestamped sub-directory will be created with in `--results_dir`.  See `setup_variables.png` for equilibration information. Trajectory data will be in `trajectory.dcd`. 

### 3. Relaxation and simulation of ligand-protein complex

You will need to prepare a system elsewhere i.e., you must have a:
- PDB/PDBx file containing all the full system (protein + ligand + water + ions)
- SDF file of the ligand. 

First relax the system: 
```
pl_relaxation --ligand_sdf ligand.sdf
              --complex_pdb complex.pdb 
              --results_dir series_x 
```
This will minimize, then heat to a target temperature with atomic restraints, then do NPT relaxation 
while removing restraints. 

Then you can do a simulation: 

```
pl_simulation --results_dir series_x
```
This will restart the simulation from where the relaxation left off and simulate for 1ns (or however long you need) in an NVT ensemble. 

Output charts are made which show the RMSD of the molecule over time, as well as the trajectory of the state variables (useful for checking convergence)


### 4. Vacuum minimization

Niche applicaton.  You can mimimize a protein ligand complex in vacuum. Useful for tidying up structures from e.g., AlphaFold2/ColabFold.
```
vac_minimization --ligand_sdf ligand.sdf
                 --complex_pdb complex.pdb 
```
      

## Contributing

1. Add a script in the `simulations` folder
2. Make an entry point in the `pyproject.toml` config file (under `[project.scripts]`)
3. Please follow [conventional commits](https://www.conventionalcommits.org/) when making merge requests and commits.
4. Run `ruff` and `black` before commiting.

 
## Maintainers

Primary maintainer: Lex O'brien - lex.e.obrien@gmail.com



