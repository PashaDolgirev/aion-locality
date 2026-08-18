# Project: aion-locality

Physics-motivated neural networks ("EwaldNN") that learn the locality structure of energy functionals
`E[ρ]` from datasets of density profiles and their energies. The repo accompanies the paper
*"Learning locality and emergence in many-body physical systems"* — see `Notes/notes_1D.pdf` and `Notes/notes_3D.pdf`
for the full mathematical setup.

**Current status**: 1D (`ewaldnn1d/`) and 3D (`ewaldnn3d/`) are complete; the 2D (`ewaldnn2d/`) one is close to complete, and is the focus of this session.

Read carefully the notes in `Notes/notes_2D.pdf` - the first out of three experiments are done; our goal is to work out the remaining two with the same level of care.


## Working style preferences (from this repo's notebooks)
- Don't introduce new dependencies casually. Current stack: `torch`, `torch_dct`, `numpy`,
  `pandas`, `matplotlib`.
- The code style should be consistent throughtout the codebase. Conciseness is preferred. Clarity is critical. 
- Write test cases which might be inefficient to test implementation.
