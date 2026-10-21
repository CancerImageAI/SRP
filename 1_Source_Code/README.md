# Source code

`SRP/` is a complete copy of the manuscript's source code repository,
also maintained publicly at:

**https://github.com/CancerImageAI/SRP**

This local copy is included so the submission package is self-contained
as a single zip file; the GitHub link is the canonical, versioned
source if you prefer to clone it (`git clone
https://github.com/CancerImageAI/SRP.git`) or want full commit history
(this copy omits the `.git/` directory to keep the package small).

See `SRP/README.md` for the pipeline overview and `SRP/docs/USAGE.md`
for the full step-by-step usage guide (feature extraction -> model
training -> evaluation -> survival analysis -> external validation ->
RNA-seq/single-cell biological validation).

For a small, fast, standalone demonstration that a reviewer can run
without the full pipeline's dependencies or the restricted-access
clinical data, see `../2_Demo/` instead.
