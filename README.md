# Modules


## Repository Structure

``` sh
.
├── README.md
├── main.nf
└── modules
    └── UMCUGenetics
└── subworkflows
    └── UMCUGenetics
```

Modules are placed under the `./modules/UMCUGenetics/` folder in the format tool/subtool. For example `./modules/UMCUGenetics/samtools/view/`
Similarly, subworkflows are placed under the `subworkflows/UMCUGenetics/` folder. Preferably a subworkflow is named `INPUT_DESCRIPTION` For example `./subworkflows/UMCUGenetics/bam_prs/` or `./subworkflows/UMCUGenetics/vcf_ancestry/`


## Linting
To lint the modules and subworkflows nf-core/tools is used. We use the default linting configuration, except for the `meta.yml` requirements. The github action for linting (triggered upon creating a PR to main) is already configured to ignore `meta.yml` checks. To run linting locally with the same settings:

``` sh
pip install nf-core==4.1.0
nf-core modules lint --key module_tests --key environment_yml --key main_nf --key module_deprecations --key module_tests <tool/name>
nf-core subworkflows lint --key subworkflow_tests --key subworkflow_if_empty_null --key subworkflow_todos --key main_nf <name>
```

## Testing
Module and subworkflow tests are evaluated using nf-test. A github action is configured to automatically run tests upon module/subworkflow change.

### Test data
Test data can be hosted on the companion repository https://github.com/UMCUGenetics/DxNextflowTestData. Be aware that this repository is public and should not contain any private data.


## Using a module or subworkflow in a nextflow pipeline
Components from this repository can be added to a pipeline similarly to how nf-core components are installed.

``` sh
nf-core modules --git-remote https://github.com/UMCUGenetics/NF-Modules install pgscatalog/combine
nf-core subworkflows --git-remote https://github.com/UMCUGenetics/NF-Modules install bam_prs
```

To view all available components:

``` sh
nf-core modules --git-remote https://github.com/UMCUGenetics/NF-Modules list remote
nf-core subworkflows --git-remote https://github.com/UMCUGenetics/NF-Modules list remote
```
