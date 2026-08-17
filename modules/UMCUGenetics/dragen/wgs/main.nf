process DRAGEN_WGS {
    // No container image as dragen requires specific hardware (FPGA) to run.
    tag "${meta.id}"
    label 'process_fpga'

    input:
    // TODO: use prefix via task.ext.prefix instead of passing it as an input
    // TODO: what to do with repeat_genotype_specs? -> or should we create a 'srWGS' specific module that has this hardcoded?
    tuple val(meta), path(r1_fastq), path(r2_fastq)
    path fastq_list
    path ref_tar
    path repeat_genotype_specs
    val output_file_prefix

    output:
    // TODO: Expand output to include all expected outputs of the process? Or can we leave it tailored to the needs of the srWGS workflow?
    tuple val(meta), path("*"), emit: output
    tuple val(meta), path('*.csv'), emit: csv
    tuple val(meta), path("${meta.id}_${output_file_prefix}.wgs_coverage_metrics.csv"), emit: wgs_coverage_metrics
    tuple val(meta), path("${meta.id}_${output_file_prefix}.cnv_metrics.csv"), emit: cnv_metrics
    tuple val(meta), path("${meta.id}_${output_file_prefix}.mapping_metrics.csv"), emit: mapping_metrics
    tuple val(meta), path("${meta.id}_${output_file_prefix}.ploidy_estimation_metrics.csv"), emit: ploidy_estimation_metrics
    tuple val(meta), path("${meta.id}_${output_file_prefix}.gvcf_metrics.csv"), emit: gvcf_metrics
    tuple val("${task.process}"), val('dragen'), eval("dragen --version 2>&1 | sed 's/^dragen Version //'"), topic: versions, emit: versions_dragen

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    // def prefix = task.ext.prefix ?: "${meta.id}"
    """
    set -ex
    mkdir -p /scratch/reference
    tar -C /scratch/reference -xf ${ref_tar}

    /opt/edico/bin/dragen --partial-reconfig HMM --ignore-version-check true

    /opt/edico/bin/dragen --lic-instance-id-location /opt/instance-identity \\
        --ref-dir /scratch/reference/DRAGEN/9 \\
        --fastq-list ${fastq_list} \\
        --fastq-list-sample-id ${meta.id} \\
        --output-file-prefix ${meta.id}_${output_file_prefix} \\
        --output-directory ./ \\
        --intermediate-results-dir /scratch \\
        --enable-map-align true \\
        --enable-map-align-output true \\
        --output-format BAM \\
        --enable-duplicate-marking true \\
        --enable-variant-caller true \\
        --vc-emit-ref-confidence GVCF \\
        --vc-enable-vcf-output true \\
        --vc-ml-enable-recalibration true \\
        --enable-cnv true \\
        --cnv-segmentation-mode SLM \\
        --enable-sv true \\
        --repeat-genotype-enable true \\
        --repeat-genotype-specs ${repeat_genotype_specs} \\
        --enable-smn true \\
        --enable-cyp2d6 true \\
        --enable-cyp2b6 false \\
        --enable-gba true \\
        --enable-star-allele false \\
        --enable-cyp21a2 false \\
        --enable-hba false \\
        --enable-lpa false \\
        --checkfingerprint-enable-vcf-comparison false \\
        --cnv-enable-self-normalization true \\
        --logging-to-output-dir true \\
        --qc-cross-cont-vcf /opt/edico/config/sample_cross_contamination_resource_hg38.vcf.gz \\
        --enable-rh true \\
        --cnv-enable-ref-calls true \\
        --enable-bam-indexing true \\
        --enable-metrics-json true \\
        --enable-vcf-compression true \\
        --vc-enable-bqd true \\
        --enable-hla true \\
        --hla-enable-class-2 false \\
        --sv-cnv-max-coord-delta 1000 \\
        ${args}
    """

    stub:
    // TODO: Expand stub output to include all expected outputs of the process?
    def args = task.ext.args ?: ''
    // def prefix = task.ext.prefix ?: "${meta.id}"
    """
    echo ${args}

    touch ${meta.id}_${output_file_prefix}.bam
    touch ${meta.id}_${output_file_prefix}.wgs_coverage_metrics.csv
    touch ${meta.id}_${output_file_prefix}.cnv_metrics.csv
    touch ${meta.id}_${output_file_prefix}.mapping_metrics.csv
    touch ${meta.id}_${output_file_prefix}.ploidy_estimation_metrics.csv
    touch ${meta.id}_${output_file_prefix}.gvcf_metrics.csv
    """
}
