process DRAGEN {
    // No container image as dragen requires specific hardware (FPGA) to run.
    tag "${meta.id}"
    label 'process_fpga'

    // Dragen 4.2.4-2 (F2 compatible container)
    container '079623148045.dkr.ecr.eu-central-1.amazonaws.com/cp-prod/f1b7ad6a-11ac-4bc1-b705-b275ff2887ad:latest'

    input:
    // TODO: what to do with repeat_genotype_specs? -> or should we create a 'srWGS' specific module that has this hardcoded?
    tuple val(meta), path(r1_fastq), path(r2_fastq)
    path fastq_list
    path ref_tar
    path repeat_genotype_specs

    output:
    // TODO: Expand output to include all expected outputs of the process? Or can we leave it tailored to the needs of the srWGS workflow?
    tuple val(meta), path("*"), emit: output
    tuple val(meta), path('*.csv'), emit: csv
    tuple val(meta), path("${prefix}.wgs_coverage_metrics.csv"), emit: wgs_coverage_metrics
    tuple val(meta), path("${prefix}.cnv_metrics.csv"), emit: cnv_metrics
    tuple val(meta), path("${prefix}.mapping_metrics.csv"), emit: mapping_metrics
    tuple val(meta), path("${prefix}.ploidy_estimation_metrics.csv"), emit: ploidy_estimation_metrics
    tuple val(meta), path("${prefix}.gvcf_metrics.csv"), emit: gvcf_metrics
    tuple val("${task.process}"), val('dragen'), eval("dragen --version 2>&1 | sed 's/^dragen Version //'"), topic: versions, emit: versions_dragen

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ''
    """
    set -ex
    mkdir -p /scratch/reference
    tar -C /scratch/reference -xf ${ref_tar}

    /opt/edico/bin/dragen --partial-reconfig HMM --ignore-version-check true

    /opt/edico/bin/dragen --lic-instance-id-location /opt/instance-identity \\
        --ref-dir /scratch/reference/DRAGEN/9 \\
        --fastq-list ${fastq_list} \\
        --fastq-list-sample-id ${meta.id} \\
        --output-file-prefix ${prefix} \\
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
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ''
    """
    echo ${args}

    touch ${prefix}.bam
    touch ${prefix}.wgs_coverage_metrics.csv
    touch ${prefix}.cnv_metrics.csv
    touch ${prefix}.mapping_metrics.csv
    touch ${prefix}.ploidy_estimation_metrics.csv
    touch ${prefix}.gvcf_metrics.csv
    """
}
