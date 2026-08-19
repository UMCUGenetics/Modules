process DRAGENUTILS_DXQC {
    tag "${meta.id}"
    label 'process_single'

    container 'ghcr.io/umcugenetics/dragen-utils:sha-b1d12b9'

    input:
    tuple val(meta), path(wgs_coverage_metrics), path(cnv_metrics), path(mapping_metrics), path(ploidy_estimation_metrics), path(gvcf_metrics)

    output:
    tuple val(meta), path("${prefix}.dragen_dx_qc.csv"), emit: csv
    tuple val("${task.process}"), val('dragenutils'), eval("dragen-utils --version"), topic: versions, emit: versions_dragenutils

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"

    """
    dragen-utils dx_qc \\
        ${prefix} \\
        ${wgs_coverage_metrics} \\
        ${cnv_metrics} \\
        ${mapping_metrics} \\
        ${ploidy_estimation_metrics} \\
        ${gvcf_metrics} \\
        > ${prefix}.dragen_dx_qc.csv
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"

    """
    touch ${prefix}.dragen_dx_qc.csv
    """
}
