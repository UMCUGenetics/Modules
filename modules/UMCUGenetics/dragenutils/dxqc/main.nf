process DRAGENUTILS_DXQC {
    tag "${meta.id}"
    label 'process_single'

    container 'ghcr.io/umcugenetics/dragen-utils:sha-b1d12b9'

    input:
    tuple val(meta), path(wgs_coverage_metrics), path(cnv_metrics), path(mapping_metrics), path(ploidy_estimation_metrics), path(gvcf_metrics)
    val output_file_prefix

    output:
    tuple val(meta), path("${meta.id}_${output_file_prefix}.dragen_dx_qc.csv"), emit: csv
    tuple val("${task.process}"), val('dragenutils'), eval("echo 0.1.0"), topic: versions, emit: versions_dragenutils

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    dragen-utils dx_qc \\
        ${meta.id}_${output_file_prefix} \\
        ${wgs_coverage_metrics} \\
        ${cnv_metrics} \\
        ${mapping_metrics} \\
        ${ploidy_estimation_metrics} \\
        ${gvcf_metrics} \\
        > ${meta.id}_${output_file_prefix}.dragen_dx_qc.csv
    """

    stub:
    """
    touch ${meta.id}_${output_file_prefix}.dragen_dx_qc.csv
    """
}
