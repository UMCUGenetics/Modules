process PRSUTILS_SAMPLEQC {
    tag "$meta.id"
    label 'process_low'

    container "ghcr.io/umcugenetics/prs-utils:1.0.0"

    input:
    tuple val(meta), path(scores), path(ancestry), path(model_summary)

    output:
    tuple val(meta), path("*_qc.tsv"), emit: score_qc
    tuple val("${task.process}"), val('prsutils'), eval('prs-utils --version'), emit: versions_prs_utils_sample_qc, topic: versions

    script:
    def prefix = task.ext.prefix ?: meta.id
    def args = task.ext.args ?: ""
    """
    prs-utils sample-qc \\
        ${args} \\
        --sample ${meta.sample_id} \\
        --model ${meta.model_id} \\
        --alpha ${meta.alpha} \\
        --scores ${scores} \\
        --ancestry ${ancestry} \\
        --model-summary ${model_summary} \\
        --output ${prefix}_qc.tsv
    """

    stub:
    def prefix = task.ext.prefix ?: meta.id
    """
    touch ${prefix}_qc.tsv
    """
}
