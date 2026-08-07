process PRSUTILS_MERGEPRSMQC {
    label 'process_single'

    container "ghcr.io/umcugenetics/prs-utils:1.0.0"

    input:
    path(tsvs)

    output:
    path("prs_scores_mqc.tsv"), emit: mqc_tsv
    tuple val("${task.process}"), val('prsutils'), eval('prs-utils --version'), emit: versions_prs_utils_merge_prs_mqc, topic: versions

    script:
    """
    prs-utils merge-prs-mqc ${tsvs.join(' ')}
    """

    stub:
    """
    touch prs_scores_mqc.tsv
    """
}
