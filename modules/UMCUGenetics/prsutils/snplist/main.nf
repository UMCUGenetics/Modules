process PRSUTILS_SNPLIST {
    tag "${meta.id}"

    container "ghcr.io/umcugenetics/prs-utils:1.0.0"

    input:
    tuple val(meta), path(scoring_file)

    output:
    tuple val(meta), path("*_snplist.list"), emit: list
    tuple val("${task.process}"), val('prsutils'), eval('prs-utils --version'), emit: versions_SNPlist, topic: versions

    script:
    def prefix = task.ext.prefix ?: meta.id
    def args = task.ext.args ?: ""
    """
    prs-utils get-snp-list \\
        --scoring_file ${scoring_file} \\
        --prefix ${prefix} \\
        ${args}

    export prefix=${prefix}
    """

    stub:
    def prefix = task.ext.prefix ?: meta.id
    """
    touch ${prefix}_snplist.list
    touch ${prefix}_snplist.tsv
    """
}
