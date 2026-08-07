process ANCESTRY_CALC {
    tag "${meta.id}"
    label "process_medium"

    container "ghcr.io/astral-sh/uv:python3.13-bookworm"

    input:
    tuple val(meta), path(eigenvec)
    tuple val(meta2), path(ref_metadata)

    output:
    tuple val(meta), path("*_knn.tsv"), emit: knn_tsv
    tuple val(meta), path("*_knn_pca.png"), emit: knn_pca_plot, optional: true
    tuple val("${task.process}"), val('ancestry_knn'), eval('knn.py --version'), emit: versions_ancestry_knn, topic: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix = task.ext.prefix ?: meta.id
    def args = task.ext.args ?: ""
    """
    knn.py \
        --eig ${eigenvec} \\
        --labels ${ref_metadata} \\
        ${args} \\
        --plot-output ${prefix}_knn_pca.png \\
        --output ${prefix}_knn.tsv
    """

}
