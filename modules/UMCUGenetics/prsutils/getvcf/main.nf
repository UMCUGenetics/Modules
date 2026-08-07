process PRSUTILS_GETVCF {
    tag "GATK VCF ${meta.id}"

    container "ghcr.io/umcugenetics/prs-utils:1.0.0"

    input:
    tuple val(meta), path(scoring_file)
    val genome_build

    output:
    tuple val(meta), path("*_genotypes.vcf"), emit: vcf
    tuple val("${task.process}"), val('prsutils'), eval('prs-utils --version'), emit: versions_gatk_vcf, topic: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix = task.ext.prefix ?: meta.id
    """
    prs-utils pgs-to-vcf \\
        ${scoring_file} \\
        ${prefix}_genotypes.vcf \\
        --genome-build ${genome_build}
    """

    stub:
    def prefix = task.ext.prefix ?: meta.id
    """
    touch ${prefix}_genotypes.vcf
    """
}
