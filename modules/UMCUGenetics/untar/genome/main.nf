process UNTAR_GENOME {
    tag "${archive}"
    label 'process_single'

    container "${workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container
        ? 'https://community-cr-prod.seqera.io/docker/registry/v2/blobs/sha256/52/52ccce28d2ab928ab862e25aae26314d69c8e38bd41ca9431c67ef05221348aa/data'
        : 'community.wave.seqera.io/library/coreutils_grep_gzip_lbzip2_pruned:838ba80435a629f8'}"

    input:
    tuple val(meta), path(archive)

    output:
    tuple val(meta), path('genome'), emit: untar
    tuple val("${task.process}"), val('UNTAR'), eval('echo 1.0.0'), emit: versions_untar_genome, topic: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    mkdir genome
    tar -xavf ${archive} -C genome
    """
}
