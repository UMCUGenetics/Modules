process ICA_UNTARGENOMEBUNDLE {
    tag "${meta.id}"
    label 'process_single'

    container "${workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container
        ? 'https://community-cr-prod.seqera.io/docker/registry/v2/blobs/sha256/52/52ccce28d2ab928ab862e25aae26314d69c8e38bd41ca9431c67ef05221348aa/data'
        : 'community.wave.seqera.io/library/coreutils_grep_gzip_lbzip2_pruned:838ba80435a629f8'}"

    input:
    tuple val(meta), path(genome_tar)

    output:
    tuple val(meta), path('dragen_ref'),   emit: dragen_ref
    tuple val(meta), path('genome.fa'),    emit: fasta
    tuple val(meta), path("genes.gtf.gz"), emit: gtf
    tuple val("${task.process}"), val('tar'), eval('tar --version | sed -n "s/.*tar)//p" | tr -d " "'), emit: versions_ica_untargenomebundle, topic: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def hashtable_version = params.dragen_hashtable_version ?: ''
    """
    tar -xzf ${genome_tar}
    mapfile -t hts < <(find . -name hash_table.cfg -printf '%h\\n' | grep -E '/[0-9]+\$' | sort -V)

    if [ \${#hts[@]} -eq 0 ]; then
        echo "ERROR: no DRAGEN hashtable (hash_table.cfg) found in ${genome_tar}" >&2
        exit 1
    fi


    if [ -n "${hashtable_version}" ]; then
        ref_dir=\$(printf '%s\\n' "\${hts[@]}" | grep -E "/${hashtable_version}\$") || {
            echo "ERROR: hashtable v${hashtable_version} not found. Available: \${hts[*]}" >&2
            exit 1
        }
    else
        ref_dir=\${hts[-1]}          # Highest version
    fi

    echo "Using DRAGEN ref-dir: \$ref_dir" >&2
    ln -s "\$(realpath \$ref_dir)" dragen_ref

    """
}
