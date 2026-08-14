process DRAGENUTILS_FASTQLIST {
    label 'process_single'

    container 'ghcr.io/umcugenetics/dragen-utils:sha-b1d12b9'

    input:
    path fastq

    output:
    path "fastq_list.csv", emit: csv
    tuple val("${task.process}"), val('dragenutils'), eval("dragen-utils --version"), topic: versions, emit: versions_dragenutils

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    dragen-utils fastq_list > fastq_list.csv
    """

    stub:
    """    
    touch fastq_list.csv
    """
}
