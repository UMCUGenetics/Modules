
include { UNTAR                          } from '../../../modules/nf-core/untar/main'
include { SAMTOOLS_FAIDX                 } from '../../../modules/nf-core/samtools/faidx/main'
include { GATK4_CREATESEQUENCEDICTIONARY } from '../../../modules/nf-core/gatk4/createsequencedictionary/main'

workflow PREPARE_ICA_REFERENCES {
    take:
    genome_tar

    main:

    UNTAR(genome_tar)

    genome_fasta = UNTAR.out.untar
        .map{ meta, dir -> [meta, file(
                    dir.resolve('genome.fa'),
                    checkIfExists: !workflow.stubRun)
            ]}

    SAMTOOLS_FAIDX(
        genome_fasta.map {meta, fasta -> [meta, fasta, []]},
        false
    )

    GATK4_CREATESEQUENCEDICTIONARY(genome_fasta)


    emit:
    genome_fasta = genome_fasta
    genome_fai   = SAMTOOLS_FAIDX.out.fai
    genome_dict  = GATK4_CREATESEQUENCEDICTIONARY.out.dict
    genome_dir   = UNTAR.out.untar

}
