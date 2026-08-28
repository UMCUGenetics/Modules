
include { UNTAR_GENOME                   } from '../../../modules/UMCUGenetics/untar/genome/main'
include { SAMTOOLS_FAIDX                 } from '../../../modules/nf-core/samtools/faidx/main'
include { GATK4_CREATESEQUENCEDICTIONARY } from '../../../modules/nf-core/gatk4/createsequencedictionary/main'

workflow PREPARE_ICA_REFERENCES {
    take:
    genome_tar

    main:

    UNTAR_GENOME(genome_tar)

    SAMTOOLS_FAIDX(
        UNTAR_GENOME.out.fasta.map {meta, fasta -> [meta, fasta, []]},
        false
    )

    GATK4_CREATESEQUENCEDICTIONARY(UNTAR_GENOME.out.fasta)


    emit:
    genome_fasta = UNTAR_GENOME.out.fasta
    genome_fai   = SAMTOOLS_FAIDX.out.fai
    genome_dict  = GATK4_CREATESEQUENCEDICTIONARY.out.dict
    genome_dir   = UNTAR_GENOME.out.dragen_ref

}
