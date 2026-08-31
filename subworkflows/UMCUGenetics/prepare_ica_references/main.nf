
include { ICA_UNTAR_GENOME_BUNDLE        } from '../../../modules/UMCUGenetics/ica/untar_genome_bundle/main'
include { SAMTOOLS_FAIDX                 } from '../../../modules/nf-core/samtools/faidx/main'
include { GATK4_CREATESEQUENCEDICTIONARY } from '../../../modules/nf-core/gatk4/createsequencedictionary/main'

workflow PREPARE_ICA_REFERENCES {
    take:
    genome_tar

    main:

    ICA_UNTAR_GENOME_BUNDLE(genome_tar)

    SAMTOOLS_FAIDX(
        ICA_UNTAR_GENOME_BUNDLE.out.fasta.map {meta, fasta -> [meta, fasta, []]},
        false
    )

    GATK4_CREATESEQUENCEDICTIONARY(ICA_UNTAR_GENOME_BUNDLE.out.fasta)


    emit:
    genome_fasta = ICA_UNTAR_GENOME_BUNDLE.out.fasta
    genome_fai   = SAMTOOLS_FAIDX.out.fai
    genome_dict  = GATK4_CREATESEQUENCEDICTIONARY.out.dict
    genome_dir   = ICA_UNTAR_GENOME_BUNDLE.out.dragen_ref

}
