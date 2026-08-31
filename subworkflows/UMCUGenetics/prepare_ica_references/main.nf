
include { ICA_UNTARGENOMEBUNDLE        } from '../../../modules/UMCUGenetics/ica/untargenomebundle/main'
include { SAMTOOLS_FAIDX                 } from '../../../modules/nf-core/samtools/faidx/main'
include { GATK4_CREATESEQUENCEDICTIONARY } from '../../../modules/nf-core/gatk4/createsequencedictionary/main'

workflow PREPARE_ICA_REFERENCES {
    take:
    genome_tar

    main:

    ICA_UNTARGENOMEBUNDLE(genome_tar)

    SAMTOOLS_FAIDX(
        ICA_UNTARGENOMEBUNDLE.out.fasta.map {meta, fasta -> [meta, fasta, []]},
        false
    )

    GATK4_CREATESEQUENCEDICTIONARY(ICA_UNTARGENOMEBUNDLE.out.fasta)


    emit:
    genome_fasta = ICA_UNTARGENOMEBUNDLE.out.fasta
    genome_fai   = SAMTOOLS_FAIDX.out.fai
    genome_dict  = GATK4_CREATESEQUENCEDICTIONARY.out.dict
    genome_dir   = ICA_UNTARGENOMEBUNDLE.out.dragen_ref

}
