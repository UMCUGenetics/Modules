/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES / SUBWORKFLOWS / FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/


include { GATK4_HAPLOTYPECALLER } from '../../../modules/nf-core/gatk4/haplotypecaller/main'
include { GATK4_GENOTYPEGVCFS } from '../../../modules/nf-core/gatk4/haplotypecaller/main'

workflow BAM_FP {

    take:

    ch_bam // channel: [ val(meta), [ bam ] ]
    ch_bai // channel: [ val(meta), [ bai ] ]
    ch_ref // channel: [ val(meta), [ ref.fasta ]]
    ch_ref_index // channel: [ val(meta), [ ref.fasta.fai ]]
    ch_intervals // channel: [ val(meta), [ intervals ]]
    ch_dict // channel: [ val(meta), [ ref.dict ]]
    ch_db_snp // channel: [ val(meta), [ db_snp.vcf ]]
    ch_db_snp_tbi // channel: [ val(meta), [db_snp.vcf.tbi ]]

    main:

    
    
    ch_bam
      .join(ch_bai)
      .combine(ch_intervals)
      .map{meta, bam, bai, meta2, intervals -> 
        return tuple([meta, bam, bai, intervals, [] ])
      }
      .set{ ch_bam_bai_intervals }

    GATK4_HAPLOTYPECALLER(
        ch_bam_bai_intervals,
        ch_ref,
        ch_ref_index,
        ch_dict,
        ch_db_snp,
        ch_db_snp_tbi


    )
    //GATK4_GENOTYPEGVCFS()

    emit:
    fp         = GATK4_HAPLOTYPECALLER.out.vcf           // channel: [ val(meta), [vcf?] ]

}
