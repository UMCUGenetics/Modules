/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES / SUBWORKFLOWS / FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
include { VCF_ANCESTRY             } from '../../../subworkflows/UMCUGenetics/vcf_ancestry/main'
include { VCF_PRS_SCORE            } from '../../../subworkflows/UMCUGenetics/vcf_prs_score/main'
include { PRS_INTERVALS            } from '../../../subworkflows/UMCUGenetics/prs_intervals/main'
include { BAM_HAPLOTYPECALLER_NORM } from '../../../subworkflows/UMCUGenetics/bam_haplotypecaller_norm/main'

include { PRSUTILS_SAMPLEQC        } from '../../../modules/UMCUGenetics/prsutils/sampleqc/main'
include { PRSUTILS_MERGEPRSMQC     } from '../../../modules/UMCUGenetics/prsutils/mergeprsmqc/main'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow BAM_PRS {

    take:
    ch_samplesheet  // channel: [ val(meta), path(bam), path(bai)]
    ch_genome_fasta // channel: [ val(meta), path(fasta)]
    ch_genome_index // channel: [ val(meta), path(fai)]
    ch_genome_dict // channel: [ val(meta), path(dict)]
    ch_dbsnp       // channel: [ val(meta), path(vcf)]
    ch_dbsnp_index // channel: [ val(meta), path(tbi)]
    ch_ancestry_ref_vcf // channel: [ val(meta), path(vcf)]
    ch_ancestry_ref_index // channel: [ val(meta), path(tbi)]
    ch_ancestry_meta // channel: [ val(meta), path(psam)]
    ch_prs_models // channel: [ val(meta), path(csv)]
    assembly_version // val(version)


    main:

    PRS_INTERVALS(ch_prs_models, assembly_version)

    BAM_HAPLOTYPECALLER_NORM(
        ch_samplesheet,
        ch_genome_fasta,
        ch_genome_index,
        ch_genome_dict,
        ch_dbsnp,
        ch_dbsnp_index,
        PRS_INTERVALS.out.list,
        PRS_INTERVALS.out.vcf.join(PRS_INTERVALS.out.tbi)
    )

    VCF_PRS_SCORE(
        BAM_HAPLOTYPECALLER_NORM.out.vcf,
        PRS_INTERVALS.out.normalised_model
    )


    ch_ancestry_vcf = BAM_HAPLOTYPECALLER_NORM.out.vcf
        .map { meta, vcf -> [meta.sample_id, vcf] }
        .groupTuple()
        .map { sid, vcfs -> [[id: sid], vcfs[0]] }

    ch_ancestry_tbi = BAM_HAPLOTYPECALLER_NORM.out.tbi
        .map { meta, tbi -> [meta.sample_id, tbi] }

        .groupTuple()
        .map { sid, tbis -> [[id: sid], tbis[0]] }

    VCF_ANCESTRY(
        ch_ancestry_vcf.join(ch_ancestry_tbi),
        ch_ancestry_ref_vcf.join(ch_ancestry_ref_index),
        ch_ancestry_meta,
        ch_genome_fasta,
        ch_genome_index
    )


    // QC step — combined meta carries (sample_id, model_id); ancestry is per-sample only.
    PRSUTILS_SAMPLEQC(
        VCF_PRS_SCORE.out.ch_score_norm
            .join(VCF_PRS_SCORE.out.ch_score_summary)
            .map { meta, scores, summary -> [meta.sample_id, meta, scores, summary] }
            .combine(
                VCF_ANCESTRY.out.knn_tsv.map { ancestry_meta, knn -> [ancestry_meta.id, knn] },
                by: 0
            )
            .map { _sid, meta, scores, summary, knn -> [meta, scores, knn, summary] }
    )


    PRSUTILS_MERGEPRSMQC(
        PRSUTILS_SAMPLEQC.out.score_qc.map{ _meta, tsv -> tsv }.collect()
    )


    emit:
    PRS_mqc  = PRSUTILS_MERGEPRSMQC.out.mqc_tsv // channel: [ val(meta), path(tsv)]

}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
