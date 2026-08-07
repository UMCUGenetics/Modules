include { GATK4_HAPLOTYPECALLERALLELES } from '../../../modules/UMCUGenetics/gatk4/haplotypecalleralleles/main'
include { BCFTOOLS_NORM                } from '../../../modules/nf-core/bcftools/norm/main'

workflow BAM_HAPLOTYPECALLER_NORM {
    take:
    ch_samplesheet  // channel: [ val(meta), path(samplesheet)]
    ch_genome_fasta // channel: [ val(meta), path(fasta)]
    ch_genome_index // channel: [ val(meta), path(fai)]
    ch_genome_dict  // channel: [ val(meta), path(dict)]
    ch_dbsnp        // channel: [ val(meta), path(vcf)]
    ch_dbsnp_index  // channel: [ val(meta), path(tbi)]
    ch_snp_list     // channel: [ val(meta), path(list)]
    ch_snp_vcf      // channel: [ val(meta), path(vcf)]


    main:

    ch_per_model = ch_snp_list.join(ch_snp_vcf)

    ch_hc = ch_samplesheet.combine(ch_per_model)
        .multiMap { sm, bam, bai, mm, snplist, vcf, tbi ->
            def meta = [
                id: "${sm.id}__${mm.id}",
                sample_id: sm.id,
                model_id: mm.id,
                mu: mm.mu,
                sd: mm.sd,
                alpha: mm.alpha,
            ]
            input:   [meta, bam, bai, snplist, []]
            alleles: [[id: meta.id], vcf, tbi]
        }

    GATK4_HAPLOTYPECALLERALLELES(
        ch_hc.input,
        ch_genome_fasta,
        ch_genome_index,
        ch_genome_dict,
        ch_dbsnp,
        ch_dbsnp_index,
        ch_hc.alleles
    )

    BCFTOOLS_NORM(
        GATK4_HAPLOTYPECALLERALLELES.out.vcf
            .join(GATK4_HAPLOTYPECALLERALLELES.out.tbi),
        ch_genome_fasta
    )

    emit:
    vcf         = BCFTOOLS_NORM.out.vcf   // channel: [ val(meta), path(vcf)]
    tbi         = BCFTOOLS_NORM.out.index // channel: [ val(meta), path(tbi)]
}
