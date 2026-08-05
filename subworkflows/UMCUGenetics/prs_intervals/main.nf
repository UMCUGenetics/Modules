include { BCFTOOLS_SORT      } from '../../../modules/nf-core/bcftools/sort/main'
include { PRSUTILS_SNPLIST   } from '../../../modules/UMCUGenetics/prsutils/snplist/main'
include { PRSUTILS_GETVCF    } from '../../../modules/UMCUGenetics/prsutils/getvcf/main'
include { PGSCATALOG_COMBINE } from '../../../modules/UMCUGenetics/pgscatalog/combine/main'


workflow PRS_INTERVALS {
    take:
    ch_PRS_model     // channel: [ val(meta), path(scoring_file)]
    assembly_version // value: GRCh38

    main:

    PRSUTILS_SNPLIST(
        ch_PRS_model
    )

    PRSUTILS_GETVCF(
        ch_PRS_model,
        assembly_version
    )

    BCFTOOLS_SORT(PRSUTILS_GETVCF.out.vcf)

    PGSCATALOG_COMBINE(
        ch_PRS_model,
        assembly_version
    )

    emit:
    list             = PRSUTILS_SNPLIST.out.list               // channel: [ val(meta), path(list)]
    vcf              = BCFTOOLS_SORT.out.vcf                   // channel: [ val(meta), path(vcf)]
    tbi              = BCFTOOLS_SORT.out.index                 // channel: [ val(meta), path(tbi)]
    normalised_model = PGSCATALOG_COMBINE.out.normalised_model // channel: [ val(meta), path(model)]
}
