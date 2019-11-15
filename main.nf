#!/usr/bin/env nextflow

params.outdir = "$baseDir/output/"
params.dryRun = false

println params
if( params.dryRun ){
    dryFlag = '--dry'
    outdir = params.outdir + '/.dry/'
} 
else {
    dryFlag = ''
    outdir = params.outdir
}

/*
 * this assumes params will be in plain text files (one for each parameter) with each value on a new line 
 * it may make more sense eventually to have all parameters in a JSON file and parse that here so that 
 * the file on its own makes more sense.
 */
network_type_ch = Channel.fromPath(params.network_type).splitText().map{ it -> it.trim() }
encoding_type_ch = Channel.fromPath(params.encoding_type).splitText().map{ it -> it.trim() }
encoding_length_ch = Channel.fromPath(params.encoding_length).splitText().map{ it -> it.trim() }

/*
 * do this process for all combinations of the 3 parameters
 */
process pythoncli {
    container "thayerfisher/shmpy:latest"

// uncomment to echo std out from each command run by nextflow, otherwise it will be ignored
    echo true

    input:
    each network_type from network_type_ch
    each encoding_type from encoding_type_ch
    each encoding_length from encoding_length_ch

    output:
    file "labels"
    file "preds"
    file "loss"
    file "model"

    publishDir "${outdir}"

    """
    cli.py ${dryFlag} train ${network_type} ${encoding_type} ${encoding_length} ${outdir}
    """
}
