#!/bin/bash
set -e

nextflow \
    -C "nextflow.local.config" \
    run \
    main.nf \
    --network-type params/network_type \
    --encoding-type params/encoding_type \
    --encoding-length params/encoding_length \
    --germline-sequence python/data/gpt.fasta \
    --aid-context-model python/data/aid_logistic_3mer.csv \
    -work-dir output/work/ \
    -resume
