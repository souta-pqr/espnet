#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="eval"

asr_config=myconf/train_asr_cbs_transformer_081616_hop132.yaml
inference_config=myconf/decode_asr_streaming.yaml

asr_tag=0612_shoji_fid_mlt_token_dependency_mechanism_four_level_classification_twostage

# LM settings
lm_config=conf/train_lm.yaml
use_lm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
# speed_perturb_factors="0.9 1.0 1.1"

./asr.sh                                               \
    --ngpu 1                                           \
    --use_streaming false                              \
    --nj 8                                             \
    --inference_nj 8                                   \
    --lang jp                                          \
    --feats_type raw                                   \
    --token_type word                                  \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --asr_tag "${asr_tag}"                             \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --lm_train_text "data/${train_set}/text"           \
    "$@"