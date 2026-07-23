# Pretrained Lamar zero-shot / linear-probe ablation

All labels are computational. The immutable dataset and existing checkpoints were not modified or retrained. All ablation configurations were frozen by dev, calibration and thresholds were frozen on the 1:1000 calibration split, and the test suite was scored only after `PRETEST_FROZEN.json`.

## Paper-level ablation table

| Model | Parameters trained | Dev AP | Test AP | Test Precision | Test Recall | FP/Million |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA Lamar | 297,217 | 0.825438 | 0.171958 | 0.476190 | 0.124224 | 136.646 |
| Partial Lamar (2 blocks) | 14,179,585 | 0.775930 | 0.103259 | 0.391304 | 0.055901 | 86.957 |
| Full fine-tuning | 85,854,098 | 0.799066 | 0.096254 | 0.400000 | 0.062112 | 93.168 |
| Lamar linear probe center | 769 | 0.698583 | 0.066262 | 0.333333 | 0.024845 | 49.689 |
| Frozen Lamar head | 2,305 | 0.686136 | 0.064390 | 0.285714 | 0.049689 | 124.224 |
| Lamar zero-shot center | 0 | 0.591302 | 0.036180 | 0.066667 | 0.006211 | 86.957 |
| CNN | 59,969 | 0.365052 | 0.013912 | 0.111111 | 0.012422 | 99.379 |
| k-mer Logistic | 344 | 0.330512 | 0.006598 | 0.000000 | 0.000000 | 0.000 |
| Lamar linear probe CLS | 769 | 0.345897 | 0.006238 | 0.000000 | 0.000000 | 0.000 |
| Lamar linear probe masked mean | 769 | 0.365517 | 0.005391 | 0.000000 | 0.000000 | 0.000 |
| Lamar linear probe mean | 769 | 0.365514 | 0.005391 | 0.000000 | 0.000000 | 0.000 |
| Lamar zero-shot masked mean | 0 | 0.220907 | 0.004307 | 0.000000 | 0.000000 | 31.056 |
| Lamar zero-shot mean | 0 | 0.220907 | 0.004307 | 0.000000 | 0.000000 | 31.056 |
| Lamar zero-shot CLS | 0 | 0.209600 | 0.003320 | 0.000000 | 0.000000 | 74.534 |
| Random baseline | 0 | 0.090802 | 0.001143 | 0.000000 | 0.000000 | 86.957 |

## Required conclusions

1. **Does pretrained Lamar zero-shot exceed CNN/k-mer?** Best zero-shot is `lamar_zero_shot_center`: dev AP `0.591302`, test AP `0.036180`. K-mer test AP is `0.006598` and CNN test AP is `0.013912`.
2. **How much editing signal is in the embedding?** The task-unadapted centroid score reaches test AP `0.036180` versus the empirical random baseline `0.001143`. This is labeled zero-shot representation evaluation even though train labels estimate class centroids.
3. **Center versus mean pooling:** center zero-shot/linear test AP are `0.036180` / `0.066262`; mean values are `0.004307` / `0.005391`. Mean and masked mean are mathematically equivalent here because every sequence has 101 nucleotides and no padding.
4. **Linear probe gain:** best linear probe improves over best zero-shot by `+0.030082` test AP.
5. **Frozen head versus linear probe:** frozen-head test AP `0.064390` versus linear-probe `0.066262`.
6. **LoRA gain:** LoRA improves over linear probe by `+0.105695` test AP.
7. **Representation adaptation versus classifier:** logistic probing tests a simple classifier on fixed embeddings; the remaining LoRA gain is consistent with representation adaptation, but this observational ablation cannot uniquely attribute every gain.
8. **Embedding versus metadata-only:** best embedding-only linear dev AP `0.698583`; metadata-only dev AP `0.759266`; combined dev AP `0.813666`.
9. **Shortcut risk:** the largest absolute PC–metadata Pearson correlation is `+0.637` (`PC3` with `gc_fraction`). Metadata remains strongly predictive, so coverage/expression-related data-generation bias cannot be excluded even though Lamar receives sequence only.
10. **Complete ordering:** see the frozen test table above. Partial 2-block AP is `0.103259` and full fine-tuning AP is `0.096254`; neither model was retrained.

## Final zero-shot and probe configurations

- zero-shot: `{'candidate': 'zero_center_diagonal_mahalanobis', 'dev_average_precision': 0.5913019371380754, 'dev_brier': 0.21002346427909754, 'dev_ece': 0.3742221430574505, 'dev_f1': 0.33890746934225197, 'dev_fn': 7, 'dev_fp': 586, 'dev_fp_per_million': 368553.4591194968, 'dev_mcc': 0.34193465409319423, 'dev_pr_auc': 0.5895018706482444, 'dev_precision': 0.20596205962059622, 'dev_recall': 0.9559748427672956, 'dev_roc_auc': 0.9211740041928721, 'dev_threshold': 0.5, 'dev_tn': 1004, 'dev_tp': 152, 'method': 'diagonal_mahalanobis', 'representation': 'center', 'trainable_parameters': 0}`
- linear probe: `{'C': 0.01, 'candidate': 'linear_center_C0.01_weight_none', 'class_weight': 'none', 'dev_average_precision': 0.6985833992748813, 'dev_brier': 0.0456795900380659, 'dev_ece': 0.02678079547183534, 'dev_f1': 0.6015625, 'dev_fn': 82, 'dev_fp': 20, 'dev_fp_per_million': 12578.61635220126, 'dev_mcc': 0.5924754708383241, 'dev_pr_auc': 0.697560511003127, 'dev_precision': 0.7938144329896907, 'dev_recall': 0.48427672955974843, 'dev_roc_auc': 0.9284996637791227, 'dev_threshold': 0.5, 'dev_tn': 1570, 'dev_tp': 77, 'representation': 'center', 'trainable_parameters': 769}`
- pretrained checkpoint: `${LAMAR_SOURCE_ROOT}/base_2k/mammalian80D_2048len1mer1sw_80M/checkpoint-250000/model.safetensors`
- pretrained backbone parameters: `85851793`
- backbone trainable parameters during embedding extraction: `0`

## Limitations

- “Zero-shot” here means no gradient-based Lamar adaptation; labeled train centroids are still estimated.
- Train negatives are a deterministic 1:10, without-replacement sample from the train-only dynamic pool, matching the selected LoRA ratio.
- External basewise mappability was unavailable in the underlying dataset.
- Mean and masked-mean rows are retained for protocol completeness but are identical for these fixed-length, unpadded inputs.
