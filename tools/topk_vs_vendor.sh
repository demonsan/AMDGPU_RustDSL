#!/usr/bin/env bash
# topk_softmax: ours vs aiter, same run (tokens 4096 / 65536 / 524288); needs build/b2/vendor_membound_host (tools/vendor_membound.sh) and the two kernels emitted as build/b2/*.co
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
TK=build/b2/topk_host; V=build/b2/vendor_membound_host
$TK build/b2/topk_softmax_bf16_e128_k8.co topk_softmax_bf16_e128_k8 4096 128 8 plain 100 | tail -1
TOPK=8 $V topk_softmax 4096 128 100
$TK build/b2/topk_softmax_bf16_e128_k8.co topk_softmax_bf16_e128_k8 65536 128 8 plain 100 | tail -1
TOPK=8 $V topk_softmax 65536 128 100
$TK build/b2/topk_softmax_bf16_e128_k8.co topk_softmax_bf16_e128_k8 524288 128 8 plain 100 | tail -1
TOPK=8 $V topk_softmax 524288 128 100
$TK build/b2/topk_softmax_renorm_bf16_e256_k8.co topk_softmax_renorm_bf16_e256_k8 4096 256 8 renorm 100 | tail -1
TOPK=8 $V topk_softmax 4096 256 100
$TK build/b2/topk_softmax_renorm_bf16_e256_k8.co topk_softmax_renorm_bf16_e256_k8 65536 256 8 renorm 100 | tail -1
TOPK=8 $V topk_softmax 65536 256 100
$TK build/b2/topk_softmax_renorm_bf16_e256_k8.co topk_softmax_renorm_bf16_e256_k8 524288 256 8 renorm 100 | tail -1
TOPK=8 $V topk_softmax 524288 256 100
