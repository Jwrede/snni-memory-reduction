# p_goowner: owner view of the Go heap

No new run. `tools/attrib_go/parse_heapdump.py` keeps 24 stack frames per sampled object
(`STACK_KEEP = 24`); `join_residency.py` aggregated on the leaf frame only. Source:
`results/<step>_goheap/run1/objects_go.jsonl.gz`, ordering of that time.

Check: re-aggregation by leaf `rlwe.NewElement` 102,838.1 MB against the published a3 table's
102,791.9 MB (0.045%).

## Depth 2 (caller <- allocating function), a3_threads16, 145,743.3 MB

```
64,895.9 MB  44.5%  schemes/ckks.NewCiphertext                <- core/rlwe.NewElement
37,494.5 MB  25.7%  circuits/common/polynomial...Evaluate     <- core/rlwe.NewElement
24,554.5 MB  16.8%  core/rlwe.NewVectorQP                     <- ring/ringqp.Ring.NewPoly
 8,996.8 MB   6.2%  schemes/ckks.Evaluator.mulRelin           <- rlwe.Element.Resize
 4,197.5 MB   2.9%  core/rlwe.NewEvaluatorBuffers             <- ring/ringqp.Ring.NewPoly
```

## First `Arion/` frame, a3_threads16 (100% of the heap carries one)

```
56,791.5 MB  39.0%  Arion/pkg/math/activation.ApproximatePolynomialChebyshevMT.func1
                        via rlwe.NewElement 43,199 MB, via Element.Resize 8,997 MB
45,076.7 MB  30.9%  Arion/pkg/math/matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT.func1
23,882.4 MB  16.4%  Arion/pkg/he.ParallelGenGaloisKeys.func1        via ringqp.Ring.NewPoly
14,491.3 MB   9.9%  Arion/pkg/btp.CiphertextMatricesBootstrappingMT.func1
 4,816.7 MB   3.3%  Arion/pkg/he.GenerateKeysAndBtsKeysMT
```

Against the 219,644 MB peak (heap objects + `go:runtime_retained` 63,839 MB + `anon:residual`
10,211 MB), rank one is 25.9%. a0-a2 of that time were parsed with `STACK_KEEP` 6 (over 95% without an
Arion frame); their dumps are gone.

## `rlwe.NewElement` across steps

```
                         a0        a1        a2        a3
ringqp.Ring.NewPoly   326,687    42,122    42,128    32,196
go:runtime_retained   158,042   153,806    66,937    63,839
rlwe.NewElement        52,430   107,783   120,680   102,792
HOST peak (GiB)           542       308       242       205
```

The jump at a1 is a phase change (a1 removed 284 GB of `NewPoly`; the peak moved to a phase where
`NewElement` is larger).

## Source at rank one

`pkg/math/activation/polynomialMT.go:13`:

```go
newCiphertexts := make([]*rlwe.Ciphertext, numCts)
...
for i := start; i < end; i++ {
    res, err := localEval.MulNew(ctMats.Ciphertexts[i], scalarmul)   // a new ciphertext
    localEval.Add(res, scalaradd, res)
    localEval.Rescale(res, res)
    res, err = polyEval.Evaluate(res, poly, ckksParams.DefaultScale())
    newCiphertexts[i] = res
}
```

Input and output lists are both whole at the end of the function. Candidate at the time, conditional
on the caller not reading `ctMats` afterwards; became `a2_activation_input_release`.
