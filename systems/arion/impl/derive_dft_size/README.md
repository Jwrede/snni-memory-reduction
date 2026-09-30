# derive_dft_size: size of ARION's bootstrapping DFT matrices

Derivation, not a run. The census attributes 210.6 GiB to `lintrans.NewLinearTransformation` under
`dft.NewMatrixFromLiteral` but cannot split it (identical stacks). A `dft.Matrix` is one
`ringqp.Poly` per BSGS key per sub-matrix:

```
bytes = SUM over sub-matrices of  len(Vec) * (LevelQ+1 + LevelP+1) * N * 8
```

## Running

```
L=$(go env GOMODCACHE)/github.com/tuneinsight/lattigo/v6@v6.1.1
cp -r "$L" ./lattigo && chmod -R u+w ./lattigo
diff -r "$L" ./lattigo            # must be silent: the copy is verbatim v6.1.1
cp snni_indexmap.go ./lattigo/circuits/ckks/dft/
mkdir -p ./lattigo/snni_dftsplit && cp main.go ./lattigo/snni_dftsplit/
(cd lattigo && go run ./snni_dftsplit/)
```

`snni_indexmap.go` forwards to lattigo's `computeBootstrappingDFTIndexMap` (value-free twin of
`GenMatrices`); no diagonals are materialised.

## Result (2026-08-07)

```
CoeffsToSlots   93 polys x 34 limbs x 524,288 B = 1,657,798,656 B   1.54 GiB
SlotsToCoeffs  158 polys x 22 limbs x 524,288 B = 1,822,425,088 B   1.70 GiB
one evaluator                                     3,480,223,744 B   3.24 GiB
```

Parameters from the image's `configs/config.go:327` (`SetHEParams`):

```
derived   logN=16, logSlots=15, H=192, logQP=928.00, levels=14, scale=2^45
run 1 log logN=16, logSlots=15, H=192, sigma={3.20 19.20}, logQP=928.00, levels=14, scale=2^45
```

```
210.6 GiB measured / 3.24122 GiB derived = 64.98
```

`pkg/btp/matbtp.go:60` builds a private evaluator per worker (`DeepCopyBootstrapEvaluator` ->
`bootstrapping.NewEvaluator`, both DFT sets); `threads=64`. 65 x 3.24122 GiB = 210.68 GiB (0.04% from
the census). Basis of `a1_btpeval_shallowcopy`; retires `a1_dft_one_at_a_time`.
