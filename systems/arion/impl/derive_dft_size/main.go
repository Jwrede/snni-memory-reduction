// Derives the split of ARION's 210.6 GiB of bootstrapping DFT matrices between CoeffsToSlots and
// SlotsToCoeffs (both built in bootstrapping/evaluator.go:231,235; identical census stacks).
//
// Size of a dft.Matrix from its diagonal index sets and level pair only:
//
//     bytes = SUM over sub-matrices of  len(Vec) * (LevelQ+1 + LevelP+1) * N * 8
//
// (lintrans.NewLinearTransformation: one ringQP.NewPoly() per BSGS key, one limb per modulus, N
// uint64 per limb). Index sets from lattigo's computeBootstrappingDFTIndexMap. No run needed.
// Parameters transcribed from the image's configs/config.go:327 (SetHEParams); the CKKS parameter
// line is printed in ARION's format for comparison. Check: predicted total vs census 210.6 GiB.
package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/dft"
	"github.com/tuneinsight/lattigo/v6/circuits/common/lintrans"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
)

// polyCount mirrors lintrans.NewLinearTransformation exactly: it calls the same two exported
// helpers on the same arguments and counts the keys that would be allocated, instead of
// allocating them.
func polyCount(diags []int, logdSlots, logBSGSRatio int) (n int, n1 int) {
	cols := 1 << logdSlots
	keys := map[int]bool{}
	if logBSGSRatio < 0 {
		for _, i := range diags {
			idx := i
			if idx < 0 {
				idx += cols
			}
			keys[idx] = true
		}
		return len(keys), 0
	}
	n1 = lintrans.FindBestBSGSRatio(diags, cols, logBSGSRatio)
	index, _, _ := lintrans.BSGSIndex(diags, cols, n1)
	for j := range index {
		for _, i := range index[j] {
			keys[j+i] = true
		}
	}
	return len(keys), n1
}

// matrixBytes reproduces dft.NewMatrixFromLiteral's allocation without performing it.
func matrixBytes(d dft.MatrixLiteral, params ckks.Parameters, label string) uint64 {
	logSlots := d.LogSlots
	logdSlots := logSlots
	if maxLogSlots := params.LogMaxDimensions().Cols; logdSlots < maxLogSlots && d.Format == dft.RepackImagAsReal {
		logdSlots++
	}

	N := uint64(1) << uint(params.LogN())
	limbs := uint64(d.LevelQ+1) + uint64(d.LevelP+1)
	bytesPerPoly := limbs * N * 8

	idxMaps := dft.SNNIIndexMap(d, params.LogN())

	fmt.Printf("\n%s\n", label)
	fmt.Printf("  Type=%v  LogSlots=%d  LevelQ=%d  LevelP=%d  Levels=%v  Format=%v  LogBSGSRatio=%d\n",
		d.Type, d.LogSlots, d.LevelQ, d.LevelP, d.Levels, d.Format, d.LogBSGSRatio)
	fmt.Printf("  depth(actual)=%d  depth(factorisation)=%d  sub-matrices=%d\n",
		d.Depth(true), d.Depth(false), len(idxMaps))
	fmt.Printf("  one poly = (%d+1 Q + %d+1 P) limbs * %d coeffs * 8 B = %s\n",
		d.LevelQ, d.LevelP, N, human(bytesPerPoly))

	var total uint64
	var totalPolys int
	for i, m := range idxMaps {
		diags := utils.GetKeys(m)
		n, n1 := polyCount(diags, logdSlots, d.LogBSGSRatio)
		total += uint64(n) * bytesPerPoly
		totalPolys += n
		fmt.Printf("    sub-matrix %d: %4d non-zero diagonals, N1=%d -> %4d polys  %s\n",
			i, len(diags), n1, n, human(uint64(n)*bytesPerPoly))
	}
	fmt.Printf("  TOTAL %s in %d polynomials\n", human(total), totalPolys)
	return total
}

func human(b uint64) string {
	const g = 1024.0 * 1024.0 * 1024.0
	return fmt.Sprintf("%.2f GiB", float64(b)/g)
}

func main() {
	// Transcribed from the measured image, configs/config.go:327 SetHEParams. The workload marker
	// of the published run says model=bert_base_data_5, and that run printed
	// "logN=16, logSlots=15, H=192, logQP=928.00, levels=14, scale=2^45", which is this literal.
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{58, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
		LogP:            []int{60, 60, 60, 60},
		LogDefaultScale: 45,
		Xs:              ring.Ternary{H: 192},
	})
	if err != nil {
		panic(err)
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(params, bootstrapping.ParametersLiteral{
		LogN: utils.Pointy(params.LogN()),
		LogP: []int{60, 60, 60, 60},
		Xs:   params.Xs(),
	})
	if err != nil {
		panic(err)
	}

	// The same line ARION prints, so this derivation can be checked against the measured run.
	fmt.Printf("CKKS parameters: logN=%d, logSlots=%d, H=%d, logQP=%.2f, levels=%d, scale=2^%d\n",
		btpParams.ResidualParameters.LogN(),
		btpParams.ResidualParameters.LogMaxSlots(),
		btpParams.ResidualParameters.XsHammingWeight(),
		params.LogQP(),
		btpParams.ResidualParameters.MaxLevel(),
		btpParams.ResidualParameters.LogDefaultScale())

	bp := btpParams.BootstrappingParameters
	c2s := matrixBytes(btpParams.CoeffsToSlotsParameters, bp, "CoeffsToSlots (eval.C2SDFTMatrix, evaluator.go:231)")
	s2c := matrixBytes(btpParams.SlotsToCoeffsParameters, bp, "SlotsToCoeffs (eval.S2CDFTMatrix, evaluator.go:235)")

	tot := c2s + s2c
	fmt.Printf("\nBOTH RESIDENT: %s\n", human(tot))
	fmt.Printf("  CoeffsToSlots %s  %.1f%%\n", human(c2s), 100*float64(c2s)/float64(tot))
	fmt.Printf("  SlotsToCoeffs %s  %.1f%%\n", human(s2c), 100*float64(s2c)/float64(tot))
	fmt.Printf("\nHolding one at a time saves the smaller set: %s\n", human(min64(c2s, s2c)))
	fmt.Printf("Census measured the pair at 210.6 GiB; this derivation says %s (%.1f%% of it).\n",
		human(tot), 100*float64(tot)/(210.6*1024*1024*1024))
}

func min64(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}
