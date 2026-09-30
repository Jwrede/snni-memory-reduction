package dft

// SNNIIndexMap exposes computeBootstrappingDFTIndexMap for the ARION a1 derivation.
//
// This is the ONLY addition to an otherwise byte-identical copy of lattigo v6.1.1 (verified by
// `diff -r` against the module cache before this file was written). It adds no logic: it forwards
// to the upstream method so the diagonal sets counted outside the package are the same ones
// NewMatrixFromLiteral builds from.
//
// The value-free index map is upstream's own twin of GenMatrices -- GaloisElements uses it for
// exactly this reason -- so counting from it avoids materialising the plaintext diagonals, which
// on these parameters do not fit in the machine doing the counting.
func SNNIIndexMap(d MatrixLiteral, logN int) []map[int]bool {
	return d.computeBootstrappingDFTIndexMap(logN)
}
