// ARION QK^T-instant target micro-benchmark (analogue of MOAI-CPU p_floor_bench_att).
// Per-head QK^T working set as one set at T=1, parameter literal of Arion configs/config.go
// SetHEParams, every page touched; VmRSS after GC + FreeOSMemory per stage.
// Derived target (systems/arion/MANIFEST.md): 30 attention keys 2,334,720 + relin 77,824 +
// Q,K,rotQ,rotK,V (5 x 64 ct @ level 13) 4,587,520 + scores (128 ct @ level 12) 1,703,936 = 8,704,000 kB.
package main

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func rssKB() int {
	runtime.GC()
	debug.FreeOSMemory()
	f, _ := os.Open("/proc/self/status")
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		if strings.HasPrefix(s.Text(), "VmRSS:") {
			var v int
			fmt.Sscanf(strings.TrimSpace(strings.TrimPrefix(s.Text(), "VmRSS:")), "%d", &v)
			return v
		}
	}
	return -1
}

func block(enc *rlwe.Encryptor, n, level int) []*rlwe.Ciphertext {
	out := make([]*rlwe.Ciphertext, n)
	for i := range out {
		out[i] = enc.EncryptZeroNew(level)
	}
	return out
}

func main() {
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: 16, LogQ: []int{58, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
		LogP: []int{60, 60, 60, 60}, LogDefaultScale: 45, Xs: ring.Ternary{H: 192}})
	if err != nil {
		panic(err)
	}
	r0 := rssKB()
	kgen := rlwe.NewKeyGenerator(p)
	sk := kgen.GenSecretKeyNew()
	enc := rlwe.NewEncryptor(p, sk)
	_ = ckks.NewEncoder(p)
	_ = ckks.NewEvaluator(p, nil)
	ctx := rssKB()
	fmt.Printf("RSS_START_KB=%d\nRSS_CTX_KB=%d\n", r0, ctx)

	var att []int
	for i := 1; i < 12; i++ {
		att = append(att, i*256)
	}
	for i := 1; i < 11; i++ {
		att = append(att, -i*256*12, i*256*12)
	}
	uniq := map[uint64]bool{}
	var gal []uint64
	for _, g := range p.GaloisElements(att) {
		if !uniq[g] {
			uniq[g] = true
			gal = append(gal, g)
		}
	}
	gks := kgen.GenGaloisKeysNew(gal, sk)
	k1 := rssKB()
	fmt.Printf("GALOIS_KEYS=%d RSS_KB=%d KEYS_KB=%d (derived %d)\n", len(gks), k1, k1-ctx, len(gks)*77824)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	k2 := rssKB()
	fmt.Printf("RELIN RSS_KB=%d RLK_KB=%d (derived 77824)\n", k2, k2-k1)

	Q := block(enc, 64, 13)
	K := block(enc, 64, 13)
	rQ := block(enc, 64, 13)
	rK := block(enc, 64, 13)
	V := block(enc, 64, 13)
	b1 := rssKB()
	fmt.Printf("QK_ROT_V RSS_KB=%d BLOCKS_KB=%d (derived 4587520)\n", b1, b1-k2)
	S := block(enc, 128, 12)
	b2 := rssKB()
	fmt.Printf("SCORES RSS_KB=%d SCORES_KB=%d (derived 1703936)\n", b2, b2-b1)

	fmt.Printf("TARGET_MEASURED_KB=%d (instant - ctx; derived 8704000, diff %+.2f%%)\n", b2-ctx, 100*float64(b2-ctx-8704000)/8704000)
	runtime.KeepAlive(gks)
	runtime.KeepAlive(rlk)
	runtime.KeepAlive([][]*rlwe.Ciphertext{Q, K, rQ, rK, V, S})
}
