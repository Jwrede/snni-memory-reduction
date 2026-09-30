# BumbleBee: stack depth of the object key

Written 2026-08-11. `b1_dot_encode_chunk`, `b2_result_ct_pack_free`, `b4_slice_budget_64m` declared
`seal::util::MemoryPoolHeadMT::get()` (SEAL's pool, through which every allocation passes) as their
object: too coarse for rule 4's test (one change must lift the whole group).

| | |
|---|---|
| probes | `p_depth3`, `p_depth4`, `p_depth5`, jobs 46065405-20, one replicate each |
| image | `bumblebee-b0_default.sif` (the contested `b1` was chosen from `b0`) |
| peaks | 15,480,372 / 15,469,408 / 15,502,380 kB (0.2% band); published `b0` at the time 15,446,988 kB |

| depth | the 6.9 GB object | the 5.1 GB object |
|---|---|---|
| 1 | `seal::util::MemoryPoolHeadMT::get()`, 8445.9 MB, 54.7%, one row for everything | -- |
| 3 | `seal::DynArray<unsigned long>::resize` | `YaclFerretOt::Impl::Impl`, one row |
| 4 | `VectorEncoder::Forward` | `YaclFerretOt::YaclFerretOt`, one row |
| 5 | `MatMatProtocol::EncodeMatrix` | `BasicOTProtocols` at `basic_ot_prot.cc:46` and `:47`, 2560.1 MB each |

Depth 5, party 0, 15,502,380 kB:

```
6919.1 MB  45.7%  matmat_prot.cc:262:MatMatProtocol::EncodeMatrix
2560.1 MB  16.9%  basic_ot_prot.cc:46:BasicOTProtocols::BasicOTProtocols
2560.1 MB  16.9%  basic_ot_prot.cc:47:BasicOTProtocols::BasicOTProtocols
1145.3 MB   7.6%  matmat_prot.cc:296:MatMatProtocol::FusedMulAddInplace
 648.2 MB   4.3%  context.h:212:spu::dynDispatch
 384.1 MB   2.5%  std_function.h:291:_M_invoke <- yacl::Buffer::Buffer(long)
```

| depth | verdict |
|---|---|
| 3 | fails: a container's growth function, no owner |
| 4 | names the encoder stage (`b1`'s target) but merges the OT memory into one 5120.1 MB row |
| 5 | chosen: the operation that builds the 6.9 GB object, and two separate OT allocations (the unit `b5`-`b8` change) |

BOLT also uses depth 5 (program stage there). The whole line was re-measured at depth 5 (tables at
different depths are not comparable). Expectation stated before the runs: peaks unchanged, `b1`'s
object resolves to the encoding object.
