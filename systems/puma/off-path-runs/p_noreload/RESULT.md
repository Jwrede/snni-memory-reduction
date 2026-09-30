# p_noreload: double model load in the driver

Job 46122987, one run, diagnostic driver `impl/snni_bert_puma_noreload.py`, stock image and policy.

`snni_bert_puma.py:124`:

```python
def bert_forward(input_ids, attention_mask, token_type_ids, params):
    model = FlaxBertForSequenceClassification.from_pretrained(MODEL, seed=SEED & 0xFFFF)
    out = model(..., params=params, ...)      # params overwrites those weights immediately
```

SPU's own examples build from the config here (`flax_resnet`, `flax_gpt2`). Hypothesis: the
`_PyBytes_Resize` (1,125,175,296 B) and `PyBytes_FromStringAndSize` (437,739,520 B) rows are this
load.

```
                peak kB      _PyBytes_Resize   PyBytes_FromStringAndSize   wall s
s0_default      7,320,212      1,125,175,296            437,739,520          357
p_noreload      7,287,564      1,125,175,296            437,739,520          334
```

- Both rows byte-identical; peak inside the baseline range (7,272,656 to 7,337,192); gate 1.938e-03
  to 3.391e-03 against s0 (s0 against itself 2.560e-03 to 5.180e-03).
- Wall -6.4%: one model load removed. `bert_forward` is traced where the driver runs; the published
  party is an SPU node.
- Neither caller offset lies in a sized ELF symbol of the interpreter.
