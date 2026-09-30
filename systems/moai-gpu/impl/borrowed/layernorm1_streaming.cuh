// ---------------------------------------------------------------------------------------------------------
// STREAMING variant of layernorm (LN1), BORROWED VERBATIM from the retired campaign's
// `s9_streaming_ln1` image (/moai/src/include/source/non_linear_func/layernorm.cuh:1504-1687).
// Not reimplemented: the arithmetic of a LayerNorm over 768 secret-shared ciphertexts is exactly the
// kind of code where a re-derivation would put a new numerical path in front of the gate, and the
// gate is what certifies this whole line.
//
// The donor's own header records what makes this admissible as a memory lever rather than a
// different computation, and it is the same argument our `layernorm2_streaming.cuh` carries:
// layernorm and layernorm2 differ in exactly TWO constants (the variance normaliser 1/768^2 against
// 1/768^3, and the gamma scaling gamma/sqrt(768) against gamma/768), and everything else -- the
// three passes, the 48x16 variance grouping, the scales and the levels -- is the layernorm2
// streaming code that that campaign's `v1_verify` measured bit-exact, maxabsdiff = 0.
// ---------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------
// STREAMING variant of layernorm (LN1). layernorm and layernorm2 are identical except for TWO constants:
// the variance normaliser (1/768^2 here vs 1/768^3 in layernorm2) and the gamma scaling (gamma/sqrt(768)
// here vs gamma/768). Everything else -- the three passes, the 48x16 variance grouping, the scales and
// levels -- is the layernorm2_streaming code that v1_verify proved bit-exact (maxabsdiff=0).
// ---------------------------------------------------------------------------------------------------------
vector<PhantomCiphertext> layernorm_streaming(std::vector<HostCipher> &x_host, vector<double> &gamma, vector<double> &beta,
                                              const vector<int> &bias_vec, PhantomContext &context,
                                              PhantomRelinKey &relin_keys, PhantomSecretKey &sk)
{
  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);
  Decryptor decryptor(&context, &sk);

  size_t slot_count = encoder.slot_count();
  int num_ct = (int)x_host.size();
  if (num_ct != 768)
  {
    cout << "ERROR: INPUT SIZE IS NOT CORRECT. " << endl;
  }

  // input scale + the n=768 plaintext, taken from CT 0 (same as layernorm2's x[0])
  double scale;
  PhantomPlaintext d;
  {
    PhantomCiphertext x0;
    reload_cipher_from_host(x0, x_host[0]);
    scale = x0.scale();
    vector<double> ecd_n(slot_count, 0);
    for (size_t i = 0; i < slot_count; ++i)
    {
      if (bias_vec[i] == 1)
      {
        ecd_n[i] = 768.0;
      }
    }
    encoder.encode(ecd_n, x0.params_id(), x0.scale(), d);
  }

  // nx[i] = x[i] * 768 (rescaled back to `scale`), recomputed on demand from the host copy
  auto make_nx = [&](int i) {
    PhantomCiphertext nx;
    reload_cipher_from_host(nx, x_host[i]);
    evaluator.multiply_plain_inplace(nx, d);
    evaluator.rescale_to_next_inplace(nx);
    nx.scale() = scale;
    return nx;
  };

  // ---- pass 1: ave_x = x[0]+...+x[767] ----
  PhantomCiphertext ave_x;
  for (int i = 0; i < num_ct; ++i)
  {
    PhantomCiphertext xi;
    reload_cipher_from_host(xi, x_host[i]);
    if (i == 0)
      ave_x = xi;
    else
      evaluator.add_inplace(ave_x, xi);
  }

  // bring ave_x to the nx level (layernorm2 does this against nx[0].params_id())
  {
    PhantomCiphertext nx0 = make_nx(0);
    evaluator.mod_switch_to_inplace(ave_x, nx0.params_id());
  }
  ave_x.scale() = scale;

  // ---- pass 2: var = sum_i (nx[i]-ave_x)^2, accumulated in 48 groups of 16 (same grouping as layernorm2) ----
  vector<PhantomCiphertext> temp_var(48);
  for (int i = 0; i < 48; ++i)
  {
    PhantomCiphertext temp_i;
    for (int j = 0; j < 16; ++j)
    {
      PhantomCiphertext temp = make_nx(i * 16 + j);
      evaluator.sub_inplace(temp, ave_x);
      evaluator.square_inplace(temp);
      if (j == 0)
      {
        temp_i = temp;
      }
      else
      {
        evaluator.add_inplace(temp_i, temp);
      }
    }
    temp_var[i] = temp_i;
  }

  PhantomCiphertext var = temp_var[0];
  for (int i = 1; i < 48; ++i)
  {
    evaluator.add_inplace(var, temp_var[i]);
  }
  vector<PhantomCiphertext>().swap(temp_var);
  evaluator.relinearize_inplace(var, relin_keys);
  evaluator.rescale_to_next_inplace(var);

  vector<double> ecd_inv_n2(slot_count, 0);
  for (size_t i = 0; i < slot_count; ++i)
  {
    if (bias_vec[i] == 1)
    {
      ecd_inv_n2[i] = 1 / (768.0 * 768.0);
    }
  }
  PhantomPlaintext inv_d;
  encoder.encode(ecd_inv_n2, var.params_id(), var.scale(), inv_d);
  evaluator.multiply_plain_inplace(var, inv_d);
  evaluator.rescale_to_next_inplace(var);

  PhantomCiphertext inv_sqrt_var = invert_sqrt(var, 4, 2, context, relin_keys);

  {   // DIAGNOSTIC: invert_sqrt is a minimax polynomial valid only on a bounded interval. If var leaves that
      // interval the result explodes, which would make the whole layer output numerically meaningless while
      // the benchmark still runs and times fine. Check inv_sqrt_var against 1/sqrt(var) on the active slots.
    PhantomPlaintext _pv, _pi; vector<double> _vv, _vi;
    decryptor.decrypt(var, _pv); encoder.decode(_pv, _vv);
    decryptor.decrypt(inv_sqrt_var, _pi); encoder.decode(_pi, _vi);
    double maxrel = 0.0, vmin = 1e300, vmax = -1e300; long n = 0;
    for (size_t ind = 0; ind < slot_count; ++ind)
    {
      if (bias_vec[ind] == 1)
      {
        double v = _vv[ind];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        if (v > 0)
        {
          double want = 1.0 / sqrt(v), got = _vi[ind];
          double rel = fabs(got - want) / (fabs(want) > 1e-12 ? fabs(want) : 1e-12);
          if (rel > maxrel) maxrel = rel;
        }
        ++n;
      }
    }
    cout.precision(10);
    cout << "INVSQRT-LN1| var range [" << vmin << ", " << vmax << "]  max_rel_err(inv_sqrt_var vs 1/sqrt(var))="
         << maxrel << "  n=" << n << endl;
  }

  // ---- pass 3: output[i] = gamma[i]/768 * (nx[i]-ave_x) * inv_sqrt_var + beta[i] ----
  vector<PhantomCiphertext> output(num_ct);
  evaluator.mod_switch_to_inplace(ave_x, inv_sqrt_var.params_id());

  for (int i = 0; i < num_ct; ++i)
  {
    output[i] = make_nx(i);
    evaluator.mod_switch_to_inplace(output[i], inv_sqrt_var.params_id());
    evaluator.sub_inplace(output[i], ave_x);
    evaluator.multiply_inplace(output[i], inv_sqrt_var);
    evaluator.relinearize_inplace(output[i], relin_keys);
    evaluator.rescale_to_next_inplace(output[i]);

    vector<double> ecd_gamma_n(slot_count, 0);
    for (size_t j = 0; j < slot_count; ++j)
    {
      if (bias_vec[j] == 1)
      {
        ecd_gamma_n[j] = gamma[i] / sqrt(768.0);
      }
    }
    PhantomPlaintext ecd_gamma;
    encoder.encode(ecd_gamma_n, output[i].params_id(), output[i].scale(), ecd_gamma);
    evaluator.multiply_plain_inplace(output[i], ecd_gamma);
    evaluator.rescale_to_next_inplace(output[i]);

    vector<double> ecd_betai(slot_count, 0);
    for (size_t j = 0; j < slot_count; ++j)
    {
      if (bias_vec[j] == 1)
      {
        ecd_betai[j] = beta[i];
      }
    }
    PhantomPlaintext ecd_beta;
    encoder.encode(ecd_betai, output[i].params_id(), output[i].scale(), ecd_beta);
    evaluator.add_plain_inplace(output[i], ecd_beta);
  }

  return output;
}
