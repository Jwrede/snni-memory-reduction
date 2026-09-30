struct HostCipher {
    std::vector<uint64_t> host_data;
    std::size_t sz = 0, cms = 0, pmd = 0;
    std::size_t chain_index = 0;
    double scale = 0.0;
    std::uint64_t corr = 0;
    bool ntt = false;
    std::size_t nsd = 0;
    // s22: optional disk backing. If disk_fd >= 0 and host_data is empty, the payload lives in a bank file
    // at [disk_off, disk_off + payload bytes) and is pread() on demand by reload_cipher_from_host.
    int disk_fd = -1;
    std::uint64_t disk_off = 0;
};

// s22: a write-through disk bank. offload_cipher_to_disk() appends the payload to the bank file immediately
// and keeps only metadata in RAM, so a parked bank costs O(1) host memory instead of ~16 GiB. Reads are
// per-entry pread() -- measured free next to the bootstraps (s20: 64 GB of /scratch I/O was invisible).
// Env-gated like the spill helpers: without MOAI_SPILL_DIR it falls back to in-RAM parking.
struct DiskBank {
    int fd = -1;
    std::uint64_t off = 0;
    std::string path;
    bool open_bank(const char *tag)
    {
        const char *dir = std::getenv("MOAI_SPILL_DIR");
        if (!dir) return false;
        path = std::string(dir) + "/" + tag + ".bank";
        fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        return fd >= 0;
    }
    void close_bank()
    {
        if (fd >= 0) { ::close(fd); ::remove(path.c_str()); fd = -1; }
    }
};

static inline HostCipher offload_cipher_to_disk(PhantomCiphertext &ct, DiskBank &bank)
{
    HostCipher h;
    cudaDeviceSynchronize();
    h.chain_index = ct.chain_index();
    h.pmd = ct.poly_modulus_degree();
    h.cms = ct.coeff_modulus_size();
    h.sz  = ct.size();
    h.scale = ct.scale();
    h.corr  = ct.correction_factor();
    h.ntt   = ct.is_ntt_form();
    h.nsd   = ct.GetNoiseScaleDeg();
    std::size_t count = h.sz * h.cms * h.pmd;
    if (bank.fd >= 0)
    {
        static thread_local std::vector<uint64_t> staging;
        staging.resize(count);
        if (ct.data() != nullptr && count > 0)
            cudaMemcpy(staging.data(), ct.data(), count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::size_t bytes = count * sizeof(uint64_t);
        std::size_t done = 0;
        while (done < bytes)
        {
            ssize_t w = ::pwrite(bank.fd, reinterpret_cast<const char*>(staging.data()) + done,
                                 bytes - done, bank.off + done);
            if (w <= 0) { std::cout << "SPILL| pwrite failed, keeping in RAM" << std::endl;
                          h.host_data.assign(staging.begin(), staging.end()); break; }
            done += (std::size_t)w;
        }
        if (done >= bytes) { h.disk_fd = bank.fd; h.disk_off = bank.off; bank.off += bytes; }
    }
    else
    {
        h.host_data.resize(count);
        if (ct.data() != nullptr && count > 0)
            cudaMemcpy(h.host_data.data(), ct.data(), count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
    ct = PhantomCiphertext();
    return h;
}

// w-ladder: ONE process-global bank behind the ordinary host-offload helper. When MOAI_SPILL_DIR is set,
// every park site in the whole codebase writes its payload straight to disk and keeps only metadata in RAM,
// so VRAM and host RAM are minimised SIMULTANEOUSLY instead of trading one for the other. No call site
// changes; without the env var the behaviour is exactly the old in-RAM parking.
static inline DiskBank &global_park_bank()
{
    static DiskBank bank;
    static bool tried = false;
    if (!tried) { tried = true; bank.open_bank("wpark"); }
    return bank;
}

static inline HostCipher offload_cipher_to_host(PhantomCiphertext &ct)
{
    DiskBank &bank = global_park_bank();
    if (bank.fd >= 0) return offload_cipher_to_disk(ct, bank);
    // CORRECTNESS: Phantom creates its streams with cudaStreamNonBlocking (cuda_wrapper.cuh), so they are
    // NOT ordered against the legacy default stream. A plain cudaMemcpy below would therefore NOT wait for
    // the kernels that produced `ct` and could copy stale/partial data. Sync the device BEFORE reading it.
    cudaDeviceSynchronize();
    HostCipher h;
    h.chain_index = ct.chain_index();
    h.pmd = ct.poly_modulus_degree();
    h.cms = ct.coeff_modulus_size();
    h.sz  = ct.size();
    h.scale = ct.scale();
    h.corr  = ct.correction_factor();
    h.ntt   = ct.is_ntt_form();
    h.nsd   = ct.GetNoiseScaleDeg();
    std::size_t count = h.sz * h.cms * h.pmd;
    h.host_data.resize(count);
    if (ct.data() != nullptr && count > 0)
        cudaMemcpy(h.host_data.data(), ct.data(), count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    ct = PhantomCiphertext();   // free the device buffer
    return h;
}

// Non-destructive variant: copy a ciphertext to the host but LEAVE the device buffer intact. Used only by the
// in-run A/B verification, which must run the device path and the host-streaming path on the SAME ciphertexts.
static inline HostCipher copy_cipher_to_host(const PhantomCiphertext &ct)
{
    cudaDeviceSynchronize();
    HostCipher h;
    h.chain_index = ct.chain_index();
    h.pmd = ct.poly_modulus_degree();
    h.cms = ct.coeff_modulus_size();
    h.sz  = ct.size();
    h.scale = ct.scale();
    h.corr  = ct.correction_factor();
    h.ntt   = ct.is_ntt_form();
    h.nsd   = ct.GetNoiseScaleDeg();
    std::size_t count = h.sz * h.cms * h.pmd;
    h.host_data.resize(count);
    if (ct.data() != nullptr && count > 0)
        cudaMemcpy(h.host_data.data(), ct.data(), count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    return h;
}

static inline void reload_cipher_from_host(PhantomCiphertext &ct, const HostCipher &h,
        phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream)
{
    ct.set_chain_index(h.chain_index);
    ct.set_poly_modulus_degree(h.pmd);
    ct.set_coeff_modulus_size(h.cms);
    ct.set_scale(h.scale);
    ct.set_correction_factor(h.corr);
    ct.set_ntt_form(h.ntt);
    ct.SetNoiseScaleDeg(h.nsd);
    ct.reinit_like(h.sz, h.cms, h.pmd, stream.get_stream());
    std::size_t count = h.sz * h.cms * h.pmd;
    // s22: disk-backed entry -> pread the payload into a reusable staging buffer, then H2D from there.
    if (count > 0 && h.host_data.empty() && h.disk_fd >= 0)
    {
        static thread_local std::vector<uint64_t> staging;
        staging.resize(count);
        std::size_t bytes = count * sizeof(uint64_t), done = 0;
        while (done < bytes)
        {
            ssize_t r = ::pread(h.disk_fd, reinterpret_cast<char*>(staging.data()) + done,
                                bytes - done, h.disk_off + done);
            if (r <= 0) { std::cout << "SPILL| pread failed at off " << h.disk_off << std::endl; break; }
            done += (std::size_t)r;
        }
        cudaMemcpyAsync(ct.data(), staging.data(), bytes, cudaMemcpyHostToDevice, stream.get_stream());
        cudaStreamSynchronize(stream.get_stream());   // staging is reused; must complete before return
        return;
    }
    if (count > 0)
        cudaMemcpyAsync(ct.data(), h.host_data.data(), count * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, stream.get_stream());
    // CORRECTNESS: the H2D copy above is async on ONE (non-blocking) stream, but the caller may consume this
    // ciphertext with evaluator ops queued on a DIFFERENT non-blocking stream, which would not be ordered
    // against it. Make the reload synchronous so the data is guaranteed present before any use.
    cudaStreamSynchronize(stream.get_stream());
}

// ---- h1: disk spill of parked HostCipher banks -------------------------------------------------------------
// A parked residual bank sits idle in host RAM for many minutes. If MOAI_SPILL_DIR is set, these helpers move
// the payloads to a file on that (disk-backed) path and release the host memory, and read them back before
// use -- the same host->disk escalation the CPU waterfall used. With the env unset they are no-ops.
static inline void spill_bank_to_disk(std::vector<HostCipher> &v, const char *tag)
{
    const char *dir = std::getenv("MOAI_SPILL_DIR");
    if (!dir) return;
    std::string path = std::string(dir) + "/" + tag + ".bin";
    std::ofstream f(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!f) { std::cout << "SPILL| cannot open " << path << " -- keeping bank in RAM" << std::endl; return; }
    for (auto &h : v)
    {
        std::uint64_t n = h.host_data.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        if (n) f.write(reinterpret_cast<const char*>(h.host_data.data()), n * sizeof(std::uint64_t));
        std::vector<std::uint64_t>().swap(h.host_data);   // release host RAM; metadata stays
    }
    f.close();
    std::cout << "SPILL| " << tag << " -> disk" << std::endl;
}

static inline void unspill_bank_from_disk(std::vector<HostCipher> &v, const char *tag)
{
    const char *dir = std::getenv("MOAI_SPILL_DIR");
    if (!dir) return;
    std::string path = std::string(dir) + "/" + tag + ".bin";
    std::ifstream f(path.c_str(), std::ios::binary);
    if (!f) { std::cout << "SPILL| missing " << path << std::endl; return; }
    for (auto &h : v)
    {
        std::uint64_t n = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        h.host_data.resize(n);
        if (n) f.read(reinterpret_cast<char*>(h.host_data.data()), n * sizeof(std::uint64_t));
    }
    f.close();
    std::remove(path.c_str());
    std::cout << "SPILL| " << tag << " <- disk" << std::endl;
}

static inline std::vector<HostCipher> offload_vector_to_host(std::vector<PhantomCiphertext> &v)
{
    std::vector<HostCipher> out; out.reserve(v.size());
    for (auto &ct : v) out.push_back(offload_cipher_to_host(ct));
    cudaDeviceSynchronize();   // ensure all D2H copies complete before the device buffers are reused
    return out;
}

static inline void reload_vector_from_host(std::vector<PhantomCiphertext> &v, const std::vector<HostCipher> &h)
{
    v.resize(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) reload_cipher_from_host(v[i], h[i]);
    cudaStreamSynchronize(phantom::util::global_variables::default_stream->get_stream());
}
