/* Matched exact circulant benchmark; compile beside the frozen production file.
 * No production kernel is edited. Additional template instantiations are
 * benchmark configurations, not claims about the public dispatch policy.
 */
#include "perm_modp_cuda.cu"
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct Buffer {
    T *p = nullptr;
    explicit Buffer(size_t count) { checked(cudaMalloc(&p, count * sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
    void close() { if (p) { checked(cudaFree(p)); p = nullptr; } }
};
struct Events {
    cudaEvent_t start = nullptr, stop = nullptr;
    Events() { checked(cudaEventCreate(&start)); checked(cudaEventCreate(&stop)); }
    ~Events() { if (start) cudaEventDestroy(start); if (stop) cudaEventDestroy(stop); }
};
struct Measurement {
    uint64_t residue, states, tasks, device_bytes, host_bytes;
    int depth, blocks, threads;
    double wall_ms, setup_ms, device_interval_ms, finalize_ms;
};

template<int M> static Measurement fixed_sign(const std::vector<uint32_t>& base,
                                               uint32_t p) {
    const auto started = Clock::now();
    const uint64_t states = UINT64_C(1) << (M - 1);
    const uint64_t outputs = M >= 28 ? UINT64_C(1) << 18
        : M >= 20 ? UINT64_C(1) << 14 : states;
    const uint64_t chunk = (states + outputs - 1) / outputs;
    const int threads = 256, blocks = (outputs + threads - 1) / threads;
    std::vector<uint32_t> matrix(M * M);
    for (int row = 0; row < M; ++row)
        for (int column = 0; column < M; ++column)
            matrix[row * M + column] = base[(column - row + M) % M];
    Buffer<uint32_t> d_matrix(matrix.size());
    Buffer<uint64_t> partial(outputs), reduction(outputs);
    checked(cudaMemcpy(d_matrix.p, matrix.data(), matrix.size() * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
    Events events;
    const auto setup_done = Clock::now();
    checked(cudaEventRecord(events.start));
    permanent_glynn_modp_pm32_kernel<M><<<blocks, threads>>>(
        d_matrix.p, p, (uint32_t)((UINT64_C(1) << 32) - p), states,
        chunk, outputs, partial.p);
    checked(cudaGetLastError());
    // Preserve the production host launch/synchronization policy.
    checked(cudaDeviceSynchronize());
    uint64_t remaining = outputs;
    uint64_t *input = partial.p, *output = reduction.p;
    while (remaining > 1) {
        const uint64_t count = (remaining + threads * 2 - 1) / (threads * 2);
        reduce_sum_modp<<<count, threads, threads * sizeof(uint64_t)>>>(
            input, output, remaining, (uint64_t)p);
        checked(cudaGetLastError());
        checked(cudaDeviceSynchronize());
        remaining = count;
        std::swap(input, output);
    }
    checked(cudaEventRecord(events.stop));
    checked(cudaEventSynchronize(events.stop));
    float device_ms = 0;
    checked(cudaEventElapsedTime(&device_ms, events.start, events.stop));
    const auto device_done = Clock::now();
    uint64_t raw = 0;
    checked(cudaMemcpy(&raw, input, sizeof(raw), cudaMemcpyDeviceToHost));
    const uint64_t result = mul_mod_host(raw, pow_mod_host((p >> 1) + 1,
                                                        M - 1, p), p);
    d_matrix.close(); partial.close(); reduction.close();
    const auto ended = Clock::now();
    return {result, states, 0, M * M * sizeof(uint32_t) + 2 * outputs * sizeof(uint64_t),
            M * M * sizeof(uint32_t), -1, blocks, threads,
            elapsed_ms(started, ended), elapsed_ms(started, setup_done),
            device_ms, elapsed_ms(device_done, ended)};
}

template<int M, int DEPTH> static Measurement necklace(
        const std::vector<uint32_t>& base, uint32_t p, int multiprocessors) {
    const auto started = Clock::now();
    uint32_t task_count = 0;
    NecklacePrefixTask *raw_tasks = build_necklace_frontier(M, DEPTH, &task_count);
    if (!raw_tasks) throw std::runtime_error("necklace frontier allocation failed");
    std::unique_ptr<NecklacePrefixTask, decltype(&free)> tasks(raw_tasks, &free);
    uint64_t base_sum = 0;
    for (auto entry : base) base_sum += entry;
    Buffer<NecklacePrefixTask> d_tasks(task_count);
    Buffer<uint32_t> d_base(M), queue(1);
    Buffer<unsigned long long> sum(1);
    checked(cudaMemcpy(d_tasks.p, tasks.get(), task_count * sizeof(NecklacePrefixTask),
                       cudaMemcpyHostToDevice));
    checked(cudaMemcpy(d_base.p, base.data(), M * sizeof(uint32_t), cudaMemcpyHostToDevice));
    checked(cudaMemset(queue.p, 0, sizeof(uint32_t)));
    checked(cudaMemset(sum.p, 0, sizeof(unsigned long long)));
    const int threads = 256, blocks = multiprocessors * 4;
    if (blocks <= 0 || (uint64_t)blocks > UINT64_MAX / (uint64_t)p)
        throw std::runtime_error("invalid necklace grid or raw-sum overflow bound");
    Events events;
    const auto setup_done = Clock::now();
    checked(cudaEventRecord(events.start));
    schur_necklace_pm32_kernel<M, DEPTH><<<blocks, threads, threads * sizeof(uint32_t)>>>(
        d_tasks.p, task_count, d_base.p, base_sum % p, p,
        (uint32_t)((UINT64_C(1) << 32) - p), queue.p, sum.p);
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
    checked(cudaEventRecord(events.stop));
    checked(cudaEventSynchronize(events.stop));
    float device_ms = 0;
    checked(cudaEventElapsedTime(&device_ms, events.start, events.stop));
    const auto device_done = Clock::now();
    unsigned long long raw = 0;
    checked(cudaMemcpy(&raw, sum.p, sizeof(raw), cudaMemcpyDeviceToHost));
    const uint64_t result = mul_mod_host(raw % p, pow_mod_host((p >> 1) + 1, M, p), p);
    d_tasks.close(); d_base.close(); queue.close(); sum.close();
    const auto ended = Clock::now();
    return {result, UINT64_C(1) << M, task_count,
            task_count * sizeof(NecklacePrefixTask) + M * sizeof(uint32_t) + 12,
            task_count * sizeof(NecklacePrefixTask), DEPTH, blocks, threads,
            elapsed_ms(started, ended), elapsed_ms(started, setup_done),
            device_ms, elapsed_ms(device_done, ended)};
}

template<int M, int D> static Measurement dispatch_method(
        const std::string& method, const std::vector<uint32_t>& base,
        uint32_t p, int sms) {
    if (method == "fixed_sign") return fixed_sign<M>(base, p);
    if (method == "necklace") return necklace<M, D>(base, p, sms);
    throw std::runtime_error("unknown method");
}
static Measurement dispatch(int m, const std::string& method,
                            const std::vector<uint32_t>& base, uint32_t p, int sms) {
#define CASE(M,D) case M: return dispatch_method<M,D>(method, base, p, sms)
    switch (m) {
        CASE(4,4); CASE(6,6); CASE(10,10); CASE(12,12); CASE(16,16);
        CASE(18,18); CASE(22,22); CASE(28,24); CASE(30,24); CASE(36,24);
        CASE(40,24); CASE(46,28);
    }
#undef CASE
    throw std::runtime_error("unsupported benchmark size");
}

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: wrapper INPUT.txt SCHEDULE.txt");
        std::ifstream input(argv[1]), schedule(argv[2]);
        int m; uint64_t prime; int64_t expected;
        if (!input || !schedule) throw std::runtime_error("cannot open input or schedule");
        if (!(input >> m >> prime >> expected) || m < 2 || m >= 64 || !(prime & 1) ||
            prime > UINT32_MAX || (UINT64_C(1) << 32) - prime > UINT16_MAX)
            throw std::runtime_error("invalid input header");
        std::vector<uint32_t> base(m);
        for (auto& entry : base) {
            uint64_t value;
            if (!(input >> value) || value >= prime) throw std::runtime_error("invalid base");
            entry = value;
        }
        std::string extra;
        if (input >> extra) throw std::runtime_error("trailing input");
        checked(cudaFree(nullptr)); // Initialize context outside timed calls.
        int device = 0, driver = 0, runtime = 0;
        checked(cudaGetDevice(&device));
        cudaDeviceProp props;
        checked(cudaGetDeviceProperties(&props, device));
        checked(cudaDriverGetVersion(&driver)); checked(cudaRuntimeGetVersion(&runtime));
        char uuid[33];
        for (int i = 0; i < 16; ++i)
            snprintf(uuid + 2*i, 3, "%02x", (unsigned char)props.uuid.bytes[i]);
        // CUDA device names supplied by the runtime are escaped before JSON output.
        std::string name;
        for (const char *c = props.name; *c; ++c) {
            if (*c == '"' || *c == '\\') name += '\\';
            name += *c;
        }
        std::cout << "{\"type\":\"device\",\"name\":\"" << name
                  << "\",\"major\":" << props.major << ",\"minor\":" << props.minor
                  << ",\"multiprocessors\":" << props.multiProcessorCount
                  << ",\"total_global_bytes\":" << props.totalGlobalMem
                  << ",\"uuid_hex\":\"" << uuid << "\""
                  << ",\"driver\":" << driver << ",\"runtime\":" << runtime << "}" << std::endl;
        int ordinal, warmup, iteration;
        std::string method;
        std::cout.precision(17);
        while (schedule >> ordinal >> method >> warmup >> iteration) {
            const auto call_start = Clock::now();
            auto r = dispatch(m, method, base, (uint32_t)prime, props.multiProcessorCount);
            r.wall_ms = elapsed_ms(call_start, Clock::now());
            std::cout << "{\"type\":\"sample\",\"ordinal\":" << ordinal
                      << ",\"method\":\"" << method << "\",\"warmup\":" << (warmup ? "true" : "false")
                      << ",\"iteration\":" << iteration << ",\"residue\":" << r.residue
                      << ",\"wall_ms\":" << r.wall_ms << ",\"setup_ms\":" << r.setup_ms
                      << ",\"device_interval_ms\":" << r.device_interval_ms
                      << ",\"finalize_ms\":" << r.finalize_ms
                      << ",\"represented_sign_states\":" << r.states << ",\"frontier_tasks\":" << r.tasks
                      << ",\"depth\":" << r.depth << ",\"blocks\":" << r.blocks
                      << ",\"threads\":" << r.threads << ",\"explicit_device_bytes\":" << r.device_bytes
                      << ",\"host_payload_bytes\":" << r.host_bytes << "}" << std::endl;
            if (expected >= 0 && r.residue != (uint64_t)expected)
                throw std::runtime_error("EXACT_MISMATCH (failing sample retained)");
            expected = r.residue;
        }
        if (!schedule.eof()) throw std::runtime_error("invalid schedule");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "schur benchmark failed: " << error.what() << std::endl;
        return 1;
    }
}
