#!/usr/bin/env python3
"""Bounded CPU checks of Schur sign orbits and extracted production logic.

No CUDA library is loaded. The production generator and modular helpers are
compiled as ordinary C++ with CUDA qualifiers removed; this does not validate
CUDA compilation or GPU execution. Output directories must be fresh.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA256 = "34f0910dc68954f81ea3d18f9f94293670608ab6c70f7f3af59d006049a42440"
SEED = 20260905


def sha(data):
    return hashlib.sha256(data).hexdigest()


CPP_DRIVER = r'''
uint64_t rotate_ref(uint64_t x,int m,int r) {
    uint64_t y=0;
    for(int j=0;j<m;++j) y|=((x>>j)&1ULL)<<((j+r)%m);
    return y;
}
std::map<uint64_t,uint32_t> reference_orbits(int m) {
    uint64_t mask=(1ULL<<m)-1; std::map<uint64_t,uint32_t> out;
    for(uint64_t x=0;x<=mask;++x) {
        std::set<uint64_t> words;
        for(int r=0;r<m;++r) {words.insert(rotate_ref(x,m,r));words.insert(rotate_ref(mask^x,m,r));}
        auto better=[](uint64_t a,uint64_t b){int wa=__builtin_popcountll(a),wb=__builtin_popcountll(b);return wa!=wb?wa<wb:a<b;};
        uint64_t representative=*std::min_element(words.begin(),words.end(),better);
        out[representative]=words.size();
    }
    return out;
}
unsigned tested=0;uint64_t total_emitted=0;
template<int M,int D> void check() {
    auto expected=reference_orbits(M);std::map<uint64_t,uint32_t> actual;uint32_t count=0;
    auto *tasks=build_necklace_frontier(M,D,&count);assert(tasks&&count);
    for(uint32_t t=0;t<count;++t) {
        int position=D+1;uint64_t word=tasks[t].word;PackedNecklaceDfsState stack={0,0,0};stack.set(0,tasks[t].period);
        uint64_t emitted;uint32_t weight;
        const uint32_t prime=4294967291u;
        std::vector<uint32_t> base(M),sums(M);
        uint64_t initial=0,previous=0;
        for(int j=0;j<M;++j){base[j]=j%3==0?prime-1:uint32_t(j*j+1);initial+=base[j];}
        for(auto &v:sums)v=initial%prime;
        while(next_necklace_representative<M,D>(position,word,stack,emitted,weight)) {
            assert(actual.emplace(emitted,weight).second);
            uint64_t difference=previous^emitted;
            while(difference){int bit=__builtin_ctzll(difference);int column=M-1-bit;bool minus=(emitted>>bit)&1;
                for(int row=0;row<M;++row){uint32_t entry=base[(column-row+M)%M];uint32_t twice=add_mod_u32(entry,entry,prime);sums[row]=minus?sub_mod_u32(sums[row],twice,prime):add_mod_u32(sums[row],twice,prime);}
                difference&=difference-1;
            }
            previous=emitted;
            uint32_t product=1;uint64_t product_ref=1;
            for(int row=0;row<M;++row){int64_t direct=0;for(int j=0;j<M;++j)direct+=((emitted>>(M-1-j))&1)?-int64_t(base[(j-row+M)%M]):int64_t(base[(j-row+M)%M]);direct%=prime;if(direct<0)direct+=prime;assert(sums[row]==direct);
                product=mul_mod_pseudo_mersenne_u32(product,sums[row],prime,5);product_ref=(product_ref*uint64_t(direct))%prime;
            }
            assert(product==product_ref);
        }
    }
    free(tasks);assert(actual==expected);uint64_t total=0;
    for(auto entry:actual)total+=entry.second;
    assert(total==(1ULL<<M));++tested;total_emitted+=actual.size();
    if constexpr(D<M)check<M,D+1>();
}
int main(){check<2,0>();check<4,0>();check<6,0>();check<8,0>();check<10,0>();check<12,0>();printf("{\"generator_dimension_depth_cases\":%u,\"emitted_representatives_checked\":%llu,\"status\":\"PASS\"}\n",tested,(unsigned long long)total_emitted);}
'''


def extract_cpp(source):
    intervals = (
        ("struct NecklacePrefixTask", "static int is_prime_order_host"),
        ("struct PackedNecklaceDfsState", "/*\n * Each thread advances complete FKM prefix subtrees."),
        ("__device__ __forceinline__ uint32_t mul_mod_pseudo_mersenne_u32(",
         "/*\n * Each CUDA thread processes a contiguous block of Gray code states."),
    )
    parts = [source[source.index(start):source.index(stop, source.index(start))]
             for start, stop in intervals]
    header = "\n".join(f"#include <{name}>" for name in
                       ("cstdio", "cstdint", "cstdlib", "cassert", "map", "set", "algorithm", "vector"))
    header += "\nstatic int __popcll(uint64_t x){return __builtin_popcountll(x);}\n"
    return header + "\n".join(parts).replace("__device__ __forceinline__", "inline") + CPP_DRIVER


def permutations_permanent(matrix):
    m = len(matrix)
    return sum(math.prod(matrix[i][sigma[i]] for i in range(m))
               for sigma in itertools.permutations(range(m)))


def rotations(word, m):
    mask = (1 << m) - 1
    return {((word << r) & mask) | (word >> (m - r)) if r else word for r in range(m)}


def representatives(m):
    mask, remaining, out = (1 << m) - 1, set(range(1 << m)), []
    while remaining:
        word = min(remaining)
        orbit = rotations(word, m) | rotations(mask ^ word, m)
        remaining -= orbit
        representative = min(orbit, key=lambda value: (value.bit_count(), value))
        period = len(rotations(representative, m))
        expected = period if mask ^ representative in rotations(representative, m) else 2 * period
        assert len(orbit) == expected
        out.append((representative, expected))
    return out


def summand(base, word):
    m = len(base)
    return (-1) ** word.bit_count() * math.prod(
        sum(base[(j-i) % m] * (1 - 2 * ((word >> j) & 1)) for j in range(m))
        for i in range(m))


def independent_checks():
    rng, matrices, words = random.Random(SEED), 0, 0
    for m in range(1, 9):
        reps = representatives(m)
        assert sum(weight for _, weight in reps) == 1 << m
        for _ in range(5):
            base = [rng.randint(-3, 3) for _ in range(m)]
            matrix = [[base[(j-i) % m] for j in range(m)] for i in range(m)]
            assert sum(weight * summand(base, word) for word, weight in reps) == (
                1 << m) * permutations_permanent(matrix)
            matrices += 1
            for word in range(1 << m):
                value = summand(base, word)
                assert value == summand(base, ((1 << m) - 1) ^ word)
                assert value == summand(base, next(iter(rotations(word, m) - {word}), word))
                words += 1
    counts = {}
    for m in (40, 46):
        fixed = sum(1 << math.gcd(m, r) for r in range(m))
        fixed += sum(1 << math.gcd(m, r) for r in range(m) if (m // math.gcd(m, r)) % 2 == 0)
        assert fixed % (2 * m) == 0
        counts[str(m)] = fixed // (2 * m)
    assert counts == {"40": 13743921632, "46": 764877836468}
    return {"status": "PASS", "seed": SEED, "orders": list(range(1, 9)),
            "random_signed_circulant_permutation_checks": matrices,
            "invariance_word_cases": words, "burnside_representative_counts": counts}


def execute(command, output, stem, timeout):
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    paths = (output / f"{stem}.stdout.log", output / f"{stem}.stderr.log")
    paths[0].write_bytes(stdout)
    paths[1].write_bytes(stderr)
    record = {"command": command, "exit_code": process.returncode, "timed_out": timed_out,
              "seconds": time.monotonic()-started,
              "logs": {path.name: sha(path.read_bytes()) for path in paths}}
    if timed_out or process.returncode:
        raise RuntimeError(json.dumps(record))
    return record, stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "perm_modp_cuda.cu")
    parser.add_argument("--expected-source-sha256", default=SOURCE_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    if not __debug__:
        parser.error("assertions must be enabled; do not use python -O")
    source = args.source.read_bytes()
    if sha(source) != args.expected_source_sha256:
        parser.error("production source hash differs from the requested snapshot")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "schur-method-cpu-audit-v1", "status": "RUNNING",
              "started_utc": datetime.now(timezone.utc).isoformat(), "invocation": sys.argv,
              "source_path": str(args.source.resolve()), "source_sha256": sha(source),
              "checker_sha256": sha(Path(__file__).read_bytes()), "gpu_initialized": False,
              "scope": "CPU mathematics and extracted production generator/arithmetic; no CUDA compilation or GPU execution"}
    try:
        report["independent_checks"] = independent_checks()
        cpp, binary = output / "generator-row-check.cpp", output / "generator-row-check"
        cpp.write_text(extract_cpp(source.decode()))
        report["extracted_cpp_sha256"] = sha(cpp.read_bytes())
        compiler = shutil.which(args.compiler)
        if compiler is None:
            raise RuntimeError("C++ compiler unavailable")
        report["compiler_sha256"] = sha(Path(compiler).read_bytes())
        version, version_text = execute([compiler, "--version"], output, "compiler-version", 10)
        report["compiler_version"] = version_text.decode().strip()
        report["compiler_version_command"] = version
        report["compile"], _ = execute([compiler, "-O2", "-std=c++17", str(cpp), "-o", str(binary)],
                                      output, "compile", 40)
        report["binary_sha256"] = sha(binary.read_bytes())
        report["execution"], stdout = execute([str(binary)], output, "execute", 20)
        report["production_checks"] = json.loads(stdout)
        assert report["production_checks"] == {"generator_dimension_depth_cases": 48,
                                               "emitted_representatives_checked": 3218, "status": "PASS"}
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "production_checks": report["production_checks"],
                      "independent_checks": report["independent_checks"]}))


if __name__ == "__main__":
    main()
