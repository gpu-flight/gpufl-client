// Itanium-ABI demangler tests.
//
// The mangled inputs here are REAL kernel names captured from a
// Windows-agent CUDA trace where the OS-bundled `__cxa_demangle`
// wasn't available. They're the regression cases the Layer 2 fix
// targets: without the demangler, the frontend's op-catalog regex
// can't classify them as GEMM / ELEMENTWISE / etc.
//
// We don't insist on byte-exact equality with libstdc++'s
// __cxa_demangle output (different libraries format spaces and
// commas differently). Instead we assert on the substrings the
// downstream regex catalog cares about - that's the contract.

#include <gtest/gtest.h>

#include "gpufl/core/itanium_demangle.hpp"
#include "gpufl/core/stack_trace.hpp"

using gpufl::core::DemangleFunctionKey;
using gpufl::core::DemangleItaniumName;

// ── Pass-through cases ───────────────────────────────────────────────

TEST(ItaniumDemangle, PassesThroughEmpty) {
    EXPECT_EQ(DemangleItaniumName(""), "");
}

TEST(ItaniumDemangle, PassesThroughNonMangled) {
    // Plain CUDA kernel names (cuBLAS-style architecture-prefixed
    // SASS kernels) are not Itanium-mangled and must pass through
    // verbatim - the catalog matches them as-is.
    EXPECT_EQ(DemangleItaniumName("ampere_sgemm_64x32_nn"),
              "ampere_sgemm_64x32_nn");
    EXPECT_EQ(DemangleItaniumName("vectorAdd"), "vectorAdd");
}

TEST(ItaniumDemangle, PassesThroughMSVCMangled) {
    // MSVC-style names start with '?' - not our format; pass through.
    EXPECT_EQ(DemangleItaniumName("?foo@bar@@YAHXZ"),
              "?foo@bar@@YAHXZ");
}

// ── Simple Itanium mangled names ─────────────────────────────────────

TEST(ItaniumDemangle, SimpleSourceName) {
    // _Z16branchByWarpQuadPfPKfi
    //   _Z 16 branchByWarpQuad P f P K f i
    //   → branchByWarpQuad(float*, float const*, int)
    auto out = DemangleItaniumName("_Z16branchByWarpQuadPfPKfi");
    EXPECT_NE(out.find("branchByWarpQuad"), std::string::npos);
    EXPECT_NE(out.find("float*"), std::string::npos);
    EXPECT_NE(out.find("const"), std::string::npos);
}

TEST(ItaniumDemangle, NestedNameSimple) {
    // _ZN2at6native10some_thingEv  →  at::native::some_thing()
    auto out = DemangleItaniumName("_ZN2at6native10some_thingEv");
    EXPECT_NE(out.find("at::native::some_thing"), std::string::npos);
}

// ── Real CUDA kernel regressions (from a Windows trace) ──────────────

TEST(ItaniumDemangle, CutlassSgemmKernel_RealCapture) {
    // Captured verbatim from a 5060/Windows trace. The Linux/Ubuntu
    // run of the same code rendered this as
    //   void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(...)
    // which the catalog regex
    //   ^(void\s+)?cutlass(_\d+)?(::Kernel\d?)?.*(gemm|Gemm|sgemm).*
    // matches → GEMM chip. We assert the same bits the regex needs:
    // "cutlass::Kernel2", "<", and "sgemm" all appear in the result.
    const char* mangled =
        "_ZN7cutlass7Kernel2I43cutlass_80_simt_sgemm_256x128_8x4_nn_align1EEvNT_6ParamsE";
    auto out = DemangleItaniumName(mangled);

    EXPECT_NE(out.find("cutlass::Kernel2"), std::string::npos)
        << "missing namespace::class - got: " << out;
    EXPECT_NE(out.find("<"), std::string::npos)
        << "missing template-arg open - got: " << out;
    EXPECT_NE(out.find("sgemm"), std::string::npos)
        << "missing sgemm substring (catalog regex hinges on this) - got: " << out;
    EXPECT_NE(out.find("cutlass_80_simt_sgemm_256x128_8x4_nn_align1"), std::string::npos)
        << "template-arg name should propagate through - got: " << out;
}

TEST(ItaniumDemangle, AtenElementwiseKernel_RealCapture) {
    // Captured verbatim from a 5060/Windows trace. This is the
    // PyTorch ATen "distribution_elementwise_kernel" (random-init
    // launcher). The frontend regex
    //   ^(void\s+)?at::native::(vectorized_)?elementwise_kernel.*
    // requires the demangled output to start with `at::native::` and
    // contain `elementwise_kernel`. We assert both.
    const char* mangled =
        "_ZN2at6native54_GLOBAL__N__966ce9c1_21_DistributionNormal_cu_0c5b6e85"
        "43distribution_elementwise_kernelILi4ELi64EvEEv";
    auto out = DemangleItaniumName(mangled);

    EXPECT_NE(out.find("at::native"), std::string::npos)
        << "missing at::native namespace prefix - got: " << out;
    EXPECT_NE(out.find("distribution_elementwise_kernel"), std::string::npos)
        << "missing elementwise_kernel suffix - got: " << out;
}

TEST(ItaniumDemangle, AtenVectorizedElementwiseKernel_RealCapture) {
    // Captured verbatim from a 5060/Windows trace AFTER the first
    // round of fixes - this one was still showing up mangled because
    // its template-arg list contains `St5arrayI…E` (std::array<…>)
    // and the old parseSubstitution returned a bare "std" without
    // consuming the "5array" continuation, derailing the rest of the
    // parse. With the St-handling fix, this one demangles cleanly.
    const char* mangled =
        "_ZN2at6native29vectorized_elementwise_kernel"
        "ILi2ENS0_11FillFunctorIxEESt5arrayIPcLy1EEEEviT0_";
    auto out = DemangleItaniumName(mangled);

    EXPECT_NE(out.find("at::native::vectorized_elementwise_kernel"), std::string::npos)
        << "missing at::native::vectorized_elementwise_kernel prefix - got: " << out;
    // The frontend catalog regex
    //   ^(void\s+)?at::native::(vectorized_)?elementwise_kernel.*
    // needs to match this - confirmed if the prefix is present.
    EXPECT_NE(out.find("std::array"), std::string::npos)
        << "St5array should expand to std::array (regression for the St "
           "namespace-shortcut bug) - got: " << out;
}

// Additional real CUDA kernel regression: local ATen lambda closure.

TEST(ItaniumDemangle, AtenReduceKernelWithLambdaClosure_RealCapture) {
    // Captured from a PyTorch PC sampling/SASS run. The template args include
    // a local lambda closure type (`UlffE_`) inside sum_functor::operator().
    // Without local-name + closure parsing this fell back to the raw mangled
    // symbol and showed up unreadable in Source/SASS.
    const char* mangled =
        "_ZN2at6native13reduce_kernelILi128ELi4ENS0_8ReduceOpIfNS0_14func_wrapper_t"
        "IfZNS0_11sum_functorIfffEclERNS_14TensorIteratorEEUlffE_EEjfLi4ELi4EEEEEvT1_";
    auto out = DemangleItaniumName(mangled);

    EXPECT_NE(out, mangled) << "must not fall back to the raw mangled name";
    EXPECT_NE(out.find("at::native::reduce_kernel"), std::string::npos)
        << "missing reducer prefix - got: " << out;
    EXPECT_NE(out.find("ReduceOp"), std::string::npos)
        << "missing ReduceOp template arg - got: " << out;
    EXPECT_NE(out.find("sum_functor"), std::string::npos)
        << "missing functor context - got: " << out;
    EXPECT_NE(out.find("operator()"), std::string::npos)
        << "missing local operator() context - got: " << out;
    EXPECT_NE(out.find("lambda(float, float)"), std::string::npos)
        << "missing lambda closure signature - got: " << out;
}

// ── Failure modes - must always return a non-empty displayable string ─

TEST(ItaniumDemangle, MalformedReturnsOriginal) {
    // Truncated / nonsense after the _Z prefix → parser fails → we
    // hand the original mangled string back so the UI still has
    // SOMETHING to display.
    const char* bad = "_ZN7cutlass";  // truncated - no closing E
    EXPECT_EQ(DemangleItaniumName(bad), bad);
}

TEST(ItaniumDemangle, UnknownTemplateArgFallsBack) {
    // X / L / J template arguments use an expression sub-grammar we
    // don't fully implement. When the parser hits one and can't
    // make sense of the rest, it MUST still return a non-empty
    // displayable string (the original mangled form is the safe
    // fallback). This test guards the contract - never return
    // garbage, never return empty.
    const char* withLiteral = "_ZN3foo3barIXLi42EEEvv";  // synthetic, off-by-one E
    auto out = DemangleItaniumName(withLiteral);
    EXPECT_FALSE(out.empty())
        << "must always return something displayable, even on parse failure";
    // Doesn't matter whether it's the original mangled form or a
    // partial demangle - we just guarantee the caller can show it.
}

// ── DemangleFunctionKey: "name@source" → "demangled@source" ──────────
//
// SASS / PC sampling intern their per-symbol identity as
// `mangled_name + "@" + source_file`. DemangleFunctionKey demangles ONLY the
// name half so the result joins Trace's already-demangled kernel_dict for the
// multi-pass merge, while the source path survives verbatim. Delegates to
// DemangleName, so the same substring-not-byte-exact contract applies.

TEST(DemangleFunctionKey, DemanglesNameKeepsSourceTail) {
    const auto out =
        DemangleFunctionKey("_Z16branchByWarpQuadPfPKfi@/kernels/foo.cu");
    EXPECT_NE(out.find("branchByWarpQuad"), std::string::npos)
        << "name half must be demangled - got: " << out;
    EXPECT_NE(out.find("@/kernels/foo.cu"), std::string::npos)
        << "the @source tail must survive verbatim - got: " << out;
}

TEST(DemangleFunctionKey, NonMangledNamePassesThrough) {
    // Already-demangled / plain names + their source survive byte-for-byte.
    EXPECT_EQ(DemangleFunctionKey("vectorAdd@/kernels/foo.cu"),
              "vectorAdd@/kernels/foo.cu");
}

TEST(DemangleFunctionKey, NoAtSignDemanglesWholeKey) {
    const auto out = DemangleFunctionKey("_Z16branchByWarpQuadPfPKfi");
    EXPECT_NE(out.find("branchByWarpQuad"), std::string::npos) << out;
    EXPECT_EQ(out.find('@'), std::string::npos)
        << "must not invent a source tail when the key has none - got: " << out;
}

TEST(DemangleFunctionKey, EmptySourceKeepsTrailingAt) {
    const auto out = DemangleFunctionKey("_Z16branchByWarpQuadPfPKfi@");
    EXPECT_NE(out.find("branchByWarpQuad"), std::string::npos) << out;
    ASSERT_FALSE(out.empty());
    EXPECT_EQ(out.back(), '@')
        << "empty source → trailing '@' preserved - got: " << out;
}

TEST(DemangleFunctionKey, EmptyNameKeepsSource) {
    // First '@' is at index 0 → empty name half, source survives.
    EXPECT_EQ(DemangleFunctionKey("@/kernels/foo.cu"), "@/kernels/foo.cu");
}
