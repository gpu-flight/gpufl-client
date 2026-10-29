; Two kernels written the way Tensile writes the GEMM kernels of rocBLAS and
; hipBLASLt: global function symbols without a .size, and local labels among
; their instructions. Built by ROCm 7.2.0's clang and lld:
;   clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 -c \
;     pc_sampling_unsized.s -o pc_sampling_unsized.o
;   ld.lld -shared pc_sampling_unsized.o -o pc_sampling_unsized.gfx942.co
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.text
.globl first_kernel
.p2align 8
.type first_kernel,@function
first_kernel:
label_first_start:
  s_mov_b32 s0, 0
label_first_loop:
  s_add_u32 s0, s0, 1
  s_cmp_lt_u32 s0, 4
  s_cbranch_scc1 label_first_loop
  s_endpgm

.globl second_kernel
.p2align 8
.type second_kernel,@function
second_kernel:
label_second_start:
  v_mov_b32 v0, 0
  s_endpgm

.rodata
.p2align 6
.amdhsa_kernel first_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
  .amdhsa_accum_offset 4
.end_amdhsa_kernel
.p2align 6
.amdhsa_kernel second_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
  .amdhsa_accum_offset 4
.end_amdhsa_kernel
