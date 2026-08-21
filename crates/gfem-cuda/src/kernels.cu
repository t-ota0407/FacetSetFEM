// GenericFEM の CUDA カーネル（SPEC §4.4 / §8 M5）。
//
// # ここでいちばん大事なこと: CPU とビット単位で一致させる
//
// CLAUDE.md 設計原則 1 は「同一入力から CPU 逐次・CPU 並列・GPU のいずれでも
// ビット単位で同じ結果」を要求する。IEEE 754 の倍精度の加算・乗算・平方根は
// 正しく丸められるので、**演算の順序さえ揃えれば** ハードウェアが違っても
// 結果は 1 ビットも変わらない。順序を崩す要因は 2 つあり、両方を潰してある。
//
// 1. **FMA への縮約**。`a += b * c` を 1 命令の fused multiply-add にまとめると、
//    途中の丸めが 1 回減って CPU と値が変わる。Rust は勝手に FMA を作らないので、
//    GPU 側も作らせない。`__dmul_rn` / `__dadd_rn`（round-to-nearest の
//    乗算・加算そのもの）を明示的に書き、NVRTC にも `--fmad=false` を渡している
//
// 2. **総和の木の形**。ワープ内リダクションのような「速い」総和は
//    加算順序が CPU と違う。ここでは CPU 側 `solver::reduce` と**同じ規則**——
//    長さ 512 の連続ブロックをスレッド 1 本が先頭から順に足し、
//    そのブロック和の列に同じ規則を再帰的に適用する——を使う。
//    連続 512 要素を 1 スレッドが読むためメモリアクセスは合体しないが、
//    総和は CG の律速ではないので、速度より再現性を採る

extern "C" {

// y = A x（CSR）。行ごとに 1 スレッド。行内は列番号の昇順に足すので、
// CPU の `CsrMatrix::spmv` と同じ順序になる。
__global__ void gfem_spmv(
    const unsigned long long n,
    const unsigned long long* __restrict__ row_ptr,
    const unsigned int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long s = row_ptr[i];
    const unsigned long long e = row_ptr[i + 1];
    double acc = 0.0;
    for (unsigned long long k = s; k < e; ++k) {
        acc = __dadd_rn(acc, __dmul_rn(values[k], x[col_idx[k]]));
    }
    y[i] = acc;
}

// y += alpha * x
__global__ void gfem_axpy(
    const unsigned long long n,
    const double alpha,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __dadd_rn(y[i], __dmul_rn(alpha, x[i]));
}

// y = x + beta * y
__global__ void gfem_xpby(
    const unsigned long long n,
    const double* __restrict__ x,
    const double beta,
    double* __restrict__ y)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __dadd_rn(x[i], __dmul_rn(beta, y[i]));
}

// x *= alpha
__global__ void gfem_scale(
    const unsigned long long n,
    const double alpha,
    double* __restrict__ x)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __dmul_rn(x[i], alpha);
}

// y = a * x（成分ごと）
__global__ void gfem_multiply(
    const unsigned long long n,
    const double* __restrict__ a,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __dmul_rn(a[i], x[i]);
}

// x += delta
__global__ void gfem_shift(
    const unsigned long long n,
    const double delta,
    double* __restrict__ x)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __dadd_rn(x[i], delta);
}

// out[b] = Σ v[b*block .. min((b+1)*block, n)]（先頭から順に）
// CPU の `reduce::sum_block` と同じ。
__global__ void gfem_block_sum(
    const unsigned long long n,
    const unsigned long long block,
    const double* __restrict__ v,
    double* __restrict__ out)
{
    unsigned long long b = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const unsigned long long s = b * block;
    if (s >= n) return;
    unsigned long long e = s + block;
    if (e > n) e = n;
    double acc = 0.0;
    for (unsigned long long k = s; k < e; ++k) {
        acc = __dadd_rn(acc, v[k]);
    }
    out[b] = acc;
}

// out[b] = Σ x[k] * y[k]（ブロック内を先頭から順に）
// CPU の `reduce::dot_block` と同じ。
__global__ void gfem_block_dot(
    const unsigned long long n,
    const unsigned long long block,
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ out)
{
    unsigned long long b = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const unsigned long long s = b * block;
    if (s >= n) return;
    unsigned long long e = s + block;
    if (e > n) e = n;
    double acc = 0.0;
    for (unsigned long long k = s; k < e; ++k) {
        acc = __dadd_rn(acc, __dmul_rn(x[k], y[k]));
    }
    out[b] = acc;
}

} // extern "C"
