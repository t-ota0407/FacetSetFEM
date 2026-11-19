//! 最粗レベルの直接解法（SPEC §4.5）。
//!
//! V サイクルの効き目は「最粗レベルをきちんと解けるか」で決まる。ここを
//! 平滑化で誤魔化すと、レベルを増やしても反復回数が頭打ちにならない。
//!
//! # 特異性の扱い
//!
//! 粗レベルの行列も特異である。`A_c b_c = Pᵀ A P b_c = Pᵀ A b = 0`
//! （[`super::prolongation`] の議論）なので、零空間は既知の 1 次元 span{b_c}。
//! 1 節点ピン留めは使わない（CLAUDE.md 設計原則 3）ので、代わりに
//!
//! ```text
//! M = A_c + c b_c b_cᵀ,    c = (tr A_c / n) / ‖b_c‖²
//! ```
//!
//! を分解する。`M b_c = c‖b_c‖² b_c` なので、`b_c` 方向の固有値が
//! 対角平均になり、M は正定値かつ条件数も素直になる。解いた後に `b_c` 成分を
//! 落とせば `A_c` の擬似逆行列による解が得られる。
//!
//! 密 Cholesky なので O(n³/6)。最粗レベルは数百節点に抑えるので一瞬で終わる。
//!
//! # 大きすぎる最粗レベルは分解しない
//!
//! ふつうは [`AmgOptions::coarse_size`](super::AmgOptions::coarse_size)（400）以下まで
//! 粗くしてから分解するが、粗化が途中で進まなくなる（強連結がほとんど無い等）と、
//! そのレベルがそのまま最粗になる。大きさを見ずに密に展開すると、n = 10 万で 80 GB を
//! 確保しようとしてプロセスごと落ちるか、O(n³) の分解で何時間も止まる。
//! [`MAX_DIRECT_SIZE`] を超えたら分解せず、呼び出し側の平滑化による代用に任せる
//! （反復回数は増えるが答えは変わらない。実行サマリに `direct_coarse_solve: false` が残る）。
//!
//! # 正則な系（完全電極モデル。SPEC §2.5 / §4.5）
//!
//! 接触インピーダンスの項が入ると行列は正則になり、`A_c b_c = 0` は成り立たない。
//! そこへ `c b_c b_cᵀ` を足すと別の行列を解くことになり、解いた後に `b_c` 成分を落とすと
//! 前処理が正定値でなくなる（CG が破綻する）。[`Gauge::NotNeeded`] のときは `A_c` を
//! そのまま分解し、成分も落とさない。

use crate::assemble::csr::CsrMatrix;
use crate::solver::cg::Gauge;

/// 密 Cholesky で分解する最粗レベルの上限（節点数）。
///
/// 2000 で密行列 32 MB・分解 1.3 × 10⁹ 回の積和（1 秒前後）。
pub const MAX_DIRECT_SIZE: usize = 2000;

/// 最粗レベルのソルバ。
#[derive(Debug, Clone)]
pub struct CoarseSolver {
    n: usize,
    /// 下三角 L（行優先の密。`l[i * n + j]`、j ≤ i のみ有効）
    l: Vec<f64>,
    /// 近零空間ベクトル
    nullspace: Vec<f64>,
    /// ‖b‖²
    null_norm2: f64,
    /// 零空間方向の rank-1 項で正則化し、解の `b` 成分を落とすか（特異な系のときだけ）
    deflate: bool,
    /// Cholesky に成功したか。失敗したら呼び出し側は平滑化で代用する
    direct: bool,
}

impl CoarseSolver {
    /// 分解する。失敗しても panic せず、[`is_direct`](Self::is_direct) が `false` になる。
    ///
    /// `gauge` が [`Gauge::MeanZero`] なら `A_c` は特異（零空間 span{b}）として扱い、
    /// [`Gauge::NotNeeded`] なら正則としてそのまま分解する（モジュールの説明を参照）。
    pub fn factor(a: &CsrMatrix, nullspace: &[f64], gauge: Gauge) -> Self {
        let n = a.n_rows();
        let deflate = gauge == Gauge::MeanZero;
        let mut solver = Self {
            n,
            l: Vec::new(),
            nullspace: nullspace.to_vec(),
            null_norm2: nullspace.iter().map(|v| v * v).sum(),
            deflate,
            direct: false,
        };
        if n == 0 || n > MAX_DIRECT_SIZE || (deflate && solver.null_norm2 == 0.0) {
            return solver;
        }

        // 密行列に展開してから、零空間方向の rank-1 項で正則化する
        let mut m = vec![0.0f64; n * n];
        let mut trace = 0.0;
        for i in 0..n {
            let (cols, vals) = a.row(i);
            for (&j, &v) in cols.iter().zip(vals) {
                m[i * n + j as usize] = v;
                if j as usize == i {
                    trace += v;
                }
            }
        }
        if trace <= 0.0 || !trace.is_finite() {
            return solver;
        }
        if deflate {
            let c = (trace / n as f64) / solver.null_norm2;
            for i in 0..n {
                for j in 0..n {
                    m[i * n + j] += c * nullspace[i] * nullspace[j];
                }
            }
        }

        // 素直な Cholesky（行優先、上から順に）
        for i in 0..n {
            for j in 0..=i {
                let mut s = m[i * n + j];
                for k in 0..j {
                    s -= m[i * n + k] * m[j * n + k];
                }
                if i == j {
                    if s <= 0.0 || !s.is_finite() {
                        return solver; // direct = false のまま
                    }
                    m[i * n + i] = s.sqrt();
                } else {
                    m[i * n + j] = s / m[j * n + j];
                }
            }
            // 上三角は使わないので消しておく（デバッグしやすさのため）
            for j in (i + 1)..n {
                m[i * n + j] = 0.0;
            }
        }
        solver.l = m;
        solver.direct = true;
        solver
    }

    /// 密分解が使えるか。`false` なら呼び出し側は平滑化で代用する。
    pub fn is_direct(&self) -> bool {
        self.direct
    }

    /// 最粗レベルの節点数。
    pub fn n(&self) -> usize {
        self.n
    }

    /// `A_c x = rhs` の擬似逆行列解（特異な系では `b_c` 成分を落とした代表元、正則な系では解そのもの）。
    ///
    /// [`is_direct`](Self::is_direct) が `false` のときは何もしない。
    // 三角行列の前進・後退代入は「既に求めた成分を添字で引く」形が自然で、
    // イテレータに直すとかえって読めなくなる
    #[allow(clippy::needless_range_loop)]
    pub fn solve(&self, rhs: &[f64], out: &mut [f64]) {
        debug_assert_eq!(rhs.len(), self.n);
        debug_assert_eq!(out.len(), self.n);
        if !self.direct {
            out.fill(0.0);
            return;
        }
        let n = self.n;
        // 前進代入 L y = rhs
        for i in 0..n {
            let mut s = rhs[i];
            for k in 0..i {
                s -= self.l[i * n + k] * out[k];
            }
            out[i] = s / self.l[i * n + i];
        }
        // 後退代入 Lᵀ x = y
        for i in (0..n).rev() {
            let mut s = out[i];
            for k in (i + 1)..n {
                s -= self.l[k * n + i] * out[k];
            }
            out[i] = s / self.l[i * n + i];
        }
        if !self.deflate {
            return;
        }
        // 零空間成分を落として代表元を揃える
        let mut proj = 0.0;
        for i in 0..n {
            proj += self.nullspace[i] * out[i];
        }
        let scale = proj / self.null_norm2;
        for i in 0..n {
            out[i] -= scale * self.nullspace[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::laplacian_1d;

    #[test]
    fn solves_a_singular_system_exactly() {
        let n = 40;
        let a = laplacian_1d(n);
        let b = vec![1.0; n];
        let solver = CoarseSolver::factor(&a, &b, Gauge::MeanZero);
        assert!(solver.is_direct());

        // 零空間に直交する右辺
        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        rhs[n - 1] = -1.0;

        let mut x = vec![0.0; n];
        solver.solve(&rhs, &mut x);

        // A x = rhs を満たすか
        let mut ax = vec![0.0; n];
        a.spmv(&x, &mut ax);
        let worst = ax
            .iter()
            .zip(&rhs)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-10, "残差 {worst}");
        // 代表元は平均ゼロ
        let mean: f64 = x.iter().sum::<f64>() / n as f64;
        assert!(mean.abs() < 1e-12, "平均 {mean}");
    }

    #[test]
    fn repeated_solves_are_bitwise_identical() {
        let n = 30;
        let a = laplacian_1d(n);
        let b = vec![1.0; n];
        let solver = CoarseSolver::factor(&a, &b, Gauge::MeanZero);
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let mut first = vec![0.0; n];
        solver.solve(&rhs, &mut first);
        for _ in 0..5 {
            let mut again = vec![0.0; n];
            solver.solve(&rhs, &mut again);
            assert_eq!(first, again);
        }
    }

    #[test]
    fn a_coarse_level_that_is_too_large_is_not_factorised() {
        // 粗化が途中で止まって大きなレベルが最粗になっても、n² の密行列を
        // 確保しに行かない（以前は大きさを見ずに展開していた）
        let n = MAX_DIRECT_SIZE + 1;
        let a = laplacian_1d(n);
        let solver = CoarseSolver::factor(&a, &vec![1.0; n], Gauge::MeanZero);
        assert!(!solver.is_direct());
        assert_eq!(solver.n(), n);
        assert!(solver.l.is_empty(), "密行列を確保している");
    }

    #[test]
    fn an_indefinite_matrix_is_reported_rather_than_panicking() {
        let row_ptr = vec![0, 1, 2];
        let col_idx = vec![0, 1];
        let mut m = CsrMatrix::from_structure(2, row_ptr, col_idx);
        m.values_mut().copy_from_slice(&[-1.0, -1.0]);
        let solver = CoarseSolver::factor(&m, &[1.0, 1.0], Gauge::MeanZero);
        assert!(!solver.is_direct());
        let mut out = vec![9.0; 2];
        solver.solve(&[1.0, 1.0], &mut out);
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn a_regular_system_is_solved_as_it_is() {
        // 完全電極モデル（SPEC §2.5）の粗い行列を模して、両端にだけ正の対角を足す。
        // 零空間方向の項を足したり b 成分を落としたりすると、ここで答えがずれる
        let n = 25;
        let mut a = laplacian_1d(n);
        let (row_ptr, col_idx, values) = a.parts_mut();
        for i in [0, n - 1] {
            let k = (row_ptr[i]..row_ptr[i + 1])
                .find(|&k| col_idx[k] as usize == i)
                .unwrap();
            values[k] += 0.3;
        }
        let solver = CoarseSolver::factor(&a, &vec![1.0; n], Gauge::NotNeeded);
        assert!(solver.is_direct());
        let rhs: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64).cos()).collect();
        let mut x = vec![0.0; n];
        solver.solve(&rhs, &mut x);
        let mut ax = vec![0.0; n];
        a.spmv(&x, &mut ax);
        let worst = ax
            .iter()
            .zip(&rhs)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-10, "残差 {worst}");
    }
}
