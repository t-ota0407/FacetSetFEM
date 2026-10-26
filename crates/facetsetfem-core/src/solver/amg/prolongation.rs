//! 補間演算子の構築（SPEC §4.5）。
//!
//! # 近零空間を壊さないこと
//!
//! 全 Neumann の電流場では K の零空間が定数ベクトル span{1}（SPEC §2.2）。
//! マルチグリッドが効くかどうかは、**この成分が粗レベルに正確に伝わるか**で決まる。
//! 定数が range(P) に入っていなければ、粗格子補正はいちばん収束の遅い成分を
//! 直せず、レベルを増やしても反復回数が減らない。
//!
//! そこで各レベルで近零空間ベクトル `b`（最細レベルでは 1）を持ち回り、
//!
//! - 試行補間 P₀ の列を `b` の凝集への制限そのものにする → `P₀ b_c = b` が厳密
//! - フィルタは `b` 重みで対角に寄せる → `A_f b = A b` が厳密
//! - 補間の平滑化 `P = (I − ω D_f⁻¹ A_f) P₀` は
//!   `P b_c = b − ω D_f⁻¹ A_f b = b − ω D_f⁻¹ A b = b`（A b = 0 なので）
//!
//! となるようにしてある。`A_c b_c = Pᵀ A P b_c = Pᵀ A b = 0` なので、
//! **零空間は全レベルで厳密に保たれる**。テストで各レベルの ‖A_l b_l‖ を見ている。

use crate::assemble::csr::CsrMatrix;
use crate::error::{FacetSetFemError, Result};
use crate::solver::amg::aggregate::{Aggregation, Strength};
use crate::solver::amg::rowwise;

/// `ρ(D⁻¹A)` の Gershgorin 上界。
///
/// 冪乗法で固有値を推定する実装が多いが、反復回数と初期ベクトル次第で値が変わり、
/// 階層が入力だけで決まらなくなる。ここでは**入力から一意に決まる上界**を使う:
///
/// ```text
/// ρ(D⁻¹A) ≤ max_i (Σ_j |A_ij|) / A_ii
/// ```
///
/// 少し保守的（＝ω がやや小さめ）になるが、再現性と引き換えにする価値がある。
///
/// 行ごとに独立なので並列に求める。最大値はどの順に取っても同じ値になり、対角が正でない
/// 行は「最初の 1 行」を探すので、報告する行も逐次と同じ。
pub fn spectral_bound(a: &CsrMatrix) -> Result<f64> {
    use rayon::prelude::*;
    // 行 i の (Σ_j |A_ij|, A_ii)
    let row = |i: usize| -> (f64, f64) {
        let (cols, vals) = a.row(i);
        let mut abs_sum = 0.0;
        let mut diag = 0.0;
        for (&j, &v) in cols.iter().zip(vals) {
            abs_sum += v.abs();
            if j as usize == i {
                diag = v;
            }
        }
        (abs_sum, diag)
    };
    let bad = |diag: f64| diag <= 0.0 || !diag.is_finite();
    let rows = || {
        (0..a.n_rows())
            .into_par_iter()
            .with_min_len(rowwise::ROWS_PER_TASK)
    };
    if let Some(i) = rows().find_first(|&i| bad(row(i).1)) {
        let diag = row(i).1;
        return Err(FacetSetFemError::internal(crate::tr!(
            "AMG: the diagonal entry of row {i} is not positive ({diag})",
            "AMG: 行 {i} の対角成分が正ではありません（{diag}）"
        ))
        .with_hint(crate::tr!(
            "There may be degenerate elements or regions with zero or negative conductivity. \
             preconditioner: jacobi reports the same problem sooner",
            "退化要素か導電率がゼロ以下の領域が残っている可能性があります。\
             preconditioner: jacobi なら同じ診断がより早く出ます"
        )));
    }
    let bound = rows()
        .map(|i| {
            let (abs_sum, diag) = row(i);
            abs_sum / diag
        })
        .reduce(|| 0.0f64, f64::max);
    // 完全に対角な行列（非対角ゼロ）では 1 になる。0 は返さない
    Ok(bound.max(1.0))
}

/// 弱連結を落とし、その分を近零空間ベクトル `b` の重みで対角に寄せた行列。
///
/// ```text
/// A_f[i,j] = A[i,j]                              (j が強連結 または j = i)
/// A_f[i,i] = A[i,i] + (Σ_{j 弱, j≠i} A[i,j] b_j) / b_i
/// ```
///
/// この寄せ方なら `A_f b = A b` が厳密に成り立つ（モジュール冒頭の議論）。
pub fn filtered(a: &CsrMatrix, strength: &Strength, b: &[f64]) -> Result<CsrMatrix> {
    let n = a.n_rows();
    if let Some(i) = b.iter().position(|&v| v == 0.0) {
        return Err(FacetSetFemError::internal(crate::tr!(
            "AMG: component {i} of the near-null-space vector is zero",
            "AMG: 近零空間ベクトルの成分 {i} がゼロです"
        )));
    }

    // 行ごとに独立なので並列に作ってよい（[`super::rowwise`] の説明を参照）。
    // フィルタを当てられるかどうか（対角が正のまま残るか）は、行を作る前に
    // 対角だけ先に計算して判断する
    let keep = |i: usize| -> Option<f64> { lumped_diagonal(a, i, strength.neighbors(i), b) };

    Ok(rowwise::build(
        n,
        n,
        |i| match keep(i) {
            Some(_) => strength.neighbors(i).len() + 1,
            None => a.row(i).0.len(),
        },
        |i, cols, vals| {
            let (a_cols, a_vals) = a.row(i);
            let Some(diag) = keep(i) else {
                // フィルタできない行は元のまま残す。A b = 0 は行ごとに成り立つので、
                // 残しても近零空間の保存は崩れない
                cols.copy_from_slice(a_cols);
                vals.copy_from_slice(a_vals);
                return;
            };
            // 残す列（強連結 ∪ {i}）を昇順に並べ、対角には寄せた後の値を入れる
            let strong = strength.neighbors(i);
            let split = strong.partition_point(|&j| (j as usize) < i);
            cols[..split].copy_from_slice(&strong[..split]);
            cols[split] = i as u32;
            cols[split + 1..].copy_from_slice(&strong[split..]);
            vals[split] = diag;

            // 残した列に A の値を入れる（どちらも昇順なので二重走査で済む）
            let mut cursor = 0usize;
            for (&j, &v) in a_cols.iter().zip(a_vals) {
                if j as usize == i {
                    continue;
                }
                while cursor < cols.len() && cols[cursor] < j {
                    cursor += 1;
                }
                if cursor < cols.len() && cols[cursor] == j {
                    vals[cursor] = v;
                }
            }
        },
    ))
}

/// 弱連結を寄せた後の対角成分。正にならない行では `None`（フィルタを当てない）。
///
/// 強連結が 1 本も無い行では行和がゼロゆえに対角がちょうど消えるので、
/// 必ず `None` になる（粗レベルで実際に起きる）。
fn lumped_diagonal(a: &CsrMatrix, i: usize, strong: &[u32], b: &[f64]) -> Option<f64> {
    if strong.is_empty() {
        return None;
    }
    let (cols, vals) = a.row(i);
    let mut base = 0.0;
    let mut dropped = 0.0;
    let mut cursor = 0usize;
    for (&j, &v) in cols.iter().zip(vals) {
        if j as usize == i {
            base += v;
            continue;
        }
        while cursor < strong.len() && strong[cursor] < j {
            cursor += 1;
        }
        if cursor >= strong.len() || strong[cursor] != j {
            dropped += v * b[j as usize];
        }
    }
    let lumped = base + dropped / b[i];
    (lumped > 0.0).then_some(lumped)
}

/// 補間の平滑化に使う行列 `S = I − ω D⁻¹ A_f`。
///
/// これを試行補間に掛ける（`P = S P₀`）ことで、`P` の列が滑らかになり
/// 補間誤差が下がる。平滑化凝集の名前の由来。
pub fn jacobi_smoother_matrix(af: &CsrMatrix, omega: f64) -> CsrMatrix {
    into_jacobi_smoother(af.clone(), omega)
}

/// [`jacobi_smoother_matrix`] の、`A_f` をその場で書き換える版（複製しない）。
///
/// 階層の構築では `A_f` をこのあと使わないので、こちらを使う（1000 万節点の最細レベルで
/// 1 GB を超える複製が省ける）。値の計算は同じなので、結果もビット単位で同じ。
pub fn into_jacobi_smoother(mut s: CsrMatrix, omega: f64) -> CsrMatrix {
    let diag = s.diagonal();
    let (row_ptr, col_idx, values) = s.parts_mut();
    // 構造は A_f と同じで、値だけ書き換える。行ごとに独立なので並列でよい
    rowwise::update_values(row_ptr, col_idx, values, |i, cols, vals| {
        let scale = omega / diag[i];
        for (v, &c) in vals.iter_mut().zip(cols) {
            let x = -scale * *v;
            *v = if c as usize == i { 1.0 + x } else { x };
        }
    });
    s
}

/// 試行補間 P₀（各行にちょうど 1 つの非ゼロ）と、粗レベルの近零空間ベクトル。
///
/// 凝集 `a` に属する成分を取り出して正規化したものが列になる:
/// `P₀[i, a] = b_i / ‖b|_a‖`、`b_c[a] = ‖b|_a‖`。
pub fn tentative(agg: &Aggregation, b: &[f64]) -> Result<(CsrMatrix, Vec<f64>)> {
    let n = b.len();
    let mut norm2 = vec![0.0f64; agg.count];
    // 節点番号の昇順に足すので順序は一意
    for (i, &g) in agg.of_node.iter().enumerate() {
        norm2[g as usize] += b[i] * b[i];
    }
    let coarse_b: Vec<f64> = norm2.iter().map(|v| v.sqrt()).collect();
    if let Some((g, _)) = coarse_b.iter().enumerate().find(|&(_, &v)| v == 0.0) {
        return Err(FacetSetFemError::internal(crate::tr!(
            "AMG: the near-null-space components of aggregate {g} are all zero",
            "AMG: 凝集 {g} の近零空間成分がすべてゼロです"
        )));
    }

    let row_ptr: Vec<usize> = (0..=n).collect();
    let col_idx: Vec<u32> = agg.of_node.clone();
    let mut p0 = CsrMatrix::from_structure_rect(n, agg.count, row_ptr, col_idx);
    let values = p0.values_mut();
    for i in 0..n {
        values[i] = b[i] / coarse_b[agg.of_node[i] as usize];
    }
    Ok((p0, coarse_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::amg::aggregate::{aggregate, strength_graph};

    fn laplacian_1d(n: usize) -> CsrMatrix {
        let mut row_ptr = vec![0usize];
        let mut col_idx: Vec<u32> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for i in 0..n {
            let mut diag = 0.0;
            if i > 0 {
                col_idx.push(i as u32 - 1);
                values.push(-1.0);
                diag += 1.0;
            }
            let slot = col_idx.len();
            col_idx.push(i as u32);
            values.push(0.0);
            if i + 1 < n {
                col_idx.push(i as u32 + 1);
                values.push(-1.0);
                diag += 1.0;
            }
            values[slot] = diag;
            row_ptr.push(col_idx.len());
        }
        let mut m = CsrMatrix::from_structure(n, row_ptr, col_idx);
        m.values_mut().copy_from_slice(&values);
        m
    }

    #[test]
    fn the_spectral_bound_brackets_the_laplacian() {
        // 1 次元ラプラシアンでは ρ(D⁻¹A) = 2 に近い
        let a = laplacian_1d(64);
        let rho = spectral_bound(&a).unwrap();
        assert!((1.9..=2.1).contains(&rho), "ρ = {rho}");
    }

    #[test]
    fn filtering_preserves_the_action_on_the_nullspace() {
        let a = laplacian_1d(50);
        let s = strength_graph(&a, 0.08);
        let b = vec![1.0; 50];
        let af = filtered(&a, &s, &b).unwrap();
        // A b = 0 なので A_f b = 0 でなければならない
        let mut y = vec![0.0; 50];
        af.spmv(&b, &mut y);
        let worst = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(worst < 1e-14, "A_f b の最大成分 {worst}");
    }

    #[test]
    fn filtering_drops_only_weak_entries() {
        // 強い連結が 1 本、弱い連結が 1 本ある 3×3
        let row_ptr = vec![0, 3, 6, 9];
        let col_idx = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let mut m = CsrMatrix::from_structure(3, row_ptr, col_idx);
        m.values_mut().copy_from_slice(&[
            1.001, -1.0, -0.001, //
            -1.0, 1.001, -0.001, //
            -0.001, -0.001, 0.002,
        ]);
        let s = strength_graph(&m, 0.08);
        let b = vec![1.0; 3];
        let af = filtered(&m, &s, &b).unwrap();
        // 行 0 は 0-1 が強く、0-2 は弱い → 列は {0, 1} だけ
        let (cols, vals) = af.row(0);
        assert_eq!(cols, &[0, 1]);
        // 落とした -0.001 が対角に寄る
        assert!((vals[0] - (1.001 - 0.001)).abs() < 1e-15, "{vals:?}");
    }

    #[test]
    fn the_tentative_prolongator_reproduces_the_nullspace_exactly() {
        let n = 120;
        let a = laplacian_1d(n);
        let s = strength_graph(&a, 0.08);
        let agg = aggregate(&s, n);
        let b = vec![1.0; n];
        let (p0, bc) = tentative(&agg, &b).unwrap();
        assert_eq!(p0.n_rows(), n);
        assert_eq!(p0.n_cols(), agg.count);
        // 1 行 1 非ゼロ
        for i in 0..n {
            assert_eq!(p0.row(i).0.len(), 1);
        }
        // P₀ b_c = b が厳密であること
        let mut y = vec![0.0; n];
        p0.spmv(&bc, &mut y);
        for (i, &v) in y.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-15, "節点 {i}: {v}");
        }
    }

    #[test]
    fn the_smoothed_prolongator_also_reproduces_the_nullspace() {
        let n = 120;
        let a = laplacian_1d(n);
        let s = strength_graph(&a, 0.08);
        let agg = aggregate(&s, n);
        let b = vec![1.0; n];
        let af = filtered(&a, &s, &b).unwrap();
        let omega = 4.0 / (3.0 * spectral_bound(&af).unwrap());
        let smoother = jacobi_smoother_matrix(&af, omega);
        let (p0, bc) = tentative(&agg, &b).unwrap();
        let p = crate::solver::amg::spmm::spmm(&smoother, &p0);

        let mut y = vec![0.0; n];
        p.spmv(&bc, &mut y);
        let worst = y
            .iter()
            .map(|v| (v - 1.0).abs())
            .fold(0.0f64, |m, v| m.max(v));
        assert!(worst < 1e-13, "P b_c − b の最大成分 {worst}");
        // 平滑化で列が広がっていること（試行補間は 1 行 1 非ゼロだった）
        assert!(p.nnz() > p0.nnz());
    }
}
