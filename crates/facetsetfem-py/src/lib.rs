//! FacetSetFEM の Python バインディング（SPEC §6.5）。
//!
//! ここには**ロジックを置かない**。`facetsetfem-core` の型を Python から触れるように
//! 包むだけにして、GUI もバッチスクリプトも同じコアを叩く形にする（SPEC §7）。
//!
//! ```python
//! import facetsetfem
//! mesh = facetsetfem.read_nastran("model.bdf", length_unit="mm")
//! case = facetsetfem.Case(mesh)
//! case.set_conductivity({1: 0.10, 2: 0.04, 3: 0.35})
//! case.add_electrode("E1", role="anode",   current=2e-3, shell_pids=[101])
//! case.add_electrode("E2", role="cathode", current=2e-3, shell_pids=[201])
//! result = case.solve()
//! print(result.impedance)
//! result.write_vtu("out.vtu")
//! ```
//!
//! # ファイルの分け方
//!
//! 以前は 1 つのファイル（2,300 行）だった。Python から見えるものごとに分けてある。
//!
//! | モジュール | 中身 |
//! |---|---|
//! | `mesh` | `Mesh` / `ReadReport`、`read_nastran` |
//! | `case` | `Case`（導電率・電極・ソルバ設定・検証・求解） |
//! | `solution` | `Solution`（結果の配列・統計・書き出し・等値面・流線） |
//! | `patch` | 体表面パッチの辞書の解釈、`check_patch` |
//! | `functions` | モジュール直下の関数（長さ単位・表示言語・選択肢・GPU の問い合わせ） |
//! | `convert` | エラーの変換・numpy 配列・遅延初期化の小さな道具 |
//! | `cuda` | GPU を使うかの判断とフォールバック（`cuda` フィーチャ無しでもビルドできる） |

mod case;
mod convert;
mod cuda;
mod functions;
mod mesh;
mod patch;
mod solution;

use case::PyCase;
use mesh::{PyMesh, PyReadReport};
use pyo3::create_exception;
use pyo3::prelude::*;
use solution::PySolution;

create_exception!(
    _core,
    FacetSetFemError,
    pyo3::exceptions::PyException,
    "FacetSetFEM が投げるエラー。原因・場所・対処が本文に含まれる"
);

create_exception!(
    _core,
    SolverCancelled,
    FacetSetFemError,
    "進捗コールバックから計算が中断された"
);

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", facetsetfem_core::VERSION)?;
    m.add("FacetSetFemError", m.py().get_type::<FacetSetFemError>())?;
    m.add("SolverCancelled", m.py().get_type::<SolverCancelled>())?;
    m.add_class::<PyMesh>()?;
    m.add_class::<PyReadReport>()?;
    m.add_class::<PyCase>()?;
    m.add_class::<PySolution>()?;
    m.add_function(wrap_pyfunction!(mesh::read_nastran, m)?)?;
    m.add_function(wrap_pyfunction!(functions::length_units, m)?)?;
    m.add_function(wrap_pyfunction!(functions::metres_per_unit, m)?)?;
    m.add_function(wrap_pyfunction!(functions::set_language, m)?)?;
    m.add_function(wrap_pyfunction!(functions::language, m)?)?;
    m.add_function(wrap_pyfunction!(functions::backend_choices, m)?)?;
    m.add_function(wrap_pyfunction!(functions::preconditioner_choices, m)?)?;
    m.add_function(wrap_pyfunction!(functions::distribution_choices, m)?)?;
    m.add_function(wrap_pyfunction!(functions::parse_backend, m)?)?;
    m.add_function(wrap_pyfunction!(functions::parse_preconditioner, m)?)?;
    m.add_function(wrap_pyfunction!(functions::wedge_order_choices, m)?)?;
    m.add_function(wrap_pyfunction!(functions::parse_wedge_order, m)?)?;
    m.add_function(wrap_pyfunction!(patch::check_patch, m)?)?;
    m.add_function(wrap_pyfunction!(functions::cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(functions::cuda_devices, m)?)?;
    m.add_function(wrap_pyfunction!(functions::cuda_check, m)?)?;
    Ok(())
}
