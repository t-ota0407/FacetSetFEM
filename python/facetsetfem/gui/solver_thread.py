"""solve をワーカースレッドで走らせる（SPEC §6.4）。

Rust 側は `py.detach()` で GIL を解放して計算するので、進捗コールバックが
呼ばれる瞬間だけ GIL を取り直す。そのコールバックはこのスレッド上で動くため、
Qt シグナルを emit すれば自動的にキュー接続でメインスレッドへ渡る。
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from .. import Case, SolverCancelled, Study

__all__ = ["SolverThread"]

#: 進捗シグナルを送る間隔（反復数）。
#:
#: 数万反復のたびに emit するとイベントループが詰まるので間引く。
#: 収束履歴そのものは Rust 側が全反復ぶん保持しているので、
#: 表示を間引いても後から完全な曲線を描ける。
PROGRESS_STRIDE = 10


class SolverThread(QThread):
    """`Case.solve()` を別スレッドで実行する。"""

    #: (反復回数, 相対残差)
    progressed = Signal(int, float)
    #: 成功したときの `facetsetfem.Solution`
    succeeded = Signal(object)
    #: 失敗したときのメッセージ
    failed = Signal(str)
    #: 利用者が中断した（失敗ではないので別のシグナルにする。警告のダイアログを出さない）
    cancelled = Signal()

    def __init__(self, case: Case, study: Study, parent=None) -> None:
        super().__init__(parent)
        self._case = case
        #: `case` を組み立てたときの解析条件の写し。解いている間に利用者が条件を
        #: 編集しても、結果には解いた条件のほうを添える（実行サマリ。SPEC §5.5）
        self.study = study
        self._cancelled = False

    def cancel(self) -> None:
        """次の反復で計算を打ち切る。"""
        self._cancelled = True

    def run(self) -> None:  # noqa: D102 - QThread の規約
        try:
            result = self._case.solve(progress=self._on_iteration)
        except SolverCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - GUI に出すため全部拾う
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)

    def _on_iteration(self, iteration: int, residual: float) -> bool:
        if iteration <= 5 or iteration % PROGRESS_STRIDE == 0:
            self.progressed.emit(iteration, residual)
        return not self._cancelled
