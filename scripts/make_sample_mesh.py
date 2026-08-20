#!/usr/bin/env python
"""直方体の多層メッシュ（Nastran バルクデータ）を生成する。

SPEC §8 の M0「小規模サンプルモデル（直方体多層メッシュ）」に相当する。
皮膚 / 脂肪 / 筋の 3 層を z 方向に積んだスラブを作り、上面に電極パッチ 2 枚を
CTRIA3 のシェル要素として貼る。

六面体格子を **Kuhn の 6 分割**で四面体化する。全ての立方体で同じ主対角線
（局所節点 0 → 7）を使うため、隣り合う立方体の共有面が同じ対角線で割れ、
四面体メッシュ全体が適合（conforming）になる。

使い方::

    .venv/Scripts/python.exe scripts/make_sample_mesh.py -o tests/models/slab_3layer.bdf
    .venv/Scripts/python.exe scripts/make_sample_mesh.py --format free -o /tmp/free.bdf
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# --- モデル定義（長さの単位は mm）------------------------------------------

#: 層: (名前, PSOLID の PID, 厚さ [mm], z 方向の分割数)
LAYERS: list[tuple[str, int, float, int]] = [
    ("skin", 1, 2.0, 1),
    ("fat", 2, 4.0, 2),
    ("muscle", 3, 14.0, 4),
]

#: 電極パッチ: (名前, PSHELL の PID, x 範囲 [mm], y 範囲 [mm])
#: 既定は上面に 2 枚並べる配置（経皮電気刺激の典型）
ELECTRODES: list[tuple[str, int, tuple[float, float], tuple[float, float]]] = [
    ("E1", 101, (5.0, 20.0), (10.0, 30.0)),
    ("E2", 201, (40.0, 55.0), (10.0, 30.0)),
]

#: 上面全体を陽極、下面全体を陰極にする配置。
#: 一次元問題になるので解析解（各層で E = I/(σA)）と直接照合できる（SPEC §8 M2）
FULL_FACE_ELECTRODES = [("E1", 101, "top"), ("E2", 201, "bottom")]

#: Kuhn の 6 分割。局所節点番号は n = bx + 2*by + 4*bz。
#: いずれも (0,0,0) から (1,1,1) へ 1 軸ずつ進む経路になっている。
KUHN_TETS: list[tuple[int, int, int, int]] = [
    (0, 1, 3, 7),
    (0, 1, 5, 7),
    (0, 2, 3, 7),
    (0, 2, 6, 7),
    (0, 4, 5, 7),
    (0, 4, 6, 7),
]


@dataclass
class Mesh:
    """生成したメッシュ。座標は mm。"""

    coords: list[tuple[float, float, float]]  # 節点索引 -> 座標
    tets: list[tuple[int, int, int, int]]  # 節点索引
    tet_pids: list[int]
    tris: list[tuple[int, int, int]]  # 節点索引
    tri_pids: list[int]

    def node_id(self, index: int) -> int:
        return index + 1


# --- 幾何 -------------------------------------------------------------------


def signed_volume(p0, p1, p2, p3) -> float:
    """四面体の符号付き体積（SPEC §2.3 と同じ式）。"""
    a = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    b = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
    c = (p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2])
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    return (cx * c[0] + cy * c[1] + cz * c[2]) / 6.0


def build_mesh(
    lx: float, ly: float, nx: int, ny: int, refine: int, electrodes: str = "patches"
) -> Mesh:
    """スラブの四面体メッシュと電極シェルを作る。

    ``electrodes`` が ``"patches"`` なら上面に 2 枚のパッチ、``"full"`` なら
    上面全体と下面全体を電極にする（後者は一次元問題の解析解と照合できる）。
    """
    # z 方向の格子。層ごとの分割数に refine を掛ける
    z_levels: list[float] = [0.0]
    z_pid_of_cell: list[int] = []
    for _name, pid, thickness, ndiv in LAYERS:
        ndiv = ndiv * refine
        z0 = z_levels[-1]
        for k in range(1, ndiv + 1):
            z_levels.append(z0 + thickness * k / ndiv)
            z_pid_of_cell.append(pid)
    nz = len(z_levels) - 1

    xs = [lx * i / nx for i in range(nx + 1)]
    ys = [ly * j / ny for j in range(ny + 1)]

    def nid(i: int, j: int, k: int) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    coords = [
        (xs[i], ys[j], z_levels[k])
        for k in range(nz + 1)
        for j in range(ny + 1)
        for i in range(nx + 1)
    ]

    tets: list[tuple[int, int, int, int]] = []
    tet_pids: list[int] = []
    for k in range(nz):
        pid = z_pid_of_cell[k]
        for j in range(ny):
            for i in range(nx):
                # 立方体の 8 頂点（局所番号 n = bx + 2*by + 4*bz）
                corner = [
                    nid(i + (n & 1), j + ((n >> 1) & 1), k + ((n >> 2) & 1))
                    for n in range(8)
                ]
                for a, b, c, d in KUHN_TETS:
                    quad = (corner[a], corner[b], corner[c], corner[d])
                    v = signed_volume(*(coords[t] for t in quad))
                    if v < 0.0:
                        # Nastran の CTETRA は正の体積を要求する。節点を 2 つ入れ替える
                        quad = (quad[0], quad[1], quad[3], quad[2])
                        v = -v
                    if v <= 0.0:
                        raise RuntimeError("退化した四面体が生成されました")
                    tets.append(quad)
                    tet_pids.append(pid)

    # --- 電極面 ---
    # Kuhn 分割では z = 一定の面が (i,j)-(i+1,j+1) の対角線で割れる。
    # シェル要素はこの三角形分割に一致していなければならない（SPEC §11.1）。
    tris: list[tuple[int, int, int]] = []
    tri_pids: list[int] = []

    def add_face(i: int, j: int, k: int, pid: int, outward_up: bool) -> None:
        n00 = nid(i, j, k)
        n10 = nid(i + 1, j, k)
        n11 = nid(i + 1, j + 1, k)
        n01 = nid(i, j + 1, k)
        if outward_up:
            faces = [(n00, n10, n11), (n00, n11, n01)]
        else:
            faces = [(n00, n11, n10), (n00, n01, n11)]
        tris.extend(faces)
        tri_pids.extend([pid] * len(faces))

    if electrodes == "full":
        for _name, pid, side in FULL_FACE_ELECTRODES:
            k, up = (nz, True) if side == "top" else (0, False)
            for j in range(ny):
                for i in range(nx):
                    add_face(i, j, k, pid, up)
    else:
        for _name, pid, (x0, x1), (y0, y1) in ELECTRODES:
            for j in range(ny):
                yc0, yc1 = ys[j], ys[j + 1]
                if yc0 < y0 - 1e-9 or yc1 > y1 + 1e-9:
                    continue
                for i in range(nx):
                    xc0, xc1 = xs[i], xs[i + 1]
                    if xc0 < x0 - 1e-9 or xc1 > x1 + 1e-9:
                        continue
                    add_face(i, j, nz, pid, True)

    return Mesh(coords, tets, tet_pids, tris, tri_pids)


# --- Nastran の書式 ---------------------------------------------------------


def _nastran_to_float(s: str) -> float:
    """Nastran 実数表記（指数の E を省略した形を含む）を float に戻す。"""
    body = s[1:] if s[:1] in "+-" else s
    for i, ch in enumerate(body):
        if ch in "+-" and i > 0 and body[i - 1] not in "eEdD":
            body = body[:i] + "e" + body[i:]
            break
    return float((s[:1] if s[:1] in "+-" else "") + body.replace("D", "e").replace("d", "e"))


def real8(v: float) -> str:
    """8 桁に収まる Nastran 実数表記を返す。可能な限り元の値を保つ。"""
    if v == 0.0:
        return "0.0"
    # 固定小数点（精度の高い順）。小数点以下は必ず 1 桁以上残して読みやすくする
    fixed = [f"{v:.{prec}f}" for prec in range(7, 0, -1)]
    fixed = [s for s in fixed if len(s) <= 8]
    # 指数表記。E を省くと 1 桁稼げる（例: 1.234-3）
    expo = []
    for prec in range(6, -1, -1):
        mantissa, _, exp = f"{v:.{prec}e}".partition("e")
        s = f"{mantissa}{int(exp):+d}"
        if len(s) <= 8:
            expo.append(s)
    # 値を保てるものを優先し、その中では読みやすい固定小数点を先に選ぶ
    for group in (fixed, expo):
        okay = [s for s in group if _nastran_to_float(s) == v]
        if okay:
            return min(okay, key=len)
    for group in (fixed, expo):
        if group:
            return group[0]
    raise ValueError(f"8 桁に収まる Nastran 実数表記が作れません: {v!r}")


def small(fields: list[object]) -> str:
    """小フィールド形式（8 桁固定）の 1 行を組み立てる。"""
    out = []
    for k, f in enumerate(fields):
        s = real8(f) if isinstance(f, float) else str(f)
        if len(s) > 8:
            raise ValueError(f"フィールドが 8 桁を超えます: {s!r}")
        # カード名は左詰め、値も左詰め（Nastran はどちらでもよい）
        out.append(s.ljust(8))
        if k == 8:
            break
    return "".join(out).rstrip()


def large(name: str, fields: list[object]) -> list[str]:
    """大フィールド形式（16 桁）の行を組み立てる。1 行 4 フィールド。"""
    lines = []
    chunks = [fields[i : i + 4] for i in range(0, len(fields), 4)] or [[]]
    for k, chunk in enumerate(chunks):
        head = (name + "*").ljust(8) if k == 0 else "*".ljust(8)
        body = "".join(
            (repr(f) if isinstance(f, float) else str(f)).ljust(16) for f in chunk
        )
        lines.append((head + body).rstrip())
    return lines


def free(fields: list[object]) -> str:
    """自由形式（カンマ区切り）の 1 行を組み立てる。"""
    return ",".join(repr(f) if isinstance(f, float) else str(f) for f in fields)


def _electrode_name(pid: int) -> str:
    """PID から電極名を引く（見つからなければ空文字）。"""
    for name, p, *_rest in ELECTRODES:
        if p == pid:
            return name
    for name, p, _side in FULL_FACE_ELECTRODES:
        if p == pid:
            return name
    return ""


def write_bdf(path: Path, mesh: Mesh, fmt: str, header: list[str]) -> None:
    lines: list[str] = [f"$ {h}" for h in header]
    lines.append("BEGIN BULK")

    def card(name: str, fields: list[object]) -> None:
        if fmt == "free":
            lines.append(free([name] + fields))
        elif fmt == "large":
            lines.extend(large(name, fields))
        else:
            # 小フィールドは 1 行 8 データフィールドまで。それ以上は継続行へ
            first, rest = fields[:8], fields[8:]
            lines.append(small([name] + first))
            while rest:
                chunk, rest = rest[:8], rest[8:]
                lines.append(small([""] + chunk))

    lines.append("$ --- material (not used as conductivity; kept for reference integrity) ---")
    card("MAT1", [1, 1.0, "", 0.3])

    lines.append("$ --- regions: PSOLID PID is the unit sigma is assigned to ---")
    for name, pid, thickness, _ndiv in LAYERS:
        lines.append(f"$ PID {pid}: {name} (thickness {thickness} mm)")
        card("PSOLID", [pid, 1])

    lines.append("$ --- candidate electrode faces ---")
    # 実際に使われたシェル PID から書き出す（電極配置の指定によらず正しくなる）
    for pid in sorted(set(mesh.tri_pids)):
        name = _electrode_name(pid)
        lines.append(f"$ PID {pid}: {name}")
        card("PSHELL", [pid, 1, 0.1])

    lines.append("$ --- nodes ---")
    for index, (x, y, z) in enumerate(mesh.coords):
        card("GRID", [mesh.node_id(index), "", x, y, z])

    lines.append("$ --- solid elements ---")
    eid = 1
    for (a, b, c, d), pid in zip(mesh.tets, mesh.tet_pids):
        card(
            "CTETRA",
            [
                eid,
                pid,
                mesh.node_id(a),
                mesh.node_id(b),
                mesh.node_id(c),
                mesh.node_id(d),
            ],
        )
        eid += 1

    lines.append("$ --- electrode faces (shell elements) ---")
    for (a, b, c), pid in zip(mesh.tris, mesh.tri_pids):
        card("CTRIA3", [eid, pid, mesh.node_id(a), mesh.node_id(b), mesh.node_id(c)])
        eid += 1

    lines.append("ENDDATA")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


# --- エントリポイント -------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", type=Path, default=Path("tests/models/slab_3layer.bdf"))
    p.add_argument("--lx", type=float, default=60.0, help="x 方向の長さ [mm]")
    p.add_argument("--ly", type=float, default=40.0, help="y 方向の長さ [mm]")
    p.add_argument("--nx", type=int, default=12, help="x 方向の分割数")
    p.add_argument("--ny", type=int, default=8, help="y 方向の分割数")
    p.add_argument("--refine", type=int, default=1, help="z 方向の分割数の倍率")
    p.add_argument(
        "--format",
        choices=["small", "free", "large"],
        default="small",
        help="出力する Nastran の書式",
    )
    p.add_argument(
        "--electrodes",
        choices=["patches", "full"],
        default="patches",
        help="patches: 上面に 2 枚のパッチ / full: 上面全体と下面全体（解析解と照合できる）",
    )
    args = p.parse_args(argv)

    mesh = build_mesh(args.lx, args.ly, args.nx, args.ny, args.refine, args.electrodes)
    total_z = sum(t for _n, _p, t, _d in LAYERS)
    header = [
        "GenericFEM sample model: 3-layer slab (skin / fat / muscle)",
        "generated by scripts/make_sample_mesh.py -- length unit is mm",
        f"size: {args.lx} x {args.ly} x {total_z} mm",
        f"divisions: nx={args.nx}, ny={args.ny}, refine={args.refine}",
        f"electrode layout: {args.electrodes}",
    ]
    write_bdf(args.out, mesh, args.format, header)

    print(f"書き出しました: {args.out}")
    print(f"  節点     {len(mesh.coords):>8}")
    print(f"  CTETRA   {len(mesh.tets):>8}")
    print(f"  CTRIA3   {len(mesh.tris):>8}")
    for name, pid, _t, _d in LAYERS:
        n = sum(1 for q in mesh.tet_pids if q == pid)
        print(f"  PID {pid:<4} {name:<8} {n:>8} 要素")
    pid_names = (
        [(n, p) for n, p, _s in FULL_FACE_ELECTRODES]
        if args.electrodes == "full"
        else [(n, p) for n, p, _x, _y in ELECTRODES]
    )
    for name, pid in pid_names:
        n = sum(1 for q in mesh.tri_pids if q == pid)
        area = sum(
            abs(
                (mesh.coords[b][0] - mesh.coords[a][0])
                * (mesh.coords[c][1] - mesh.coords[a][1])
                - (mesh.coords[c][0] - mesh.coords[a][0])
                * (mesh.coords[b][1] - mesh.coords[a][1])
            )
            / 2.0
            for (a, b, c), q in zip(mesh.tris, mesh.tri_pids)
            if q == pid
        )
        print(f"  PID {pid:<4} {name:<8} {n:>8} 面, 面積 {area:.1f} mm^2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
