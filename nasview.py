#!/usr/bin/env python3
"""Visor mínim de desplaçaments Nastran (.h5). Arrossega un .h5 a la finestra."""
import sys
import time

import h5py
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy import QtCore, QtWidgets

COMPONENTS = ["DX", "DY", "DZ", "DT"]


def _index(ids, wanted):
    """Posició a `ids` (qualsevol ordre) de cada valor de `wanted`."""
    order = np.argsort(ids)
    pos = np.searchsorted(ids, wanted, sorter=order)
    return order[np.clip(pos, 0, len(ids) - 1)]


class Model:
    """Geometria + desplaçaments per domini (subcas/mode) d'un h5 de Nastran."""

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            grid = f["/NASTRAN/INPUT/NODE/GRID"][:]
            self.node_ids = grid["ID"]
            self.xyz = np.ascontiguousarray(grid["X"], dtype=np.float64)

            cells = []  # (nombre de nodes, matriu d'ids)
            el = f["/NASTRAN/INPUT/ELEMENT"]
            if "CQUAD4" in el:
                cells.append((4, el["CQUAD4"][:]["G"]))
            if "CTRIA3" in el:
                cells.append((3, el["CTRIA3"][:]["G"]))
            if "CBAR" in el:
                b = el["CBAR"][:]
                cells.append((2, np.stack([b["GA"], b["GB"]], axis=1)))

            self.domains = {}
            res = "/NASTRAN/RESULT/NODAL/DISPLACEMENT"
            if res in f:
                d = f[res][:]
                labels = {}
                if "/NASTRAN/RESULT/DOMAINS" in f:
                    for r in f["/NASTRAN/RESULT/DOMAINS"][:]:
                        labels[int(r["ID"])] = (
                            f"Dom {r['ID']}  SC{r['SUBCASE']}  mode {r['MODE']}  "
                            f"{r['TIME_FREQ_EIGR']:.4g}"
                        )
                for dom in np.unique(d["DOMAIN_ID"]):
                    s = d[d["DOMAIN_ID"] == dom]
                    u = np.zeros((len(self.node_ids), 3))
                    u[_index(self.node_ids, s["ID"])] = np.stack(
                        [s["X"], s["Y"], s["Z"]], axis=1
                    )
                    self.domains[int(dom)] = (labels.get(int(dom), f"Dom {dom}"), u)

        faces, lines = [], []
        for n, g in cells:
            conn = _index(self.node_ids, g.ravel()).reshape(-1, n)
            rows = np.hstack([np.full((len(conn), 1), n), conn]).ravel()
            (lines if n == 2 else faces).append(rows)
        self.mesh = pv.PolyData(self.xyz.copy())
        self.mesh.verts = np.empty(0, dtype=np.int64)  # PolyData(points) crea un vèrtex per node
        if faces:
            self.mesh.faces = np.concatenate(faces)
        if lines:
            self.mesh.lines = np.concatenate(lines)
        self.diag = float(np.linalg.norm(np.ptp(self.xyz, axis=0)))

    def component(self, u, name):
        if name == "DT":
            return np.linalg.norm(u, axis=1)
        return u[:, COMPONENTS.index(name)]


class Viewer(QtWidgets.QMainWindow):
    def __init__(self, path=None):
        super().__init__()
        self.setWindowTitle("Nastran displacement viewer")
        self.resize(1200, 800)
        self.setAcceptDrops(True)
        self.model = None
        self.actor = None
        self.ghost = None
        self.u = None
        self.phase_t0 = time.perf_counter()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        lay = QtWidgets.QHBoxLayout(central)

        self.plotter = QtInteractor(central)
        self.plotter.interactor.setAcceptDrops(False)  # que el drop arribi a la finestra
        self.plotter.enable_anti_aliasing("fxaa")
        self.plotter.set_background("#1e2228", top="#3a414c")
        self.plotter.add_axes()
        lay.addWidget(self.plotter.interactor, 1)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(240)
        form = QtWidgets.QFormLayout(panel)
        lay.addWidget(panel)

        self.dom = QtWidgets.QComboBox()
        self.comp = QtWidgets.QComboBox()
        self.comp.addItems(COMPONENTS)
        self.comp.setCurrentText("DT")
        self.scale = QtWidgets.QDoubleSpinBox()
        self.scale.setDecimals(4)
        self.scale.setRange(-1e12, 1e12)
        self.scale.setValue(1.0)
        self.scale.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.anim = QtWidgets.QCheckBox("Animar")
        self.anim.setChecked(True)
        self.freq = QtWidgets.QDoubleSpinBox()
        self.freq.setRange(0.05, 5)
        self.freq.setSingleStep(0.1)
        self.freq.setValue(0.5)
        self.freq.setSuffix(" Hz")
        self.edges = QtWidgets.QCheckBox("Arestes")
        self.undef = QtWidgets.QCheckBox("Sense deformar")
        self.undef.setChecked(True)
        self.info = QtWidgets.QLabel("Arrossega un .h5 aquí")
        self.info.setWordWrap(True)
        for lbl, w in [
            ("Domini", self.dom), ("Component", self.comp), ("Escala", self.scale),
            ("", self.anim), ("Velocitat", self.freq), ("", self.edges),
            ("", self.undef),
        ]:
            form.addRow(lbl, w)
        form.addRow(self.info)

        self.dom.currentIndexChanged.connect(self.rebuild_data)
        self.comp.currentIndexChanged.connect(self.rebuild_data)
        self.scale.valueChanged.connect(self.tick)
        self.edges.toggled.connect(self.restyle)
        self.undef.toggled.connect(self.restyle)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

        if path:
            self.load(path)

    # --- drag & drop ---
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if p.lower().endswith((".h5", ".hdf5")):
                self.load(p)
                break

    # --- càrrega ---
    def load(self, path):
        try:
            m = Model(path)
        except Exception as ex:  # fitxer fora d'abast
            self.info.setText(f"Error: {ex}")
            return
        self.model = m
        self.setWindowTitle(f"Nastran viewer — {path}")
        self.dom.blockSignals(True)
        self.dom.clear()
        for k, (label, _) in m.domains.items():
            self.dom.addItem(label, k)
        self.dom.blockSignals(False)

        self.plotter.clear()
        self.plotter.add_axes()
        self.mesh = m.mesh
        self.ghost = self.plotter.add_mesh(
            m.mesh.copy(), color="white", style="wireframe", opacity=0.12,
            lighting=False, pickable=False,
        )
        self.actor = None
        self.rebuild_data(reset_scale=True)
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.info.setText(
            f"{len(m.xyz)} nodes\n{m.mesh.n_cells} elements\n{len(m.domains)} dominis"
        )

    def rebuild_data(self, *_, reset_scale=False):
        m = self.model
        if m is None or not m.domains:
            return
        self.u = m.domains[self.dom.currentData()][1]
        vals = m.component(self.u, self.comp.currentText())
        self.mesh.point_data["val"] = vals
        lo, hi = float(vals.min()), float(vals.max())
        if lo == hi:
            hi = lo + 1e-30
        if reset_scale:
            dmax = float(np.linalg.norm(self.u, axis=1).max())
            self.scale.blockSignals(True)
            self.scale.setValue(round(0.05 * m.diag / dmax, 4) if dmax > 0 else 1.0)
            self.scale.blockSignals(False)
        if self.actor is not None:
            self.plotter.remove_actor(self.actor)
        self.actor = self.plotter.add_mesh(
            self.mesh, scalars="val", cmap="turbo", clim=[lo, hi],
            show_edges=self.edges.isChecked(), edge_color="#202020",
            scalar_bar_args={"title": self.comp.currentText(), "color": "white"},
            show_scalar_bar=True, line_width=2, render_lines_as_tubes=False,
        )
        self.restyle()
        self.tick()

    def restyle(self, *_):
        if self.actor is not None:
            self.actor.prop.show_edges = self.edges.isChecked()
        if self.ghost is not None:
            self.ghost.SetVisibility(self.undef.isChecked())
        self.plotter.render()

    def tick(self, *_):
        if self.u is None:
            return
        f = 1.0
        if self.anim.isChecked():
            t = time.perf_counter() - self.phase_t0
            f = np.sin(2 * np.pi * self.freq.value() * t)
        pts = self.mesh.points
        np.add(self.model.xyz, (f * self.scale.value()) * self.u, out=pts)
        self.mesh.GetPoints().Modified()
        self.plotter.render()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Viewer(sys.argv[1] if len(sys.argv) > 1 else None)
    w.show()
    sys.exit(app.exec_() if hasattr(app, "exec_") else app.exec())


if __name__ == "__main__":
    main()
