"""Complete PyQt5 workspace backed by :mod:`cycpep_master.application`."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from .. import application as services
from .params_panel import GEOMETRY_PARAM_FIELDS, GeometryParamsPanel
from .render import smiles_to_pixmap
from .workers import ServiceWorker


_PATHS = [
    ("v6", "V6 - fail-closed reconstruction (recommended)"),
    ("a", "A - diagnostic Unified residue-template candidate"),
    ("b", "B - diagnostic HELM/MAP candidate"),
    ("c", "C - diagnostic explicit-cap candidate"),
    ("e", "E - diagnostic geometry-assisted candidate"),
    ("f", "F - diagnostic geometric candidate"),
    ("g", "G - diagnostic special-library candidate"),
    ("h", "H - diagnostic CONECT candidate"),
]

_COORDINATE_FILTER = (
    "Coordinates (*.pdb *.ent *.pdb.gz *.ent.gz *.cif *.mmcif *.cif.gz "
    "*.mmcif.gz);;All files (*)"
)
_PDB_FILTER = "PDB coordinates (*.pdb *.ent);;All files (*)"
_PDBQT_FILTER = "PDBQT (*.pdbqt);;All files (*)"


class MainWindow(QtWidgets.QMainWindow):
    """Task-oriented GUI exposing the same services as the complete CLI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RDpepper")
        self.resize(1380, 860)
        self.setMinimumSize(1040, 680)
        self._workers: set[ServiceWorker] = set()
        self._closing = False
        self._last_result: dict = {}
        self._last_smiles = ""
        self._build_ui()
        QtCore.QTimer.singleShot(0, self._refresh_capabilities)

    def _build_ui(self):
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

        self.workspace_tabs = QtWidgets.QTabWidget()
        self.workspace_tabs.setDocumentMode(True)
        self.workspace_tabs.addTab(self._build_reconstruction_tab(), "Reconstruction")
        self.workspace_tabs.addTab(self._build_notation_tab(), "Representation & audit")
        self.workspace_tabs.addTab(self._build_analysis_tab(), "3D & properties")
        self.workspace_tabs.addTab(self._build_docking_tab(), "PDBQT & docking")
        self.workspace_tabs.addTab(self._build_monomer_tab(), "Monomer library")
        self.workspace_tabs.addTab(self._build_system_tab(), "Capabilities")
        splitter.addWidget(self.workspace_tabs)
        splitter.addWidget(self._build_result_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([820, 560])

        self.status = self.statusBar()
        self.status.showMessage("Ready")

    def _build_result_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QtWidgets.QHBoxLayout()
        self.result_title = QtWidgets.QLabel("No result")
        font = self.result_title.font()
        font.setBold(True)
        self.result_title.setFont(font)
        self.result_status = QtWidgets.QLabel("idle")
        self.result_status.setAlignment(QtCore.Qt.AlignCenter)
        self.result_status.setMinimumWidth(90)
        self.result_status.setFixedHeight(24)
        self.result_rigor = QtWidgets.QLabel("rigor: --")
        self.result_rigor.setAlignment(QtCore.Qt.AlignCenter)
        self.result_rigor.setMinimumWidth(90)
        self.result_rigor.setFixedHeight(24)
        self.result_rigor.setVisible(False)
        header.addWidget(self.result_title, 1)
        header.addWidget(self.result_status)
        header.addWidget(self.result_rigor)
        layout.addLayout(header)

        self.result_geometry_note = QtWidgets.QLabel("已使用非默认几何参数")
        self.result_geometry_note.setStyleSheet(
            "color: #8a4b00; padding: 0 0 4px 0;"
        )
        self.result_geometry_note.setVisible(False)
        layout.addWidget(self.result_geometry_note)

        self.global_monomer_context = QtWidgets.QLineEdit()
        self.global_monomer_context.setPlaceholderText(
            "Optional request-level monomer context JSON"
        )
        layout.addWidget(self.global_monomer_context)

        self.result_tabs = QtWidgets.QTabWidget()
        self.preview_label = QtWidgets.QLabel("No molecular structure")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumSize(360, 300)
        self.preview_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.result_tabs.addTab(self.preview_label, "Structure")
        self.result_json = QtWidgets.QPlainTextEdit()
        self.result_json.setReadOnly(True)
        self.result_json.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.result_tabs.addTab(self.result_json, "JSON")
        layout.addWidget(self.result_tabs, 1)

        actions = QtWidgets.QHBoxLayout()
        copy_button = self._button("Copy JSON", QtWidgets.QStyle.SP_DialogSaveButton)
        copy_button.clicked.connect(self._copy_json)
        save_button = self._button("Save JSON", QtWidgets.QStyle.SP_DialogSaveButton)
        save_button.clicked.connect(self._save_json)
        actions.addWidget(copy_button)
        actions.addWidget(save_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self._set_status_badge("idle")
        return panel

    def _build_reconstruction_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.rec_input_mode = QtWidgets.QComboBox()
        self.rec_input_mode.addItem("Single coordinate file", "file")
        self.rec_input_mode.addItem("Coordinate directory", "directory")
        self.rec_input_mode.currentIndexChanged.connect(self._rec_input_mode_changed)
        form.addRow("Input mode", self.rec_input_mode)

        self.rec_input_stack = QtWidgets.QStackedWidget()
        file_widget, self.rec_file = self._file_field(
            "Select coordinate file", _COORDINATE_FILTER
        )
        dir_widget, self.rec_directory = self._directory_field("Select coordinate directory")
        self.rec_input_stack.addWidget(file_widget)
        self.rec_input_stack.addWidget(dir_widget)
        form.addRow("Input", self.rec_input_stack)

        self.rec_path = QtWidgets.QComboBox()
        for key, label in _PATHS:
            self.rec_path.addItem(label, key)
        form.addRow("Route", self.rec_path)

        chain_row = QtWidgets.QWidget()
        chain_layout = QtWidgets.QHBoxLayout(chain_row)
        chain_layout.setContentsMargins(0, 0, 0, 0)
        self.rec_chain = QtWidgets.QLineEdit("L")
        self.rec_target_chain = QtWidgets.QLineEdit("R")
        self.rec_chain.setMaximumWidth(90)
        self.rec_target_chain.setMaximumWidth(90)
        chain_layout.addWidget(QtWidgets.QLabel("Peptide"))
        chain_layout.addWidget(self.rec_chain)
        chain_layout.addSpacing(12)
        chain_layout.addWidget(QtWidgets.QLabel("Target"))
        chain_layout.addWidget(self.rec_target_chain)
        chain_layout.addStretch(1)
        form.addRow("Chains", chain_row)

        self.rec_multichain = QtWidgets.QCheckBox("Diagnostic multi-chain assembly")
        self.rec_multichain.toggled.connect(self._rec_multichain_changed)
        self.rec_multichain_ids = QtWidgets.QLineEdit()
        self.rec_multichain_ids.setPlaceholderText("A,B,C")
        multi_row = QtWidgets.QWidget()
        multi_layout = QtWidgets.QHBoxLayout(multi_row)
        multi_layout.setContentsMargins(0, 0, 0, 0)
        multi_layout.addWidget(self.rec_multichain)
        multi_layout.addWidget(self.rec_multichain_ids, 1)
        form.addRow("Assembly", multi_row)

        self.rec_monomer_context = QtWidgets.QLineEdit()
        self.rec_monomer_context.setPlaceholderText(
            '{"definitions":[...]} or {"ccd_directory":"..."}'
        )
        form.addRow("Monomer context", self.rec_monomer_context)

        options = QtWidgets.QWidget()
        option_layout = QtWidgets.QGridLayout(options)
        option_layout.setContentsMargins(0, 0, 0, 0)
        self.rec_admet = QtWidgets.QCheckBox("ADMET")
        self.rec_flexibility = QtWidgets.QCheckBox("Flexibility proxy")
        self.rec_docking = QtWidgets.QCheckBox("Vina integration")
        self.rec_empty_overlay = QtWidgets.QCheckBox("Require empty persistent overlay")
        self.rec_allow_linear_topology = QtWidgets.QCheckBox(
            "Allow linear topology (no cyclization evidence)"
        )
        option_layout.addWidget(self.rec_admet, 0, 0)
        option_layout.addWidget(self.rec_flexibility, 0, 1)
        option_layout.addWidget(self.rec_docking, 1, 0)
        option_layout.addWidget(self.rec_empty_overlay, 1, 1)
        option_layout.addWidget(self.rec_allow_linear_topology, 2, 0, 1, 2)
        form.addRow("Optional stages", options)

        self.geometry_params = GeometryParamsPanel()
        geometry_group = QtWidgets.QGroupBox("几何参数（高级）")
        geometry_layout = QtWidgets.QVBoxLayout(geometry_group)
        geometry_layout.setContentsMargins(6, 4, 6, 6)
        geometry_layout.addWidget(self.geometry_params)
        form.addRow("", geometry_group)

        export_widget, self.rec_export_dir = self._directory_field("Select export directory")
        self.rec_export_format = QtWidgets.QComboBox()
        self.rec_export_format.addItems(["mol2", "sdf"])
        export_row = QtWidgets.QWidget()
        export_layout = QtWidgets.QHBoxLayout(export_row)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.addWidget(export_widget, 1)
        export_layout.addWidget(self.rec_export_format)
        form.addRow("3D export", export_row)

        csv_widget, self.rec_csv = self._save_field(
            "Select CSV output", "CSV (*.csv);;All files (*)"
        )
        form.addRow("CSV output", csv_widget)

        self.rec_run = self._button("Run reconstruction", QtWidgets.QStyle.SP_MediaPlay)
        self.rec_run.clicked.connect(self._run_reconstruction)
        layout.addWidget(self.rec_run)

        self.rec_table = QtWidgets.QTableWidget(0, 7)
        self.rec_table.setHorizontalHeaderLabels(
            ["File", "Status", "Support", "Qualified", "Route", "Topology", "SMILES"]
        )
        self.rec_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.rec_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.rec_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.rec_table, 1)
        return page

    def _build_notation_tab(self):
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)

        convert_page = QtWidgets.QWidget()
        convert_layout = QtWidgets.QVBoxLayout(convert_page)
        convert_form = QtWidgets.QFormLayout()
        self.convert_source = QtWidgets.QComboBox()
        self.convert_target = QtWidgets.QComboBox()
        for kind in services.REPRESENTATION_KINDS:
            self.convert_source.addItem(kind.upper(), kind)
            self.convert_target.addItem(kind.upper(), kind)
        self.convert_target.setCurrentIndex(1)
        kinds = QtWidgets.QWidget()
        kinds_layout = QtWidgets.QHBoxLayout(kinds)
        kinds_layout.setContentsMargins(0, 0, 0, 0)
        kinds_layout.addWidget(self.convert_source)
        kinds_layout.addWidget(self._arrow_label())
        kinds_layout.addWidget(self.convert_target)
        kinds_layout.addStretch(1)
        convert_form.addRow("Conversion", kinds)
        self.convert_input = QtWidgets.QPlainTextEdit()
        self.convert_input.setPlaceholderText("MAP / HELM / BILN / SMILES")
        self.convert_input.setMinimumHeight(150)
        convert_form.addRow("Input", self.convert_input)
        self.convert_output = QtWidgets.QPlainTextEdit()
        self.convert_output.setReadOnly(True)
        self.convert_output.setMinimumHeight(120)
        convert_form.addRow("Output", self.convert_output)
        convert_layout.addLayout(convert_form)
        convert_button = self._button("Convert", QtWidgets.QStyle.SP_ArrowForward)
        convert_button.clicked.connect(self._run_conversion)
        convert_layout.addWidget(convert_button)
        convert_layout.addStretch(1)
        tabs.addTab(convert_page, "Convert")

        audit_page = QtWidgets.QWidget()
        audit_layout = QtWidgets.QVBoxLayout(audit_page)
        audit_form = QtWidgets.QFormLayout()
        self.audit_kind = QtWidgets.QComboBox()
        for kind in services.AUDIT_KINDS:
            self.audit_kind.addItem(kind.upper().replace("_", " "), kind)
        audit_form.addRow("Format", self.audit_kind)
        audit_file_widget, self.audit_file = self._file_field(
            "Select input file", "All files (*)"
        )
        audit_form.addRow("Input file", audit_file_widget)
        self.audit_payload = QtWidgets.QPlainTextEdit()
        self.audit_payload.setMinimumHeight(260)
        audit_form.addRow("Payload", self.audit_payload)
        audit_layout.addLayout(audit_form)
        audit_button = self._button("Run audit", QtWidgets.QStyle.SP_DialogApplyButton)
        audit_button.clicked.connect(self._run_audit)
        audit_layout.addWidget(audit_button)
        audit_layout.addStretch(1)
        tabs.addTab(audit_page, "Audit")

        compare_page = QtWidgets.QWidget()
        compare_layout = QtWidgets.QVBoxLayout(compare_page)
        compare_form = QtWidgets.QFormLayout()
        self.compare_mode = QtWidgets.QComboBox()
        self.compare_mode.addItem("Strict molecular identity", "strict")
        self.compare_mode.addItem("Legacy permissive", "permissive")
        self.compare_mode.addItem("Specified stereo constraints", "specified-stereo")
        compare_form.addRow("Mode", self.compare_mode)
        self.compare_left = QtWidgets.QPlainTextEdit()
        self.compare_left.setMinimumHeight(130)
        compare_form.addRow("Reference", self.compare_left)
        self.compare_right = QtWidgets.QPlainTextEdit()
        self.compare_right.setMinimumHeight(130)
        compare_form.addRow("Observed", self.compare_right)
        compare_layout.addLayout(compare_form)
        compare_button = self._button("Compare", QtWidgets.QStyle.SP_DialogApplyButton)
        compare_button.clicked.connect(self._run_comparison)
        compare_layout.addWidget(compare_button)
        compare_layout.addStretch(1)
        tabs.addTab(compare_page, "Compare")
        return tabs

    def _build_analysis_tab(self):
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)

        export_page = QtWidgets.QWidget()
        export_layout = QtWidgets.QVBoxLayout(export_page)
        export_form = QtWidgets.QFormLayout()
        self.export_source_kind = QtWidgets.QComboBox()
        self.export_source_kind.addItem("SMILES", "smiles")
        self.export_source_kind.addItem("Coordinate file", "coordinate")
        self.export_source_kind.currentIndexChanged.connect(self._export_source_changed)
        export_form.addRow("Source type", self.export_source_kind)
        self.export_source_stack = QtWidgets.QStackedWidget()
        self.export_smiles = QtWidgets.QPlainTextEdit()
        self.export_smiles.setMinimumHeight(100)
        coordinate_widget, self.export_coordinate = self._file_field(
            "Select coordinate file", _COORDINATE_FILTER
        )
        self.export_source_stack.addWidget(self.export_smiles)
        self.export_source_stack.addWidget(coordinate_widget)
        export_form.addRow("Source", self.export_source_stack)
        output_widget, self.export_output = self._save_field(
            "Select 3D output", "MOL2 (*.mol2);;SDF (*.sdf);;All files (*)"
        )
        export_form.addRow("Output", output_widget)
        self.export_format = QtWidgets.QComboBox()
        self.export_format.addItems(["mol2", "sdf"])
        export_form.addRow("Format", self.export_format)
        self.export_forcefield = QtWidgets.QComboBox()
        self.export_forcefield.addItems(["mmff", "uff"])
        self.export_num_confs = QtWidgets.QSpinBox()
        self.export_num_confs.setRange(1, 1000)
        self.export_num_confs.setValue(10)
        self.export_seed = QtWidgets.QSpinBox()
        self.export_seed.setRange(0, 2_147_483_647)
        self.export_seed.setValue(42)
        export_sampling = QtWidgets.QWidget()
        export_sampling_layout = QtWidgets.QHBoxLayout(export_sampling)
        export_sampling_layout.setContentsMargins(0, 0, 0, 0)
        export_sampling_layout.addWidget(self.export_forcefield)
        export_sampling_layout.addWidget(QtWidgets.QLabel("Conformers"))
        export_sampling_layout.addWidget(self.export_num_confs)
        export_sampling_layout.addWidget(QtWidgets.QLabel("Seed"))
        export_sampling_layout.addWidget(self.export_seed)
        export_sampling_layout.addStretch(1)
        export_form.addRow("3D sampling", export_sampling)
        self.export_path = QtWidgets.QComboBox()
        for key, label in _PATHS:
            self.export_path.addItem(label, key)
        export_form.addRow("Coordinate route", self.export_path)
        self.export_fallback_policy = QtWidgets.QComboBox()
        self.export_fallback_policy.addItem(
            "Maximum coverage (degrade honestly)", "max_coverage"
        )
        self.export_fallback_policy.addItem(
            "Strict V6 coordinate mapping", "strict_v6"
        )
        export_form.addRow(
            "Fallback policy", self.export_fallback_policy
        )
        self.export_chain = QtWidgets.QLineEdit("L")
        export_form.addRow("Chain", self.export_chain)
        export_layout.addLayout(export_form)
        export_button = self._button("Export 3D", QtWidgets.QStyle.SP_DialogSaveButton)
        export_button.clicked.connect(self._run_export)
        export_layout.addWidget(export_button)
        export_layout.addStretch(1)
        tabs.addTab(export_page, "Export")

        batch_export_page = QtWidgets.QWidget()
        batch_export_layout = QtWidgets.QVBoxLayout(batch_export_page)
        batch_export_form = QtWidgets.QFormLayout()
        manifest_widget, self.batch_export_manifest = self._file_field(
            "Select named SMILES manifest", "JSON (*.json);;All files (*)"
        )
        batch_export_form.addRow("Manifest", manifest_widget)
        batch_output_widget, self.batch_export_output_dir = self._directory_field(
            "Select batch export directory"
        )
        batch_export_form.addRow("Output directory", batch_output_widget)
        self.batch_export_format = QtWidgets.QComboBox()
        self.batch_export_format.addItems(["mol2", "sdf"])
        self.batch_export_forcefield = QtWidgets.QComboBox()
        self.batch_export_forcefield.addItems(["mmff", "uff"])
        batch_options = QtWidgets.QWidget()
        batch_options_layout = QtWidgets.QHBoxLayout(batch_options)
        batch_options_layout.setContentsMargins(0, 0, 0, 0)
        batch_options_layout.addWidget(QtWidgets.QLabel("Format"))
        batch_options_layout.addWidget(self.batch_export_format)
        batch_options_layout.addSpacing(12)
        batch_options_layout.addWidget(QtWidgets.QLabel("Force field"))
        batch_options_layout.addWidget(self.batch_export_forcefield)
        batch_options_layout.addStretch(1)
        batch_export_form.addRow("Options", batch_options)
        batch_export_layout.addLayout(batch_export_form)
        batch_export_button = self._button(
            "Export manifest", QtWidgets.QStyle.SP_DialogSaveButton
        )
        batch_export_button.clicked.connect(self._run_batch_export)
        batch_export_layout.addWidget(batch_export_button)
        batch_export_layout.addStretch(1)
        tabs.addTab(batch_export_page, "Batch export")

        conformer_page = QtWidgets.QWidget()
        conformer_layout = QtWidgets.QVBoxLayout(conformer_page)
        conformer_form = QtWidgets.QFormLayout()
        self.conf_smiles = QtWidgets.QPlainTextEdit()
        self.conf_smiles.setMinimumHeight(110)
        conformer_form.addRow("SMILES", self.conf_smiles)
        self.conf_count = QtWidgets.QSpinBox()
        self.conf_count.setRange(1, 1000)
        self.conf_count.setValue(50)
        self.conf_seed = QtWidgets.QSpinBox()
        self.conf_seed.setRange(0, 2_147_483_647)
        self.conf_seed.setValue(42)
        count_row = QtWidgets.QWidget()
        count_layout = QtWidgets.QHBoxLayout(count_row)
        count_layout.setContentsMargins(0, 0, 0, 0)
        count_layout.addWidget(QtWidgets.QLabel("Conformers"))
        count_layout.addWidget(self.conf_count)
        count_layout.addSpacing(12)
        count_layout.addWidget(QtWidgets.QLabel("Seed"))
        count_layout.addWidget(self.conf_seed)
        count_layout.addStretch(1)
        conformer_form.addRow("Sampling", count_row)
        self.conf_forcefield = QtWidgets.QComboBox()
        self.conf_forcefield.addItems(["mmff", "uff"])
        self.conf_optimize = QtWidgets.QCheckBox("Energy minimization")
        self.conf_optimize.setChecked(True)
        force_row = QtWidgets.QWidget()
        force_layout = QtWidgets.QHBoxLayout(force_row)
        force_layout.setContentsMargins(0, 0, 0, 0)
        force_layout.addWidget(self.conf_forcefield)
        force_layout.addWidget(self.conf_optimize)
        force_layout.addStretch(1)
        conformer_form.addRow("Force field", force_row)
        self.conf_energy_window = QtWidgets.QDoubleSpinBox()
        self.conf_energy_window.setRange(-1.0, 10000.0)
        self.conf_energy_window.setValue(-1.0)
        self.conf_energy_window.setSpecialValueText("All")
        conformer_form.addRow("Energy window", self.conf_energy_window)
        self.conf_max_heavy_atoms = QtWidgets.QSpinBox()
        self.conf_max_heavy_atoms.setRange(0, 100000)
        self.conf_max_heavy_atoms.setSpecialValueText("Default safety limit")
        conformer_form.addRow("Maximum heavy atoms", self.conf_max_heavy_atoms)
        conformer_layout.addLayout(conformer_form)
        conformer_button = self._button("Compute statistics", QtWidgets.QStyle.SP_MediaPlay)
        conformer_button.clicked.connect(self._run_conformers)
        conformer_layout.addWidget(conformer_button)
        conformer_layout.addStretch(1)
        tabs.addTab(conformer_page, "Conformers")

        template_page = QtWidgets.QWidget()
        template_layout = QtWidgets.QVBoxLayout(template_page)
        template_form = QtWidgets.QFormLayout()
        self.template_map = QtWidgets.QPlainTextEdit()
        self.template_map.setMinimumHeight(90)
        template_form.addRow("MAP", self.template_map)
        self.template_smiles = QtWidgets.QPlainTextEdit()
        self.template_smiles.setMinimumHeight(90)
        template_form.addRow("SMILES", self.template_smiles)
        template_output_widget, self.template_output = self._save_field(
            "Select template conformer SDF", "SDF (*.sdf);;All files (*)"
        )
        template_form.addRow("Ensemble output", template_output_widget)
        self.template_strategy = QtWidgets.QComboBox()
        for strategy in services.TEMPLATE_STRATEGIES:
            self.template_strategy.addItem(strategy.replace("_", " ").title(), strategy)
        self.template_count = QtWidgets.QSpinBox()
        self.template_count.setRange(1, 1000)
        self.template_count.setValue(5)
        self.template_seed = QtWidgets.QSpinBox()
        self.template_seed.setRange(0, 2_147_483_647)
        self.template_seed.setValue(42)
        template_options = QtWidgets.QWidget()
        template_options_layout = QtWidgets.QHBoxLayout(template_options)
        template_options_layout.setContentsMargins(0, 0, 0, 0)
        template_options_layout.addWidget(self.template_strategy)
        template_options_layout.addWidget(QtWidgets.QLabel("Conformers"))
        template_options_layout.addWidget(self.template_count)
        template_options_layout.addWidget(QtWidgets.QLabel("Seed"))
        template_options_layout.addWidget(self.template_seed)
        template_options_layout.addStretch(1)
        template_form.addRow("Strategy", template_options)
        template_layout.addLayout(template_form)
        template_actions = QtWidgets.QHBoxLayout()
        template_lookup_button = self._button(
            "Find template", QtWidgets.QStyle.SP_DialogApplyButton
        )
        template_lookup_button.clicked.connect(self._run_template_lookup)
        template_generate_button = self._button(
            "Generate ensemble", QtWidgets.QStyle.SP_DialogSaveButton
        )
        template_generate_button.clicked.connect(self._run_template_conformers)
        template_actions.addWidget(template_lookup_button)
        template_actions.addWidget(template_generate_button)
        template_actions.addStretch(1)
        template_layout.addLayout(template_actions)
        template_layout.addStretch(1)
        tabs.addTab(template_page, "Templates")

        admet_page = QtWidgets.QWidget()
        admet_layout = QtWidgets.QVBoxLayout(admet_page)
        self.admet_input = QtWidgets.QPlainTextEdit()
        self.admet_input.setMinimumHeight(150)
        admet_layout.addWidget(QtWidgets.QLabel("SMILES (one per line)"))
        admet_layout.addWidget(self.admet_input)
        admet_button = self._button("Run ADMET", QtWidgets.QStyle.SP_MediaPlay)
        admet_button.clicked.connect(self._run_admet)
        admet_layout.addWidget(admet_button)
        self.admet_table = QtWidgets.QTableWidget(0, 0)
        self.admet_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        admet_layout.addWidget(self.admet_table, 1)
        tabs.addTab(admet_page, "ADMET")
        return tabs

    def _build_docking_tab(self):
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)

        sequence_page = QtWidgets.QWidget()
        sequence_layout = QtWidgets.QVBoxLayout(sequence_page)
        sequence_form = QtWidgets.QFormLayout()
        self.sequence_input = QtWidgets.QPlainTextEdit()
        self.sequence_input.setMinimumHeight(80)
        sequence_form.addRow("Sequence", self.sequence_input)
        sequence_output_widget, self.sequence_output_dir = (
            self._directory_field("Select sequence artifact directory")
        )
        sequence_form.addRow("Output directory", sequence_output_widget)
        self.sequence_cyclization = QtWidgets.QComboBox()
        for label, value in (
            ("Head-to-tail", "head-to-tail"),
            ("Linear", "linear"),
            ("Infer (hypothesis)", "infer"),
        ):
            self.sequence_cyclization.addItem(label, value)
        sequence_form.addRow(
            "Cyclization", self.sequence_cyclization
        )
        self.sequence_stereochemistry = QtWidgets.QLineEdit()
        self.sequence_stereochemistry.setPlaceholderText(
            '{"2":"D"} (optional JSON)'
        )
        sequence_form.addRow(
            "Stereochemistry", self.sequence_stereochemistry
        )
        self.sequence_terminal_modifications = QtWidgets.QLineEdit()
        self.sequence_terminal_modifications.setPlaceholderText(
            '{"N":"ACE","C":"NME"} (optional JSON)'
        )
        sequence_form.addRow(
            "Terminal modifications",
            self.sequence_terminal_modifications,
        )
        self.sequence_monomer_context = QtWidgets.QLineEdit()
        self.sequence_monomer_context.setPlaceholderText(
            '{"definitions":[...],"ccd_directory":"..."} '
            "(optional entity-local JSON)"
        )
        sequence_form.addRow(
            "Monomer extensions",
            self.sequence_monomer_context,
        )
        self.sequence_protonation = QtWidgets.QComboBox()
        self.sequence_protonation.addItem(
            "Registry default", "registry_default"
        )
        self.sequence_protonation.addItem(
            "Physiological heuristic", "physiological"
        )
        sequence_form.addRow(
            "Protonation", self.sequence_protonation
        )
        sequence_controls = QtWidgets.QWidget()
        sequence_controls_layout = QtWidgets.QHBoxLayout(
            sequence_controls
        )
        sequence_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.sequence_conformers = QtWidgets.QSpinBox()
        self.sequence_conformers.setRange(1, 100)
        self.sequence_conformers.setValue(4)
        self.sequence_seed = QtWidgets.QSpinBox()
        self.sequence_seed.setRange(0, 2_147_483_647)
        self.sequence_seed.setValue(42)
        self.sequence_threads = QtWidgets.QSpinBox()
        self.sequence_threads.setRange(1, 256)
        self.sequence_threads.setValue(1)
        for label, widget in (
            ("Conformers", self.sequence_conformers),
            ("Seed", self.sequence_seed),
            ("Threads", self.sequence_threads),
        ):
            sequence_controls_layout.addWidget(QtWidgets.QLabel(label))
            sequence_controls_layout.addWidget(widget)
        sequence_controls_layout.addStretch(1)
        sequence_form.addRow("Materialization", sequence_controls)
        sequence_flexibility = QtWidgets.QWidget()
        sequence_flexibility_layout = QtWidgets.QHBoxLayout(
            sequence_flexibility
        )
        sequence_flexibility_layout.setContentsMargins(0, 0, 0, 0)
        self.sequence_generate_pdbqt = QtWidgets.QCheckBox(
            "Generate PDBQT"
        )
        self.sequence_generate_pdbqt.setChecked(True)
        self.sequence_flexibility_mode = QtWidgets.QComboBox()
        for value in ("fast", "balanced", "thorough"):
            self.sequence_flexibility_mode.addItem(
                value.capitalize(), value
            )
        self.sequence_flexibility_mode.setCurrentIndex(1)
        self.sequence_torsdof = QtWidgets.QSpinBox()
        self.sequence_torsdof.setRange(-1, 1000)
        self.sequence_torsdof.setSpecialValueText("No budget")
        self.sequence_torsdof.setValue(-1)
        self.sequence_template_strategy = QtWidgets.QComboBox()
        for value in services.TEMPLATE_STRATEGIES:
            self.sequence_template_strategy.addItem(value, value)
        sequence_flexibility_layout.addWidget(
            self.sequence_generate_pdbqt
        )
        sequence_flexibility_layout.addWidget(
            QtWidgets.QLabel("Mode")
        )
        sequence_flexibility_layout.addWidget(
            self.sequence_flexibility_mode
        )
        sequence_flexibility_layout.addWidget(
            QtWidgets.QLabel("TORSDOF")
        )
        sequence_flexibility_layout.addWidget(self.sequence_torsdof)
        sequence_flexibility_layout.addWidget(
            QtWidgets.QLabel("Templates")
        )
        sequence_flexibility_layout.addWidget(
            self.sequence_template_strategy
        )
        sequence_flexibility_layout.addStretch(1)
        sequence_form.addRow("PDBQT", sequence_flexibility)
        sequence_prior_widget, self.sequence_torsion_prior = (
            self._file_field(
                "Select torsion-prior runtime JSON",
                "JSON (*.json);;All files (*)",
            )
        )
        sequence_form.addRow(
            "Torsion prior (optional)", sequence_prior_widget
        )
        sequence_layout.addLayout(sequence_form)
        sequence_button = self._button(
            "Prepare sequence",
            QtWidgets.QStyle.SP_DialogSaveButton,
        )
        sequence_button.clicked.connect(self._run_prepare_sequence)
        sequence_layout.addWidget(sequence_button)
        sequence_layout.addStretch(1)
        tabs.addTab(sequence_page, "Sequence")

        ligand = QtWidgets.QWidget()
        ligand_layout = QtWidgets.QVBoxLayout(ligand)
        ligand_form = QtWidgets.QFormLayout()
        self.ligand_smiles = QtWidgets.QPlainTextEdit()
        self.ligand_smiles.setMinimumHeight(90)
        ligand_form.addRow("SMILES", self.ligand_smiles)
        self.ligand_map = QtWidgets.QLineEdit()
        ligand_form.addRow("MAP", self.ligand_map)
        ligand_output_widget, self.ligand_output = self._save_field(
            "Select ligand PDBQT", "PDBQT (*.pdbqt);;All files (*)"
        )
        ligand_form.addRow("Output", ligand_output_widget)
        self.ligand_torsdof = QtWidgets.QSpinBox()
        self.ligand_torsdof.setRange(-1, 1000)
        self.ligand_torsdof.setSpecialValueText("Automatic")
        self.ligand_torsdof.setValue(-1)
        self.ligand_num_confs = QtWidgets.QSpinBox()
        self.ligand_num_confs.setRange(1, 1000)
        self.ligand_num_confs.setValue(10)
        self.ligand_seed = QtWidgets.QSpinBox()
        self.ligand_seed.setRange(0, 2_147_483_647)
        self.ligand_seed.setValue(42)
        self.ligand_torsion_ensemble_size = QtWidgets.QSpinBox()
        self.ligand_torsion_ensemble_size.setRange(2, 1000)
        self.ligand_torsion_ensemble_size.setValue(8)
        self.ligand_torsion_threads = QtWidgets.QSpinBox()
        self.ligand_torsion_threads.setRange(1, 256)
        self.ligand_torsion_threads.setValue(1)
        budget = QtWidgets.QWidget()
        budget_layout = QtWidgets.QHBoxLayout(budget)
        budget_layout.setContentsMargins(0, 0, 0, 0)
        budget_layout.addWidget(QtWidgets.QLabel("TORSDOF limit"))
        budget_layout.addWidget(self.ligand_torsdof)
        budget_layout.addSpacing(12)
        budget_layout.addWidget(QtWidgets.QLabel("Conformers"))
        budget_layout.addWidget(self.ligand_num_confs)
        budget_layout.addSpacing(12)
        budget_layout.addWidget(QtWidgets.QLabel("Seed"))
        budget_layout.addWidget(self.ligand_seed)
        budget_layout.addStretch(1)
        ligand_form.addRow("Budget", budget)
        torsion_sampling = QtWidgets.QWidget()
        torsion_sampling_layout = QtWidgets.QHBoxLayout(torsion_sampling)
        torsion_sampling_layout.setContentsMargins(0, 0, 0, 0)
        torsion_sampling_layout.addWidget(QtWidgets.QLabel("Ensemble size"))
        torsion_sampling_layout.addWidget(self.ligand_torsion_ensemble_size)
        torsion_sampling_layout.addSpacing(12)
        torsion_sampling_layout.addWidget(QtWidgets.QLabel("Threads"))
        torsion_sampling_layout.addWidget(self.ligand_torsion_threads)
        torsion_sampling_layout.addStretch(1)
        ligand_form.addRow("Torsion sampling", torsion_sampling)
        self.ligand_rigid = QtWidgets.QCheckBox("Rigid macrocycles")
        self.ligand_rigid.setChecked(True)
        self.ligand_protonate = QtWidgets.QCheckBox("pH 7.4 protonation")
        self.ligand_protonate.setChecked(True)
        flags = QtWidgets.QWidget()
        flags_layout = QtWidgets.QHBoxLayout(flags)
        flags_layout.setContentsMargins(0, 0, 0, 0)
        flags_layout.addWidget(self.ligand_rigid)
        flags_layout.addWidget(self.ligand_protonate)
        flags_layout.addStretch(1)
        ligand_form.addRow("Preparation", flags)
        ligand_layout.addLayout(ligand_form)
        ligand_button = self._button("Prepare ligand PDBQT", QtWidgets.QStyle.SP_DialogSaveButton)
        ligand_button.clicked.connect(self._run_ligand_pdbqt)
        ligand_layout.addWidget(ligand_button)
        ligand_layout.addStretch(1)
        tabs.addTab(ligand, "Ligand PDBQT")

        ligand_file = QtWidgets.QWidget()
        ligand_file_layout = QtWidgets.QVBoxLayout(ligand_file)
        ligand_file_form = QtWidgets.QFormLayout()
        ligand_pdb_widget, self.ligand_pdb_input = self._file_field(
            "Select ligand PDB", _PDB_FILTER
        )
        ligand_file_form.addRow("PDB input", ligand_pdb_widget)
        ligand_pdb_output_widget, self.ligand_pdb_output = self._save_field(
            "Select ligand PDBQT", _PDBQT_FILTER
        )
        ligand_file_form.addRow("PDBQT output", ligand_pdb_output_widget)
        ligand_file_layout.addLayout(ligand_file_form)
        ligand_pdb_button = self._button(
            "Prepare from PDB", QtWidgets.QStyle.SP_DialogSaveButton
        )
        ligand_pdb_button.clicked.connect(self._run_ligand_pdbqt_from_pdb)
        ligand_file_layout.addWidget(ligand_pdb_button)
        ligand_file_layout.addStretch(1)
        tabs.addTab(ligand_file, "PDB ligand")

        receptor = QtWidgets.QWidget()
        receptor_layout = QtWidgets.QVBoxLayout(receptor)
        receptor_form = QtWidgets.QFormLayout()
        receptor_input_widget, self.receptor_input = self._file_field(
            "Select receptor PDB", _PDB_FILTER
        )
        receptor_form.addRow("Receptor", receptor_input_widget)
        receptor_output_widget, self.receptor_output = self._save_field(
            "Select receptor PDBQT", "PDBQT (*.pdbqt);;All files (*)"
        )
        receptor_form.addRow("Output", receptor_output_widget)
        receptor_layout.addLayout(receptor_form)
        receptor_button = self._button("Prepare receptor PDBQT", QtWidgets.QStyle.SP_DialogSaveButton)
        receptor_button.clicked.connect(self._run_receptor_pdbqt)
        receptor_layout.addWidget(receptor_button)
        receptor_layout.addStretch(1)
        tabs.addTab(receptor, "Receptor PDBQT")

        validate = QtWidgets.QWidget()
        validate_layout = QtWidgets.QVBoxLayout(validate)
        validate_form = QtWidgets.QFormLayout()
        validate_widget, self.pdbqt_validate_input = self._file_field(
            "Select ligand PDBQT", "PDBQT (*.pdbqt);;All files (*)"
        )
        validate_form.addRow("Input file", validate_widget)
        self.pdbqt_validate_payload = QtWidgets.QPlainTextEdit()
        self.pdbqt_validate_payload.setMinimumHeight(220)
        self.pdbqt_validate_payload.setPlaceholderText("ROOT\n...\nENDROOT\nTORSDOF 0")
        validate_form.addRow("PDBQT text", self.pdbqt_validate_payload)
        validate_layout.addLayout(validate_form)
        validate_button = self._button("Validate torsion tree", QtWidgets.QStyle.SP_DialogApplyButton)
        validate_button.clicked.connect(self._run_pdbqt_validation)
        validate_layout.addWidget(validate_button)
        validate_layout.addStretch(1)
        tabs.addTab(validate, "PDBQT audit")

        protonate = QtWidgets.QWidget()
        protonate_layout = QtWidgets.QVBoxLayout(protonate)
        self.protonate_input = QtWidgets.QPlainTextEdit()
        self.protonate_input.setMinimumHeight(120)
        protonate_layout.addWidget(QtWidgets.QLabel("SMILES"))
        protonate_layout.addWidget(self.protonate_input)
        protonate_button = self._button("Apply pH 7.4 rules", QtWidgets.QStyle.SP_ArrowForward)
        protonate_button.clicked.connect(self._run_protonation)
        protonate_layout.addWidget(protonate_button)
        protonate_layout.addStretch(1)
        tabs.addTab(protonate, "Protonation")

        vina = QtWidgets.QWidget()
        vina_layout = QtWidgets.QVBoxLayout(vina)
        vina_form = QtWidgets.QFormLayout()
        peptide_widget, self.dock_peptide = self._file_field(
            "Select peptide PDB", _PDB_FILTER
        )
        receptor_widget, self.dock_receptor = self._file_field(
            "Select receptor PDB", _PDB_FILTER
        )
        vina_form.addRow("Peptide", peptide_widget)
        vina_form.addRow("Receptor", receptor_widget)
        self.dock_center_mode = QtWidgets.QComboBox()
        self.dock_center_mode.addItem("Whole receptor center", "protein")
        self.dock_center_mode.addItem("Binding-site residues", "site")
        self.dock_center_mode.addItem("Explicit center", "explicit")
        self.dock_center_mode.currentIndexChanged.connect(self._dock_center_mode_changed)
        vina_form.addRow("Center mode", self.dock_center_mode)
        self.dock_residues = QtWidgets.QLineEdit()
        self.dock_residues.setPlaceholderText("45,46,89")
        self.dock_receptor_chain = QtWidgets.QLineEdit()
        self.dock_receptor_chain.setMaximumWidth(90)
        site_row = QtWidgets.QWidget()
        site_layout = QtWidgets.QHBoxLayout(site_row)
        site_layout.setContentsMargins(0, 0, 0, 0)
        site_layout.addWidget(self.dock_residues, 1)
        site_layout.addWidget(QtWidgets.QLabel("Chain"))
        site_layout.addWidget(self.dock_receptor_chain)
        vina_form.addRow("Binding site", site_row)
        self.dock_center_values = [self._coordinate_spin() for _ in range(3)]
        center_row = self._triple_spin_row(self.dock_center_values, ("X", "Y", "Z"))
        vina_form.addRow("Center (A)", center_row)
        self.dock_box_values = [self._coordinate_spin(25.0, 1.0, 200.0) for _ in range(3)]
        box_row = self._triple_spin_row(self.dock_box_values, ("X", "Y", "Z"))
        vina_form.addRow("Box (A)", box_row)
        self.dock_preparation = QtWidgets.QComboBox()
        self.dock_preparation.addItem("Original PDB coordinates", "original_pdb")
        self.dock_preparation.addItem("Audited SMILES", "audited_smiles")
        self.dock_preparation.currentIndexChanged.connect(self._dock_preparation_changed)
        vina_form.addRow("Ligand preparation", self.dock_preparation)
        self.dock_smiles = QtWidgets.QLineEdit()
        self.dock_map = QtWidgets.QLineEdit()
        self.dock_peptide_chain = QtWidgets.QLineEdit("L")
        self.dock_peptide_chain.setMaximumWidth(90)
        self.dock_smiles.setEnabled(False)
        self.dock_map.setEnabled(False)
        ligand_identity = QtWidgets.QWidget()
        identity_layout = QtWidgets.QGridLayout(ligand_identity)
        identity_layout.setContentsMargins(0, 0, 0, 0)
        identity_layout.addWidget(QtWidgets.QLabel("SMILES"), 0, 0)
        identity_layout.addWidget(self.dock_smiles, 0, 1)
        identity_layout.addWidget(QtWidgets.QLabel("MAP"), 1, 0)
        identity_layout.addWidget(self.dock_map, 1, 1)
        identity_layout.addWidget(QtWidgets.QLabel("Peptide chain"), 2, 0)
        identity_layout.addWidget(self.dock_peptide_chain, 2, 1)
        vina_form.addRow("Ligand graph", ligand_identity)
        self.dock_torsdof = QtWidgets.QSpinBox()
        self.dock_torsdof.setRange(-1, 1000)
        self.dock_torsdof.setSpecialValueText("Automatic")
        self.dock_torsdof.setValue(-1)
        self.dock_torsion_ensemble_size = QtWidgets.QSpinBox()
        self.dock_torsion_ensemble_size.setRange(2, 1000)
        self.dock_torsion_ensemble_size.setValue(8)
        self.dock_torsion_threads = QtWidgets.QSpinBox()
        self.dock_torsion_threads.setRange(1, 256)
        self.dock_torsion_threads.setValue(1)
        dock_budget = QtWidgets.QWidget()
        dock_budget_layout = QtWidgets.QHBoxLayout(dock_budget)
        dock_budget_layout.setContentsMargins(0, 0, 0, 0)
        dock_budget_layout.addWidget(QtWidgets.QLabel("TORSDOF"))
        dock_budget_layout.addWidget(self.dock_torsdof)
        dock_budget_layout.addWidget(QtWidgets.QLabel("Ensemble"))
        dock_budget_layout.addWidget(self.dock_torsion_ensemble_size)
        dock_budget_layout.addWidget(QtWidgets.QLabel("Threads"))
        dock_budget_layout.addWidget(self.dock_torsion_threads)
        dock_budget_layout.addStretch(1)
        vina_form.addRow("Torsion budget", dock_budget)
        dock_output_widget, self.dock_output_dir = self._directory_field(
            "Select docking output directory"
        )
        vina_form.addRow("Output", dock_output_widget)
        vina_layout.addLayout(vina_form)
        vina_actions = QtWidgets.QHBoxLayout()
        center_button = self._button("Calculate center", QtWidgets.QStyle.SP_DialogApplyButton)
        center_button.clicked.connect(self._run_center)
        dock_button = self._button("Run Vina", QtWidgets.QStyle.SP_MediaPlay)
        dock_button.clicked.connect(self._run_docking)
        vina_actions.addWidget(center_button)
        vina_actions.addWidget(dock_button)
        vina_actions.addStretch(1)
        vina_layout.addLayout(vina_actions)
        vina_layout.addStretch(1)
        tabs.addTab(vina, "Vina")

        prepared_vina = QtWidgets.QWidget()
        prepared_vina_layout = QtWidgets.QVBoxLayout(prepared_vina)
        prepared_vina_form = QtWidgets.QFormLayout()
        prepared_ligand_widget, self.prepared_vina_ligand = self._file_field(
            "Select prepared ligand PDBQT", _PDBQT_FILTER
        )
        prepared_receptor_widget, self.prepared_vina_receptor = self._file_field(
            "Select prepared receptor PDBQT", _PDBQT_FILTER
        )
        prepared_output_widget, self.prepared_vina_output = self._save_field(
            "Select Vina output PDBQT", _PDBQT_FILTER
        )
        prepared_vina_form.addRow("Ligand", prepared_ligand_widget)
        prepared_vina_form.addRow("Receptor", prepared_receptor_widget)
        prepared_vina_form.addRow("Output", prepared_output_widget)
        self.prepared_vina_center = [self._coordinate_spin() for _ in range(3)]
        self.prepared_vina_box = [
            self._coordinate_spin(25.0, 1.0, 200.0) for _ in range(3)
        ]
        prepared_vina_form.addRow(
            "Center (A)", self._triple_spin_row(self.prepared_vina_center, ("X", "Y", "Z"))
        )
        prepared_vina_form.addRow(
            "Box (A)", self._triple_spin_row(self.prepared_vina_box, ("X", "Y", "Z"))
        )
        self.prepared_vina_exhaustiveness = QtWidgets.QSpinBox()
        self.prepared_vina_exhaustiveness.setRange(1, 100000)
        self.prepared_vina_exhaustiveness.setValue(32)
        self.prepared_vina_modes = QtWidgets.QSpinBox()
        self.prepared_vina_modes.setRange(1, 1000)
        self.prepared_vina_modes.setValue(9)
        prepared_options = QtWidgets.QWidget()
        prepared_options_layout = QtWidgets.QHBoxLayout(prepared_options)
        prepared_options_layout.setContentsMargins(0, 0, 0, 0)
        prepared_options_layout.addWidget(QtWidgets.QLabel("Exhaustiveness"))
        prepared_options_layout.addWidget(self.prepared_vina_exhaustiveness)
        prepared_options_layout.addWidget(QtWidgets.QLabel("Modes"))
        prepared_options_layout.addWidget(self.prepared_vina_modes)
        prepared_options_layout.addStretch(1)
        prepared_vina_form.addRow("Search", prepared_options)
        prepared_vina_layout.addLayout(prepared_vina_form)
        prepared_vina_button = self._button(
            "Run prepared Vina", QtWidgets.QStyle.SP_MediaPlay
        )
        prepared_vina_button.clicked.connect(self._run_prepared_vina)
        prepared_vina_layout.addWidget(prepared_vina_button)
        prepared_vina_layout.addStretch(1)
        tabs.addTab(prepared_vina, "Prepared Vina")

        batch_vina = QtWidgets.QWidget()
        batch_vina_layout = QtWidgets.QVBoxLayout(batch_vina)
        batch_vina_form = QtWidgets.QFormLayout()
        batch_receptor_widget, self.batch_dock_receptor = self._file_field(
            "Select receptor PDB", _PDB_FILTER
        )
        batch_vina_form.addRow("Receptor", batch_receptor_widget)
        batch_directory_widget, self.batch_dock_directory = self._directory_field(
            "Select peptide PDB directory"
        )
        batch_vina_form.addRow("Peptide directory", batch_directory_widget)
        self.batch_dock_center = [self._coordinate_spin() for _ in range(3)]
        self.batch_dock_box = [
            self._coordinate_spin(25.0, 1.0, 200.0) for _ in range(3)
        ]
        batch_vina_form.addRow(
            "Center (A)", self._triple_spin_row(self.batch_dock_center, ("X", "Y", "Z"))
        )
        batch_vina_form.addRow(
            "Box (A)", self._triple_spin_row(self.batch_dock_box, ("X", "Y", "Z"))
        )
        batch_vina_layout.addLayout(batch_vina_form)
        batch_vina_button = self._button("Run batch docking", QtWidgets.QStyle.SP_MediaPlay)
        batch_vina_button.clicked.connect(self._run_batch_docking)
        batch_vina_layout.addWidget(batch_vina_button)
        batch_vina_layout.addStretch(1)
        tabs.addTab(batch_vina, "Batch docking")
        self._dock_center_mode_changed()
        return tabs

    def _build_monomer_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        search = QtWidgets.QHBoxLayout()
        self.monomer_query = QtWidgets.QLineEdit()
        self.monomer_query.setPlaceholderText("Symbol filter")
        refresh = self._button("Search", QtWidgets.QStyle.SP_BrowserReload)
        refresh.clicked.connect(self._run_monomer_list)
        search.addWidget(self.monomer_query, 1)
        search.addWidget(refresh)
        layout.addLayout(search)
        self.monomer_table = QtWidgets.QTableWidget(0, 5)
        self.monomer_table.setHorizontalHeaderLabels(["Symbol", "R1", "R2", "R3", "SMILES"])
        self.monomer_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.monomer_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.monomer_table, 1)

        add_box = QtWidgets.QGroupBox("Register monomer")
        add_form = QtWidgets.QFormLayout(add_box)
        self.monomer_symbol = QtWidgets.QLineEdit()
        self.monomer_smiles = QtWidgets.QLineEdit()
        add_form.addRow("Symbol", self.monomer_symbol)
        add_form.addRow("SMILES", self.monomer_smiles)
        self.monomer_rgroups = [QtWidgets.QLineEdit() for _ in range(3)]
        rgroup_row = QtWidgets.QWidget()
        rgroup_layout = QtWidgets.QHBoxLayout(rgroup_row)
        rgroup_layout.setContentsMargins(0, 0, 0, 0)
        for index, edit in enumerate(self.monomer_rgroups, start=1):
            edit.setPlaceholderText(f"R{index}")
            rgroup_layout.addWidget(edit)
        add_form.addRow("Leaving groups", rgroup_row)
        self.monomer_persist = QtWidgets.QCheckBox("Persist in user library")
        self.monomer_persist.setChecked(True)
        self.monomer_overwrite = QtWidgets.QCheckBox("Overwrite runtime symbol")
        flags = QtWidgets.QWidget()
        flags_layout = QtWidgets.QHBoxLayout(flags)
        flags_layout.setContentsMargins(0, 0, 0, 0)
        flags_layout.addWidget(self.monomer_persist)
        flags_layout.addWidget(self.monomer_overwrite)
        flags_layout.addStretch(1)
        add_form.addRow("Policy", flags)
        add_button = self._button("Register", QtWidgets.QStyle.SP_DialogApplyButton)
        add_button.clicked.connect(self._run_monomer_add)
        add_form.addRow("", add_button)
        layout.addWidget(add_box)
        return page

    def _build_system_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        refresh = self._button("Refresh", QtWidgets.QStyle.SP_BrowserReload)
        refresh.clicked.connect(self._refresh_capabilities)
        layout.addWidget(refresh, 0, QtCore.Qt.AlignLeft)
        self.capability_table = QtWidgets.QTableWidget(0, 2)
        self.capability_table.setHorizontalHeaderLabels(["Component", "Available"])
        self.capability_table.horizontalHeader().setStretchLastSection(True)
        self.capability_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.capability_table, 1)
        self.operation_list = QtWidgets.QListWidget()
        layout.addWidget(QtWidgets.QLabel("Application services"))
        layout.addWidget(self.operation_list, 1)
        return page

    # File and control helpers
    def _button(self, text, standard_icon):
        button = QtWidgets.QPushButton(text)
        button.setIcon(self.style().standardIcon(standard_icon))
        button.setMinimumHeight(30)
        return button

    def _file_field(self, caption, file_filter):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit()
        button = QtWidgets.QToolButton()
        button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        button.setToolTip(caption)
        button.clicked.connect(
            lambda: self._browse_open(edit, caption, file_filter)
        )
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return widget, edit

    def _save_field(self, caption, file_filter):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit()
        button = QtWidgets.QToolButton()
        button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        button.setToolTip(caption)
        button.clicked.connect(
            lambda: self._browse_save(edit, caption, file_filter)
        )
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return widget, edit

    def _directory_field(self, caption):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit()
        button = QtWidgets.QToolButton()
        button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        button.setToolTip(caption)
        button.clicked.connect(lambda: self._browse_directory(edit, caption))
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return widget, edit

    def _browse_open(self, edit, caption, file_filter):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption, "", file_filter)
        if path:
            edit.setText(path)

    def _browse_save(self, edit, caption, file_filter):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, caption, "", file_filter)
        if path:
            edit.setText(path)

    def _browse_directory(self, edit, caption):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, caption)
        if path:
            edit.setText(path)

    def _coordinate_spin(self, value=0.0, minimum=-100000.0, maximum=100000.0):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setValue(value)
        return spin

    def _triple_spin_row(self, spins, labels):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        for label, spin in zip(labels, spins):
            layout.addWidget(QtWidgets.QLabel(label))
            layout.addWidget(spin)
        return widget

    def _arrow_label(self):
        label = QtWidgets.QLabel()
        label.setPixmap(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward).pixmap(18, 18))
        return label

    # Shared worker and result handling
    def _run_service(self, function, *args, callback=None, **kwargs):
        if self._closing:
            return
        worker = ServiceWorker(function, *args, **kwargs)
        self._workers.add(worker)
        self.status.showMessage(f"Running {function.__name__}...")
        worker.result_ready.connect(
            lambda result, cb=callback: self._service_finished(result, cb)
        )
        worker.finished.connect(lambda w=worker: self._release_worker(w))
        worker.start()

    def _release_worker(self, worker):
        self._workers.discard(worker)
        worker.deleteLater()
        if self._closing and not self._workers:
            QtCore.QTimer.singleShot(0, self.close)

    def _service_finished(self, result, callback=None):
        if self._closing:
            return
        self._show_result(result)
        if callback is not None:
            callback(result)
        self.status.showMessage(
            f"{result.get('operation', 'operation')}: {result.get('status', 'unknown')}"
        )

    def _show_result(self, result):
        self._last_result = services.json_ready(result)
        operation = str(result.get("operation", "result"))
        status = str(result.get("status", "unknown"))
        data = result.get("data") or {}
        requested_format_status = (
            data.get("requested_format_status")
            if isinstance(data, dict)
            else None
        )
        display_status = (
            "partial"
            if status == "success"
            and requested_format_status not in {None, "fulfilled"}
            else status
        )
        self.result_title.setText(operation.replace("_", " ").title())
        self._set_status_badge(display_status)
        self._set_rigor_badge(self._find_rigor(self._last_result))
        self.result_json.setPlainText(
            json.dumps(self._last_result, ensure_ascii=False, sort_keys=True, indent=2)
        )
        smiles = self._find_smiles(self._last_result)
        if smiles:
            self._last_smiles = smiles
            self._render_structure(smiles)
        else:
            self._last_smiles = ""
            self.preview_label.setPixmap(QtGui.QPixmap())
            self.preview_label.setText("No molecular structure")

    def _find_smiles(self, value):
        if isinstance(value, dict):
            for key in ("output_smiles", "smiles", "value"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate:
                    if key != "value" or value.get("target_kind") == "smiles":
                        return candidate
            for item in value.values():
                candidate = self._find_smiles(item)
                if candidate:
                    return candidate
        elif isinstance(value, list):
            for item in value:
                candidate = self._find_smiles(item)
                if candidate:
                    return candidate
        return ""

    def _render_structure(self, smiles):
        pixmap = smiles_to_pixmap(
            smiles,
            max(self.preview_label.width() - 8, 320),
            max(self.preview_label.height() - 8, 260),
        )
        if pixmap is None:
            self.preview_label.setPixmap(QtGui.QPixmap())
            self.preview_label.setText("Structure unavailable")
        else:
            self.preview_label.setText("")
            self.preview_label.setPixmap(pixmap)

    def _find_rigor(self, value):
        if isinstance(value, dict):
            for key in ("rigor", "chemical_rigor"):
                rigor = value.get(key)
                if isinstance(rigor, str) and rigor:
                    return rigor
            for item in value.values():
                found = self._find_rigor(item)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = self._find_rigor(item)
                if found:
                    return found
        return None

    def _set_rigor_badge(self, rigor):
        if not rigor:
            self.result_rigor.setVisible(False)
            return
        provenance = rigor.rsplit(":", 1)[-1].upper()
        colors = {
            "Q": ("#009E73", "#ffffff"),
            "R": ("#E69F00", "#000000"),
            "H": ("#0072B2", "#ffffff"),
            "C": ("#999999", "#ffffff"),
            "NONE": ("#4a4a4a", "#ffffff"),
        }
        background, color = colors.get(provenance, colors["NONE"])
        self.result_rigor.setText(rigor)
        self.result_rigor.setVisible(True)
        self.result_rigor.setStyleSheet(
            f"QLabel {{ background: {background}; color: {color}; "
            "border: 1px solid #aab1b8; padding: 2px 8px; }}"
        )

    def _set_status_badge(self, status):
        accepted = {"success", "match"}
        caution = {"partial", "mismatch", "not_supported", "not_comparable"}
        if status in accepted:
            background, color = "#d9f2e3", "#155c35"
        elif status in caution:
            background, color = "#fff0c2", "#6b4b00"
        elif status == "idle":
            background, color = "#e9ecef", "#39434d"
        else:
            background, color = "#f9d8dc", "#7a1f2b"
        self.result_status.setText(status)
        self.result_status.setStyleSheet(
            f"QLabel {{ background: {background}; color: {color}; "
            "border: 1px solid #aab1b8; padding: 2px 8px; }}"
        )

    def _copy_json(self):
        QtWidgets.QApplication.clipboard().setText(self.result_json.toPlainText())
        self.status.showMessage("JSON copied")

    def _save_json(self):
        if not self._last_result:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save result", "cycpep_result.json", "JSON (*.json)"
        )
        if path:
            Path(path).write_text(self.result_json.toPlainText() + "\n", encoding="utf-8")
            self.status.showMessage(f"Saved {path}")

    # Reconstruction
    def _rec_input_mode_changed(self):
        self.rec_input_stack.setCurrentIndex(self.rec_input_mode.currentIndex())
        directory_mode = self.rec_input_mode.currentData() == "directory"
        self.rec_multichain.setEnabled(not directory_mode)
        if directory_mode:
            self.rec_multichain.setChecked(False)

    def _rec_multichain_changed(self, checked):
        self.rec_path.setEnabled(not checked)
        self.rec_chain.setEnabled(not checked)
        self.rec_target_chain.setEnabled(not checked)
        self.rec_admet.setEnabled(not checked)
        self.rec_flexibility.setEnabled(not checked)
        self.rec_docking.setEnabled(not checked)
        self.rec_empty_overlay.setEnabled(not checked)
        self.rec_allow_linear_topology.setEnabled(not checked)
        self.rec_export_dir.setEnabled(not checked)
        self.rec_csv.setEnabled(not checked)

    def _run_reconstruction(self):
        try:
            monomer_context = self._gui_monomer_context(
                self.rec_monomer_context
            )
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_local_error("reconstruct", exc)
            return
        if self.rec_input_mode.currentData() == "directory":
            try:
                inputs = services.discover_coordinate_files(self.rec_directory.text().strip())
            except Exception as exc:
                self._show_local_error("reconstruct", exc)
                return
        else:
            inputs = [self.rec_file.text().strip()] if self.rec_file.text().strip() else []
        geometry_values = self.geometry_params.values()
        self.result_geometry_note.setVisible(
            any(not row.is_default() for row in self.geometry_params.rows)
        )
        if self.rec_multichain.isChecked():
            if len(inputs) != 1:
                self._show_local_error("reconstruct_multichain", ValueError("one input is required"))
                return
            chain_ids = [
                item.strip()
                for item in self.rec_multichain_ids.text().split(",")
                if item.strip()
            ]
            self._run_service(
                services.reconstruct_multichain,
                inputs[0],
                chain_ids=chain_ids or None,
                monomer_context=monomer_context,
                **self._geometry_kwargs(services.reconstruct_multichain, geometry_values),
            )
            return
        self._run_service(
            services.reconstruct_coordinates,
            inputs,
            path=self.rec_path.currentData(),
            chain_id=self.rec_chain.text().strip() or "L",
            target_chain_id=self.rec_target_chain.text().strip() or "R",
            run_admet=self.rec_admet.isChecked(),
            compute_flexibility=self.rec_flexibility.isChecked(),
            run_docking=self.rec_docking.isChecked(),
            export_dir=self.rec_export_dir.text().strip() or None,
            export_format=self.rec_export_format.currentText(),
            csv_output=self.rec_csv.text().strip() or None,
            require_empty_persistent_overlay=self.rec_empty_overlay.isChecked(),
            allow_linear_topology=self.rec_allow_linear_topology.isChecked(),
            monomer_context=monomer_context,
            callback=self._populate_reconstruction_table,
            **self._geometry_kwargs(services.reconstruct_coordinates, geometry_values),
        )

    @staticmethod
    def _geometry_kwargs(function, values):
        """Forward geometry params only when the backend declares them.

        The backend is gaining radius_multiplier/distance_ceiling keyword-only
        arguments across the reconstruction entry points; until that lands,
        forwarding is skipped so the GUI still runs unchanged.
        """
        try:
            params = inspect.signature(function).parameters
        except (TypeError, ValueError):
            return {}
        return {
            name: values[name]
            for name in GEOMETRY_PARAM_FIELDS
            if name in values and name in params
        }

    def _gui_monomer_context(self, override=None):
        text = (
            override.text().strip()
            if override is not None and override.text().strip()
            else self.global_monomer_context.text().strip()
        )
        if not text:
            return None
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("monomer context must be a JSON object")
        return value

    def _populate_reconstruction_table(self, result):
        rows = result.get("data", {}).get("results", [])
        self.rec_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row.get("file", ""),
                row.get("status", ""),
                row.get("support_status", ""),
                str(bool(row.get("qualified_success"))),
                row.get("path_used", ""),
                row.get("cyclization_type", ""),
                row.get("smiles", ""),
            ]
            for column, value in enumerate(values):
                self.rec_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))

    # Representation, audit, and compare
    def _run_conversion(self):
        try:
            monomer_context = self._gui_monomer_context()
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_local_error("convert", exc)
            return
        self._run_service(
            services.convert_representation,
            self.convert_source.currentData(),
            self.convert_target.currentData(),
            self.convert_input.toPlainText().strip(),
            monomer_context=monomer_context,
            callback=self._conversion_finished,
        )

    def _conversion_finished(self, result):
        self.convert_output.setPlainText(str(result.get("data", {}).get("value", "")))

    def _run_audit(self):
        try:
            monomer_context = self._gui_monomer_context()
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_local_error("audit", exc)
            return
        path = self.audit_file.text().strip()
        if path:
            self._run_service(
                services.audit_chemistry,
                self.audit_kind.currentData(),
                input_path=path,
                monomer_context=monomer_context,
            )
        else:
            self._run_service(
                services.audit_chemistry,
                self.audit_kind.currentData(),
                payload=self.audit_payload.toPlainText(),
                monomer_context=monomer_context,
            )

    def _run_comparison(self):
        self._run_service(
            services.compare_chemistry,
            self.compare_left.toPlainText().strip(),
            self.compare_right.toPlainText().strip(),
            mode=self.compare_mode.currentData(),
        )

    # 3D and properties
    def _export_source_changed(self):
        self.export_source_stack.setCurrentIndex(self.export_source_kind.currentIndex())

    def _run_export(self):
        try:
            monomer_context = self._gui_monomer_context()
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_local_error("export", exc)
            return
        source = (
            self.export_smiles.toPlainText().strip()
            if self.export_source_kind.currentData() == "smiles"
            else self.export_coordinate.text().strip()
        )
        self._run_service(
            services.export_structure,
            source,
            self.export_output.text().strip(),
            source_kind=self.export_source_kind.currentData(),
            output_format=self.export_format.currentText(),
            chain_id=self.export_chain.text().strip() or "L",
            path=self.export_path.currentData(),
            fallback_policy=self.export_fallback_policy.currentData(),
            force_field=self.export_forcefield.currentText(),
            num_confs=self.export_num_confs.value(),
            random_seed=self.export_seed.value(),
            monomer_context=monomer_context,
        )

    def _read_named_smiles_manifest(self, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = list(payload.items())
        elif isinstance(payload, list):
            rows = []
            for index, row in enumerate(payload, start=1):
                if not isinstance(row, dict) or "name" not in row or "smiles" not in row:
                    raise ValueError(
                        f"manifest row {index} must contain name and smiles"
                    )
                rows.append((row["name"], row["smiles"]))
        else:
            raise ValueError("SMILES manifest must be a JSON object or list")
        return [(str(name), str(smiles)) for name, smiles in rows]

    def _run_batch_export(self):
        try:
            items = self._read_named_smiles_manifest(
                self.batch_export_manifest.text().strip()
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._show_local_error("batch_export", exc)
            return
        self._run_service(
            services.batch_export_structures,
            items,
            self.batch_export_output_dir.text().strip(),
            output_format=self.batch_export_format.currentText(),
            force_field=self.batch_export_forcefield.currentText(),
        )

    def _run_conformers(self):
        window = self.conf_energy_window.value()
        self._run_service(
            services.conformer_statistics,
            self.conf_smiles.toPlainText().strip(),
            num_confs=self.conf_count.value(),
            random_seed=self.conf_seed.value(),
            force_field=self.conf_forcefield.currentText(),
            optimize=self.conf_optimize.isChecked(),
            energy_window=None if window < 0 else window,
            max_heavy_atoms=(
                None
                if self.conf_max_heavy_atoms.value() == 0
                else self.conf_max_heavy_atoms.value()
            ),
        )

    def _run_template_lookup(self):
        try:
            monomer_context = self._gui_monomer_context()
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_local_error("template_lookup", exc)
            return
        self._run_service(
            services.find_conformer_template,
            self.template_map.toPlainText().strip(),
            template_strategy=self.template_strategy.currentData(),
            random_seed=self.template_seed.value(),
            monomer_context=monomer_context,
        )

    def _run_template_conformers(self):
        try:
            monomer_context = self._gui_monomer_context()
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_local_error("template_conformers", exc)
            return
        self._run_service(
            services.generate_template_conformers,
            self.template_smiles.toPlainText().strip(),
            self.template_map.toPlainText().strip(),
            self.template_output.text().strip(),
            n_conformers=self.template_count.value(),
            template_strategy=self.template_strategy.currentData(),
            random_seed=self.template_seed.value(),
            monomer_context=monomer_context,
        )

    def _run_admet(self):
        values = [line.strip() for line in self.admet_input.toPlainText().splitlines() if line.strip()]
        self._run_service(services.predict_admet, values, callback=self._populate_admet)

    def _populate_admet(self, result):
        rows = result.get("data", {}).get("predictions", [])
        columns = sorted({key for row in rows for key in row})
        self.admet_table.setColumnCount(len(columns))
        self.admet_table.setHorizontalHeaderLabels(columns)
        self.admet_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column, key in enumerate(columns):
                self.admet_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(row.get(key, ""))))

    # PDBQT and docking
    def _run_prepare_sequence(self):
        try:
            optional_objects = {}
            for key, widget in (
                (
                    "stereochemistry",
                    self.sequence_stereochemistry,
                ),
                (
                    "terminal_modifications",
                    self.sequence_terminal_modifications,
                ),
                (
                    "monomer_context",
                    self.sequence_monomer_context,
                ),
            ):
                text = widget.text().strip()
                if not text:
                    optional_objects[key] = None
                    continue
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError(f"{key} must be a JSON object")
                optional_objects[key] = value
            if optional_objects["monomer_context"] is None:
                optional_objects["monomer_context"] = (
                    self._gui_monomer_context()
                )
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_result({
                "operation": "prepare_ligand_from_sequence",
                "status": "invalid_input",
                "data": {},
                "error": str(exc),
            })
            return
        limit = self.sequence_torsdof.value()
        self._run_service(
            services.prepare_ligand_from_sequence,
            self.sequence_input.toPlainText().strip(),
            self.sequence_output_dir.text().strip(),
            cyclization=self.sequence_cyclization.currentData(),
            stereochemistry=optional_objects["stereochemistry"],
            terminal_modifications=optional_objects[
                "terminal_modifications"
            ],
            monomer_context=optional_objects["monomer_context"],
            protonation=self.sequence_protonation.currentData(),
            conformer_count=self.sequence_conformers.value(),
            generate_pdbqt=self.sequence_generate_pdbqt.isChecked(),
            torsdof_limit=None if limit < 0 else limit,
            flexibility_mode=(
                self.sequence_flexibility_mode.currentData()
            ),
            torsion_prior_path=(
                self.sequence_torsion_prior.text().strip() or None
            ),
            template_strategy=(
                self.sequence_template_strategy.currentData()
            ),
            random_seed=self.sequence_seed.value(),
            num_threads=self.sequence_threads.value(),
        )

    def _run_ligand_pdbqt(self):
        try:
            monomer_context = self._gui_monomer_context()
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_local_error("prepare_ligand_pdbqt", exc)
            return
        limit = self.ligand_torsdof.value()
        self._run_service(
            services.prepare_ligand_pdbqt,
            self.ligand_smiles.toPlainText().strip(),
            self.ligand_output.text().strip(),
            generated_map=self.ligand_map.text().strip() or None,
            num_confs=self.ligand_num_confs.value(),
            random_seed=self.ligand_seed.value(),
            rigid_macrocycles=self.ligand_rigid.isChecked(),
            protonate=self.ligand_protonate.isChecked(),
            torsdof_limit=None if limit < 0 else limit,
            torsion_ensemble_size=self.ligand_torsion_ensemble_size.value(),
            torsion_num_threads=self.ligand_torsion_threads.value(),
            monomer_context=monomer_context,
        )

    def _run_ligand_pdbqt_from_pdb(self):
        try:
            monomer_context = self._gui_monomer_context()
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_local_error("prepare_ligand_pdbqt_from_pdb", exc)
            return
        self._run_service(
            services.prepare_ligand_pdbqt_from_pdb,
            self.ligand_pdb_input.text().strip(),
            self.ligand_pdb_output.text().strip(),
            monomer_context=monomer_context,
        )

    def _run_receptor_pdbqt(self):
        self._run_service(
            services.prepare_receptor_pdbqt,
            self.receptor_input.text().strip(),
            self.receptor_output.text().strip(),
        )

    def _run_pdbqt_validation(self):
        input_path = self.pdbqt_validate_input.text().strip()
        if input_path:
            self._run_service(services.validate_pdbqt, input_path=input_path)
        else:
            self._run_service(
                services.validate_pdbqt,
                payload=self.pdbqt_validate_payload.toPlainText(),
            )

    def _run_protonation(self):
        self._run_service(
            services.protonate_smiles, self.protonate_input.toPlainText().strip()
        )

    def _dock_center_mode_changed(self):
        mode = self.dock_center_mode.currentData()
        site = mode == "site"
        explicit = mode == "explicit"
        self.dock_residues.setEnabled(site)
        self.dock_receptor_chain.setEnabled(site)
        for spin in self.dock_center_values:
            spin.setEnabled(explicit)

    def _dock_preparation_changed(self):
        audited = self.dock_preparation.currentData() == "audited_smiles"
        self.dock_smiles.setEnabled(audited)
        self.dock_map.setEnabled(audited)

    def _parsed_residues(self):
        value = self.dock_residues.text().strip()
        if not value:
            return []
        return [int(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]

    def _run_center(self):
        try:
            residues = self._parsed_residues() if self.dock_center_mode.currentData() == "site" else None
        except ValueError as exc:
            self._show_local_error("docking_center", exc)
            return
        self._run_service(
            services.docking_center,
            self.dock_receptor.text().strip(),
            residue_ids=residues,
            chain_id=self.dock_receptor_chain.text().strip() or None,
            callback=self._center_finished,
        )

    def _center_finished(self, result):
        center = result.get("data", {}).get("center")
        if center and len(center) == 3:
            for spin, value in zip(self.dock_center_values, center):
                spin.setValue(float(value))
            self.dock_center_mode.setCurrentIndex(2)

    def _run_docking(self):
        try:
            mode = self.dock_center_mode.currentData()
            residues = self._parsed_residues() if mode == "site" else None
            center = [spin.value() for spin in self.dock_center_values] if mode == "explicit" else None
            monomer_context = self._gui_monomer_context()
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_local_error("dock", exc)
            return
        self._run_service(
            services.dock_structure,
            self.dock_peptide.text().strip(),
            self.dock_receptor.text().strip(),
            center=center,
            binding_site_residues=residues,
            receptor_chain_id=self.dock_receptor_chain.text().strip() or None,
            box_size=[spin.value() for spin in self.dock_box_values],
            output_dir=self.dock_output_dir.text().strip() or None,
            ligand_preparation_mode=self.dock_preparation.currentData(),
            ligand_smiles=self.dock_smiles.text().strip() or None,
            peptide_chain_id=self.dock_peptide_chain.text().strip() or "L",
            generated_map=self.dock_map.text().strip() or None,
            torsdof_limit=None if self.dock_torsdof.value() < 0 else self.dock_torsdof.value(),
            torsion_ensemble_size=self.dock_torsion_ensemble_size.value(),
            torsion_num_threads=self.dock_torsion_threads.value(),
            monomer_context=monomer_context,
        )

    def _run_prepared_vina(self):
        self._run_service(
            services.run_prepared_vina,
            self.prepared_vina_ligand.text().strip(),
            self.prepared_vina_receptor.text().strip(),
            self.prepared_vina_output.text().strip(),
            center=[spin.value() for spin in self.prepared_vina_center],
            box_size=[spin.value() for spin in self.prepared_vina_box],
            exhaustiveness=self.prepared_vina_exhaustiveness.value(),
            num_modes=self.prepared_vina_modes.value(),
        )

    def _run_batch_docking(self):
        try:
            inputs = services.discover_docking_coordinate_files(
                self.batch_dock_directory.text().strip()
            )
            monomer_context = self._gui_monomer_context()
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            self._show_local_error("batch_dock", exc)
            return
        self._run_service(
            services.batch_dock_structures,
            inputs,
            self.batch_dock_receptor.text().strip(),
            center=[spin.value() for spin in self.batch_dock_center],
            box_size=[spin.value() for spin in self.batch_dock_box],
            monomer_context=monomer_context,
        )

    # Monomer registry and capabilities
    def _run_monomer_list(self):
        try:
            monomer_context = self._gui_monomer_context()
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_local_error("monomer_list", exc)
            return
        self._run_service(
            services.list_monomers,
            self.monomer_query.text().strip() or None,
            monomer_context=monomer_context,
            callback=self._populate_monomers,
        )

    def _populate_monomers(self, result):
        rows = result.get("data", {}).get("monomers", [])
        self.monomer_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            rgroups = row.get("rgroups", {})
            values = [row.get("symbol", ""), rgroups.get("R1", ""), rgroups.get("R2", ""), rgroups.get("R3", ""), row.get("smiles", "")]
            for column, value in enumerate(values):
                self.monomer_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))

    def _run_monomer_add(self):
        groups = [edit.text().strip() or None for edit in self.monomer_rgroups]
        self._run_service(
            services.add_monomer,
            self.monomer_symbol.text().strip(),
            self.monomer_smiles.text().strip(),
            r1=groups[0],
            r2=groups[1],
            r3=groups[2],
            overwrite=self.monomer_overwrite.isChecked(),
            persist=self.monomer_persist.isChecked(),
            callback=lambda result: self._run_monomer_list() if result.get("status") == "success" else None,
        )

    def _refresh_capabilities(self):
        self._run_service(services.capabilities, callback=self._populate_capabilities)

    def _populate_capabilities(self, result):
        data = result.get("data", {})
        availability = data.get("availability", {})
        self.capability_table.setRowCount(len(availability))
        for row_index, name in enumerate(sorted(availability)):
            self.capability_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(name))
            self.capability_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem("Yes" if availability[name] else "No"))
        self.operation_list.clear()
        self.operation_list.addItems(data.get("operations", []))

    def _show_local_error(self, operation, error):
        self._show_result(
            {
                "operation": operation,
                "status": "invalid_input",
                "data": {},
                "error": f"{type(error).__name__}: {error}",
            }
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_smiles:
            self._render_structure(self._last_smiles)

    def closeEvent(self, event):
        if self._workers:
            self._closing = True
            self.setEnabled(False)
            self.status.showMessage(
                f"Waiting for {len(self._workers)} background task(s) before closing..."
            )
            event.ignore()
            return
        self._closing = True
        event.accept()
