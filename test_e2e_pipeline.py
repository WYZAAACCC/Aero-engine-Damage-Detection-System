"""端到端诊断流水线集成测试

测试完整链路: 任务创建 → 数据质量门 → 异常检测 → 视觉检测 →
损伤表征 → 知识检索(含跨域校验) → 风险分级 → 复检间隔 → 裂纹扩展/RUL → 状态迁移
"""
import os, sys, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, 'src')

import numpy as np

from aero_diag.domain.task import InspectionTask, TaskState, EngineInfo, OperatingCondition
from aero_diag.orchestration.state_machine import TaskStateMachine
from aero_diag.plugins.official.register import create_runner, register_all_official_assets
from aero_diag.registries import AssetRegistry


def main():
    print('=' * 65)
    print('  AERO-ENGINE DAMAGE DETECTION SYSTEM — E2E PIPELINE TEST')
    print('=' * 65)

    # ━━ INIT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    reg = AssetRegistry()
    registered = register_all_official_assets(reg)
    runner = create_runner(reg)
    n_impl = len(runner._impl_registry)
    print(f'\n[INIT] {registered} assets registered, {n_impl} implementations loaded')

    # ━━ STEP 1: Create Task ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    task = InspectionTask(
        task_id='e2e_test_001',
        title='CFM56-7B HPT Blade Stage 1 Borescope Inspection',
        description='HPT Stage 1 blade borescope: dark linear indication on leading edge, possible crack + coating anomaly.',
        objective='damage_identification_and_severity_assessment',
        engine=EngineInfo(
            engine_type='CFM56-7B', component='HPT_blade',
            component_location='HPT Blade Stage 1 Leading Edge',
            total_hours=15200, total_cycles=8500,
        ),
        operating_conditions=OperatingCondition(mode='cruise', speed_rpm=(8000, 12000)),
        state=TaskState.RECEIVED,
    )
    tsm = TaskStateMachine()
    tsm._current_state = task.state
    tsm.transition(TaskState.DATA_VALIDATION, reason='Input data received', actor='system')
    task.state = tsm.current_state

    print(f'\n[STEP 1] Task: {task.task_id}')
    print(f'  Engine: {task.engine.engine_type} | Component: {task.engine.component}')
    print(f'  Hours: {task.engine.total_hours} | Cycles: {task.engine.total_cycles}')
    print(f'  State: {task.state.value}')

    # ━━ STEP 2: Data Quality Gate ━━━━━━━━━━━━━━━━━━━━━━━━━
    sensor_data = {
        'artifact_id': 'sensor_001', 'artifact_type': 'raw_timeseries',
        'sample_rate': 1.0, 'producer_asset_id': 'data_adapter.timeseries.cmapss@1.0',
        'data_classification': 'internal', 'timestamp': '2026-07-18T10:00:00Z',
        'T2': float(518), 'T24': float(642), 'T30': float(1580), 'T50': float(1400),
        'P2': float(14.5), 'P15': float(8.2), 'Nf': float(100), 'Nc': float(98),
    }
    r = runner.execute('monitor.data_quality.data_quality_gate', inputs=[sensor_data])
    dq = r['result'].structured_output
    print(f'\n[STEP 2] Data Quality Gate:')
    print(f'  Overall: {dq["overall_status"].upper()} P:{dq["passed_count"]} W:{dq["warn_count"]} F:{dq["fail_count"]}')
    print(f'  Recommendation: {dq["recommendation"]}')
    if dq['overall_status'] == 'fail':
        print('  [BLOCKED] Pipeline stopped')
        return

    tsm.transition(TaskState.PLAN_PROPOSAL, reason='Data quality passed', actor='system')
    tsm.transition(TaskState.PLAN_COMPILE, reason='Plan auto-compiled', actor='system')
    tsm.transition(TaskState.DETECTION_EXECUTION, reason='Execution started', actor='system')
    task.state = tsm.current_state

    # ━━ STEP 3: Sensor Anomaly Detection ━━━━━━━━━━━━━━━━━━━
    if_data = {
        'exhaust_temp': 750.0, 'vibration_x': 0.5, 'vibration_y': 0.3,
        'bearing_temp': 120.0, 'inlet_pressure': 14.5, 'lube_oil_pressure': 3.2, 'fuel_flow': 0.8,
    }
    r = runner.execute('detector.scada.isolation_forest_anomaly', inputs=[if_data])
    anom = r['result'].structured_output
    print(f'\n[STEP 3] Sensor Anomaly (Isolation Forest):')
    print(f'  Anomaly: {anom["anomaly_detected"]} | Score: {anom["anomaly_score"]:.3f}')
    print(f'  Sensors: {anom["sensors_used"]}')

    # ━━ STEP 4: Vibration Analysis ━━━━━━━━━━━━━━━━━━━━━━━━
    fs = 10000
    t_sig = np.linspace(0, 0.5, 5000)
    vib_sig = np.sin(2 * np.pi * 300 * t_sig) + 0.1 * np.random.randn(5000)
    r = runner.execute('detector.vibration.wcamba_bearing_fault',
                       inputs=[{'vibration': vib_sig.tolist(), 'sample_rate': fs, 'speed_rpm': 9000}])
    vib = r['result'].structured_output
    print(f'\n[STEP 4] Vibration (WCamba 1D-CNN):')
    print(f'  Fault: {vib["fault_detected"]} | Type: {vib.get("fault_type", "none")}')
    if 'probabilities' in vib:
        print(f'  Probs: { {k: round(v, 3) for k, v in vib["probabilities"].items()} }')

    # ━━ STEP 5: Visual Detection ━━━━━━━━━━━━━━━━━━━━━━━━━━
    img = np.ones((300, 300, 3), dtype=np.uint8) * 180
    img[120:180, 140:155, :] = 40  # dark vertical stripe
    img[115:185, 138:157, :] = np.clip(img[115:185, 138:157, :].astype(np.int16) - 30, 0, 255).astype(np.uint8)
    img = (img.astype(np.int16) + np.random.randint(-10, 10, img.shape)).clip(0, 255).astype(np.uint8)

    r = runner.execute('detector.surface.slf_yolo_metal_defect', inputs=[{'image': img}])
    yolo = r['result'].structured_output
    r2 = runner.execute('detector.crack.sam_adapter_segmentation', inputs=[{'image': img}])
    sam = r2['result'].structured_output
    print(f'\n[STEP 5] Visual Detection:')
    print(f'  SLF-YOLO: {yolo["method"]} | defects: {yolo["defects_found"]}')
    print(f'  SAM-Crack: {sam["method"]} | crack: {sam["crack_detected"]}')

    # ━━ STEP 6: Damage Characterization ━━━━━━━━━━━━━━━━━━━━
    r = runner.execute('characterizer.damage.damage_type_classifier',
                       inputs=[{'phenomenon': 'dark linear indication on HPT blade leading edge, possible crack or coating crack'}])
    dtype = r['result'].structured_output

    r = runner.execute('characterizer.crack.geometry_measurement',
                       inputs=[{'bbox': [138, 115, 157, 185],
                                'scale_info': {'pixel_to_mm': 0.04, 'calibration_method': 'borescope_stereo'}}])
    geom = r['result'].structured_output

    r = runner.execute('characterizer.damage.severity_rater',
                       inputs=[{'damage_type': 'crack',
                                'geometry': {'length_mm': geom.get('length_mm', 2.0)},
                                'component': 'HPT Blade Stage 1'}])
    sev = r['result'].structured_output

    print(f'\n[STEP 6] Characterization:')
    print(f'  Type: {dtype["damage_type"]} ({dtype["confidence"]})')
    print(f'  Geometry: L={geom.get("length_mm")}mm W={geom.get("width_mm")}mm scale_ok={geom["scale_available"]}')
    if not geom['scale_available']:
        print(f'  [WARNING] No scale reference - pixel output only')
    print(f'  Severity: {sev["severity"].upper()} | criteria: {sev["criteria_met"]}')

    tsm.transition(TaskState.CHARACTERIZATION, reason='Characterization complete', actor='system')

    # ━━ STEP 7: Knowledge Retrieval + Cross-Domain ━━━━━━━━━
    r = runner.execute('knowledge_source.maintenance.omin_faa_knowledge',
                       inputs=[{'query': 'HPT blade crack leading edge disposition severity'}],
                       parameters={'component': 'HPT_blade', 'damage': 'crack'})
    kb = r['result'].structured_output
    print(f'\n[STEP 7] Expert Knowledge:')
    print(f'  Results: {kb["results_found"]}/{kb["total_indexed"]} | A:{kb["evidence_summary"]["level_A"]} B:{kb["evidence_summary"]["level_B"]}')
    for item in kb['knowledge_items'][:3]:
        print(f'  [{item["type"]}][L{item["evidence_level"]}] s={item["relevance_score"]} {item["title"][:55]}')

    # Cross-domain check
    r = runner.execute('knowledge_source.maintenance.omin_faa_knowledge',
                       inputs=[{'query': 'TBC coating spallation crack'}],
                       parameters={'component': 'fan_blade'})
    kb_wrong = r['result'].structured_output
    if kb_wrong.get('cross_domain_warnings'):
        print(f'\n  [CROSS-DOMAIN] TBC on fan blade (intentionally wrong):')
        for w in kb_wrong['cross_domain_warnings']:
            print(f'  >>> {w}')
    else:
        print(f'\n  [CROSS-DOMAIN] No warnings (unexpected!)')

    tsm.transition(TaskState.EVIDENCE_FUSION, reason='Knowledge retrieved', actor='system')

    # ━━ STEP 8: Risk Classification ━━━━━━━━━━━━━━━━━━━━━━━
    r = runner.execute('decision_rule.engine.risk_classification',
                       inputs=[{'severity': sev['severity'], 'damage_type': dtype['damage_type'],
                                'component': task.engine.component}])
    risk = r['result'].structured_output
    print(f'\n[STEP 8] Risk: {risk["risk_level"].upper()}')
    print(f'  Actions: {risk["candidate_actions"]}')
    print(f'  Review required: {risk["requires_review"]}')

    # ━━ STEP 9: Inspection Interval ━━━━━━━━━━━━━━━━━━━━━━━
    r = runner.execute('decision_rule.engine.inspection_interval',
                       inputs=[{'risk_level': risk['risk_level'], 'damage_type': dtype['damage_type']}])
    interval = r['result'].structured_output
    print(f'\n[STEP 9] Interval: {interval["recommended_interval_cycles"]} cycles')
    print(f'  Action: {interval["action_type"]} — {interval["description"]}')

    tsm.transition(TaskState.DECISION_DRAFT, reason='Decision generated', actor='system')

    # ━━ STEP 10: Crack Growth & RUL ━━━━━━━━━━━━━━━━━━━━━━━
    crack_len = geom.get('length_mm') or 2.0
    r = runner.execute('reliability_model.crack.py_fatigue_paris_law',
                       parameters={'C': 1e-11, 'm': 3.5, 'initial_crack_mm': crack_len,
                                   'critical_crack_mm': 10.0, 'delta_sigma_mpa': 200.0})
    paris = r['result'].structured_output

    sd = {}
    for ch in ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc',
               'epr', 'Ps30', 'phi', 'NRf', 'NRc']:
        trend = -0.0005 if ch in ('T30', 'Nf', 'Nc') else 0.0002
        sd[ch] = (np.arange(200) * trend + np.random.normal(0, 0.01, 200)).tolist()
    r = runner.execute('reliability_model.rul.cnn_lstm_cmapss', inputs=[sd])
    rul = r['result'].structured_output

    print(f'\n[STEP 10] Crack Growth & RUL:')
    print(f'  Paris: {int(paris["cycles_to_failure"])} cycles (method: {paris["method"]})')
    print(f'  RUL: {rul["rul_cycles"]} cycles CI=[{rul["confidence_interval"][0]}, {rul["confidence_interval"][1]}]')

    # ━━ STEP 11: Final State ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for s in [TaskState.EXPERT_REVIEW, TaskState.APPROVED, TaskState.ARCHIVED]:
        try:
            tsm.transition(s, reason='Pipeline auto-advance', actor='system')
        except Exception:
            pass

    print(f'\n[STEP 11] Final State: {tsm.current_state.value} ({len(tsm.history)} transitions)')

    # ━━ SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print()
    print('=' * 65)
    print('  DIAGNOSTIC PIPELINE COMPLETE')
    print('=' * 65)
    print(f'  Task:    {task.task_id}')
    print(f'  Engine:  {task.engine.engine_type} | {task.engine.component}')
    print(f'  Damage:  {dtype["damage_type"]} | severity: {sev["severity"]} | risk: {risk["risk_level"]}')
    print(f'  Action:  {interval["action_type"]} ({interval["recommended_interval_cycles"]} cycles)')
    print(f'  RUL:     {rul["rul_cycles"]} cycles (baseline)')
    print(f'  Knowledge: {kb["results_found"]} items | cross-domain: checked')
    print(f'  Pipeline: {len(tsm.history)} state transitions')
    print('=' * 65)


if __name__ == '__main__':
    main()
