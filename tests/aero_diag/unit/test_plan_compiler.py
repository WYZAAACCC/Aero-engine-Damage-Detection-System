"""计划编译器单元测试 — DAG验证、环检测、digest完整性"""
import pytest
import sys
sys.path.insert(0, 'src')

from aero_diag.domain.task import TaskState
from aero_diag.orchestration.plan import PlanNode, PlanProposal, PlanCompiler


class TestPlanCompilerDAG:
    """DAG 验证测试 (AER-004)"""

    def test_cycle_detected(self):
        """Kahn算法应检测到循环"""
        nodes = [
            PlanNode(node_id='a', name='A', stage=TaskState.DATA_VALIDATION,
                     depends_on=['b']),
            PlanNode(node_id='b', name='B', stage=TaskState.DETECTION_EXECUTION,
                     depends_on=['a']),
        ]
        compiler = PlanCompiler()
        issues = compiler._validate_dag(nodes)
        cycle_issues = [i for i in issues if 'CYCLE' in i.upper()]
        assert len(cycle_issues) > 0, f"Cycle not detected! Issues: {issues}"

    def test_valid_dag_passes(self):
        """合法DAG应通过验证"""
        nodes = [
            PlanNode(node_id='a', name='Entry', stage=TaskState.DATA_VALIDATION),
            PlanNode(node_id='b', name='Detect', stage=TaskState.DETECTION_EXECUTION,
                     depends_on=['a']),
        ]
        compiler = PlanCompiler()
        issues = compiler._validate_dag(nodes)
        assert len(issues) == 0, f"Valid DAG rejected: {issues}"

    def test_unknown_dependency(self):
        """引用不存在的节点应被检测"""
        nodes = [
            PlanNode(node_id='a', name='A', stage=TaskState.DATA_VALIDATION,
                     depends_on=['nonexistent']),
        ]
        compiler = PlanCompiler()
        issues = compiler._validate_dag(nodes)
        assert len(issues) > 0

    def test_isolated_node_detected(self):
        """孤立节点应被检测"""
        nodes = [
            PlanNode(node_id='a', name='A', stage=TaskState.DATA_VALIDATION),
            PlanNode(node_id='b', name='B', stage=TaskState.DETECTION_EXECUTION),
        ]
        compiler = PlanCompiler()
        issues = compiler._validate_dag(nodes)
        isolated = [i for i in issues if 'ISOLATED' in i.upper()]
        assert len(isolated) >= 2  # Both nodes are isolated

    def test_unreachable_node_detected(self):
        """不可达节点应被检测"""
        nodes = [
            PlanNode(node_id='entry', name='Entry', stage=TaskState.DATA_VALIDATION),
            PlanNode(node_id='reachable', name='R', stage=TaskState.DETECTION_EXECUTION,
                     depends_on=['entry']),
            PlanNode(node_id='orphan1', name='O1', stage=TaskState.CHARACTERIZATION,
                     depends_on=['orphan2']),
            PlanNode(node_id='orphan2', name='O2', stage=TaskState.EVIDENCE_FUSION,
                     depends_on=['orphan1']),
        ]
        compiler = PlanCompiler()
        issues = compiler._validate_dag(nodes)
        unreachable = [i for i in issues if 'UNREACHABLE' in i.upper()]
        assert len(unreachable) > 0

    def test_dag_accepts_standalone_cycle(self):
        """原本通过的独立循环现在应被拒绝"""
        nodes = [
            PlanNode(node_id='a', name='A', stage=TaskState.DATA_VALIDATION),
            PlanNode(node_id='b', name='B', stage=TaskState.DETECTION_EXECUTION,
                     depends_on=['a']),
            PlanNode(node_id='c', name='C', stage=TaskState.CHARACTERIZATION,
                     depends_on=['d']),
            PlanNode(node_id='d', name='D', stage=TaskState.EVIDENCE_FUSION,
                     depends_on=['c']),
        ]
        compiler = PlanCompiler()
        issues = compiler._validate_dag(nodes)
        assert len(issues) > 0, f"Should reject cycle c<->d"


class TestPlanDigest:
    """Plan digest 测试 (AER-005)"""

    def test_digest_changes_with_evidence(self):
        """修改 evidence_requirements 后 digest 应变化"""
        from aero_diag.orchestration.plan import ExecutionPlan
        plan1 = ExecutionPlan(
            task_id='test',
            nodes=[],
            evidence_requirements=['req_a'],
        )
        plan2 = ExecutionPlan(
            task_id='test',
            nodes=[],
            evidence_requirements=['req_a', 'req_b'],
        )
        d1 = plan1.compute_digest()
        d2 = plan2.compute_digest()
        assert d1 != d2, f"Digests should differ: {d1} vs {d2}"

    def test_digest_changes_with_approval(self):
        """修改 approval 后 digest 应变化"""
        from aero_diag.orchestration.plan import ExecutionPlan
        plan1 = ExecutionPlan(task_id='test', nodes=[], approval_required=False)
        plan2 = ExecutionPlan(task_id='test', nodes=[], approval_required=True)
        d1 = plan1.compute_digest()
        d2 = plan2.compute_digest()
        assert d1 != d2, f"Digests should differ with approval change"

    def test_digest_changes_with_task_id(self):
        """修改 task_id 后 digest 应变化"""
        from aero_diag.orchestration.plan import ExecutionPlan
        plan1 = ExecutionPlan(task_id='task_a', nodes=[])
        plan2 = ExecutionPlan(task_id='task_b', nodes=[])
        d1 = plan1.compute_digest()
        d2 = plan2.compute_digest()
        assert d1 != d2, f"Digests should differ with task_id change"
