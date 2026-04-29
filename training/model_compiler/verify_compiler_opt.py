"""
验证编译器层Sparse优化：对比修改前后的BtpScoreParam.get_score()差异。
无需运行完整编译流程，直接测试score函数即可。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
from components import ComputeNode, FeatureNode, FheParameter, config, N16QP1546H192H32
from score import BtpScoreParam, is_sparse_bootstrapping_param

def make_mock_dag(param_name: str, poly_n: int):
    """构造一个最小DAG，包含一个bootstrapping节点。"""
    dag = nx.DiGraph()
    # Feature input
    inp = FeatureNode('input', dim=2, channel=1)
    # Bootstrap compute node
    btp = ComputeNode(layer_id='btp0', layer_type='bootstrapping', channel_input=1, channel_output=1)
    # Feature output
    out = FeatureNode('output', dim=2, channel=1)
    
    dag.add_node(inp)
    dag.add_node(btp)
    dag.add_node(out)
    dag.add_edge(inp, btp)
    dag.add_edge(btp, out)
    
    # Set required node attributes
    dag.nodes[inp]['pack_num'] = 1
    dag.nodes[inp]['level'] = 0
    dag.nodes[out]['pack_num'] = 1
    dag.nodes[out]['level'] = 0
    
    param = FheParameter(
        name=param_name,
        poly_modulus_degree=poly_n,
        max_level=9,
        log_default_scale=40,
        q=[0x10000000006E0001],
        p=[0x1FFFFFFFFFE00001],
        n_slots=poly_n // 2,
    )
    # Link feature node to parameter
    inp.ckks_parameter_id = 'param0'
    param_dict = {'param0': param}
    return dag, btp, param_dict

def main():
    print("=" * 60)
    print("Compiler Optimization Verification")
    print("=" * 60)
    
    # Test 1: is_sparse_bootstrapping_param heuristic
    print("\n[Test 1] Sparse parameter detection:")
    for name, expected in [
        ("N16QP1546H192H32", True),
        ("N16QP1547H192H32", True),
        ("N15QP768H192H32", True),
        ("N16QP1767H32768H32", False),
        ("PN16QP1761", False),
    ]:
        result = is_sparse_bootstrapping_param(name)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: is_sparse('{name}') = {result} (expected {expected})")
    
    # Test 2: BtpScoreParam score comparison
    print("\n[Test 2] BtpScoreParam score comparison:")
    for label, param_name, poly_n in [
        ("Sparse", "N16QP1546H192H32", 65536),
        ("Dense",  "N16QP1767H32768H32", 65536),  # Dense param name for heuristic test
    ]:
        dag, btp_node, param_dict = make_mock_dag(param_name, poly_n)
        scorer = BtpScoreParam(dag, btp_node, param_dict)
        score = scorer.get_score()
        print(f"  {label:6s} ({param_name}): score = {score:.4f}")
    
    # Test 3: Ratio verification
    print("\n[Test 3] Sparse discount ratio verification:")
    dag_s, btp_s, pd_s = make_mock_dag("N16QP1546H192H32", 65536)
    dag_d, btp_d, pd_d = make_mock_dag("N16QP1767H32768H32", 65536)
    score_s = BtpScoreParam(dag_s, btp_s, pd_s).get_score()
    score_d = BtpScoreParam(dag_d, btp_d, pd_d).get_score()
    ratio = score_s / score_d
    print(f"  Sparse score / Dense score = {ratio:.4f}")
    print(f"  Expected discount = 0.40, actual ratio = {ratio:.4f}")
    if abs(ratio - 0.40) < 0.01:
        print("  PASS: Discount applied correctly")
    else:
        print("  FAIL: Discount mismatch")
    
    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)

if __name__ == '__main__':
    main()
