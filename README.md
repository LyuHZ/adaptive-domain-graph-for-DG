# TransDG
§3  Method
│
├─ §3.1  Framework Overview
│   └─ 四模块总览（无公式）
│
├─ §3.2  Dual-modal Domain Representation  [4 公式]
│   ├─ Prompt Learning & Feature Enhancement
│   │   └─ Eq.1  dual_prompt            Π⁺, Π⁻ 联合定义
│   ├─ Dual-modal Prototype Learning
│   │   ├─ Eq.2  real_proto             P_{c,d}^m = Norm(avg z_i^m)
│   │   ├─ Eq.3  real_proto_confidence   r_{c,d}^m (样本量×方差惩罚)
│   │   └─ Eq.4  class_center_offset     P̄_c^m + ΔP_{c,d}^m (合并)
│
├─ §3.3  Class-Conditioned Domain Expansion  [4 公式]
│   ├─ Inversion-based Domain Extrapolation
│   │   └─ Eq.5  inverse_extrap          ΔP → 逆推出 E 外推偏移
│   ├─ Confidence-based Calibration
│   │   └─ Eq.6  ext_conf                r^ext (γ, θ 门控)
│   ├─ Unified Calibrated Reconstruction
│   │   └─ Eq.7  calibrated_reconstruction   ext+int 统一 ΔP̂ → P̂
│   ├─ Convex Mixing-based Domain Interpolation
│   │   └─ Eq.8  int_conf                r^int (γ, span, θ 门控)
│
├─ §3.4  Cross-modal Graph Construction  [2 公式]
│   ├─ Cross-modal transfer-edge construction
│   │   └─ Eq.9  local_edges              E_{i,c} 边集定义
│   └─ Manifold-aware edge context
│       └─ Eq.10 context_vector           c_{i,p}(type, r, d, δ)
│
├─ §3.5  Graph-Guided Knowledge Propagation  [2 公式]
│   ├─ Context-guided message passing
│   │   └─ Eq.11 contextual_message       e → α → g (注意力+聚合)
│   └─ Node update
│       └─ Eq.12 contextual_update        h ← h + g 残差更新
│
└─ §3.6  Training Objective  [2 公式]
    ├─ Reliability-weighted Classification
    │   └─ Eq.13 class_score              s_ic = log-Σ-exp(r·相似度)
    └─ Overall Loss
        └─ Eq.14 training_objective       L = L_cls + λ·(L_proto + L_align + L_bridge)
`​``

### 公式分布总结

| 模块 | 公式数 | 核心功能 |
|------|:--:|------|
| 3.2 表示 | 4 | prompt → prototype → confidence → offset |
| 3.3 扩充 | 4 | 逆向外推 → 置信度校准 → 内插 → 重建 |
| 3.4 图建 | 2 | 边结构 + context vector |
| 3.5 传播 | 2 | attention 消息 + 残差更新 |
| 3.6 训练 | 2 | s_ic 分类 + L 总损失 |
| **合计** | **14** | |
