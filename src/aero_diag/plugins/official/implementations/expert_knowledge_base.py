"""航空发动机损伤诊断专家知识库 — 严格适用域约束版

每条知识的 applicability 精确限定：部件-损伤-材料-机型-工况-位置
exclusions 明确禁止跨场景误用
Agent 检索时必须返回 applicability_warning 当查询超出知识适用域
"""

KNOWLEDGE_ITEMS = [

    # ══════════════════════════════════════════════════════════════
    # 术语与本体 (terminology) — 适用域宽但定义精确
    # ══════════════════════════════════════════════════════════════
    {
        "knowledge_id": "term_hcf_lcf",
        "type": "terminology",
        "title": "高低周复合疲劳 (Combined HCF/LCF)",
        "content": "LCF(低周疲劳)：起降循环热应力→应变控制→<10⁴次失效。HCF(高周疲劳)：气流激振→应力控制→>10⁶次失效。涡轮叶片两者叠加：每次起降=1 LCF + 持续HCF，疲劳寿命远低于单独模式。",
        "applicability": {
            "component": ["turbine_blade"],
            "damage_type": ["fatigue", "crack"],
        },
        "exclusions": {
            "component": ["bearing", "gear", "combustor_liner"],
            "note": "不适用于轴承/齿轮等非气路部件的疲劳分析",
        },
        "source_ref": "knowledge/航空发动机叶片TC4钛合金振动疲劳裂纹扩展研究_孙宇博.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "term_fod",
        "type": "terminology",
        "title": "外物损伤 (Foreign Object Damage, FOD)",
        "content": "发动机吸入硬质外物(砂石/金属碎片/鸟/冰)导致气路部件损伤。FOD将叶片疲劳强度降低10-50%。注意：FOD仅指冲击损伤——不包括腐蚀/蠕变/涂层退化导致的材料缺失。",
        "applicability": {
            "component": ["fan_blade", "compressor_blade", "inlet_guide_vane"],
            "damage_type": ["FOD", "dent", "nick", "tear"],
            "operating_condition": ["any"],
        },
        "exclusions": {
            "component": ["turbine_blade", "combustor_liner", "nozzle"],
            "note": "涡轮叶片高温环境下的材料缺失通常是涂层剥落/烧蚀/热腐蚀，不属于FOD范畴",
        },
        "source_ref": "knowledge/外物损伤TC17钛合金叶片高周疲劳强度预测方法研究_张钧贺.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "term_tbc",
        "type": "terminology",
        "title": "热障涂层 (TBC) — 仅适用于带涂层的涡轮叶片",
        "content": "YSZ陶瓷涂层(7-8wt%Y₂O₃-ZrO₂)，厚100-500μm。通过EB-PVD或APS工艺涂覆于涡轮叶片/导向叶片表面。TBC仅存在于燃烧室后高温部件——风扇和压气机叶片无TBC。",
        "applicability": {
            "component": ["turbine_blade", "turbine_vane", "combustor_liner"],
            "damage_type": ["coating_spallation"],
            "material": ["nickel_superalloy"],  # 镍基高温合金基体
        },
        "exclusions": {
            "component": ["fan_blade", "compressor_blade", "bearing"],
            "note": "风扇/压气机叶片无TBC；将其涂层问题标记为TBC是错误的",
        },
        "source_ref": "knowledge/CFM56系列手册.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "term_paris",
        "type": "terminology",
        "title": "Paris裂纹扩展定律 — 仅适用于线弹性断裂力学(LEFM)条件",
        "content": "da/dN=C·(ΔK)^m。仅在ΔK_th<ΔK<K_c范围内有效。不适用于：短裂纹(<0.1mm近门槛值)、高温蠕变裂纹(需用C*参数)、腐蚀疲劳裂纹。C/m值随材料变化极大：TC4钛合金 C≈10^-11, m≈3.5；IN718镍基合金 C≈10^-12, m≈3.0。",
        "applicability": {
            "component": ["fan_blade", "compressor_blade", "turbine_blade", "disk"],
            "damage_type": ["crack"],
            "crack_size_range": "0.5mm - critical_size",  # 长裂纹区
        },
        "exclusions": {
            "damage_type": ["creep", "corrosion", "erosion", "coating_spallation"],
            "note": "Paris公式不适用于蠕变/腐蚀/涂层退化——这些不是裂纹扩展机制",
        },
        "source_ref": "knowledge/航空发动机叶片TC4钛合金振动疲劳裂纹扩展研究_孙宇博.pdf",
        "evidence_level": "A",
    },

    # ══════════════════════════════════════════════════════════════
    # 故障机理 (mechanism) — 严格限定部件+材料+工况
    # ══════════════════════════════════════════════════════════════
    {
        "knowledge_id": "mech_creep_hpt",
        "type": "mechanism",
        "title": "HPT叶片蠕变失效 — 仅镍基高温合金涡轮叶片",
        "content": "高压涡轮叶片在>1000°C+离心拉应力下发生蠕变。超温(>1313K)→γ'完全溶解+晶粒异常长大+硬度410→370VHN。注意：此机理仅适用于镍基高温合金(IN738/IN718/René80等)涡轮叶片——不适用于钛合金/铝合金叶片。",
        "applicability": {
            "component": ["HPT_blade", "HPT_vane"],
            "damage_type": ["creep", "elongation"],
            "material": ["nickel_superalloy"],
            "temperature_range": ">1000°C",
            "engine_section": "high_pressure_turbine",
        },
        "exclusions": {
            "component": ["fan_blade", "compressor_blade", "LPT_blade"],
            "material": ["titanium_alloy", "aluminum_alloy"],
            "note": "低压涡轮/压气机/风扇叶片工作温度远低于蠕变阈值；钛合金在>500°C已氧化脆化而非蠕变",
        },
        "source_ref": "knowledge/CFM56系列手册.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "mech_tbc_spallation",
        "type": "mechanism",
        "title": "TBC剥落失效级联 — 仅带TBC的涡轮叶片",
        "content": "TBC剥落→局部热点→基体温度骤升→γ'溶解+蠕变空洞+氧化→壁厚减薄→离心应力下蠕变-疲劳裂纹→断裂。整个过程数十循环内完成。必须是TBC剥落——普通涂层脱落(如防腐涂层)不遵循此级联。",
        "applicability": {
            "component": ["turbine_blade", "turbine_vane"],
            "damage_type": ["coating_spallation"],
            "coating_type": "TBC",
        },
        "exclusions": {
            "component": ["fan_blade", "compressor_blade"],
            "coating_type": ["anti_corrosion", "abradable", "wear_resistant"],
            "note": "非TBC涂层剥落不会引起局部热点——不得引用此级联机理",
        },
        "source_ref": "knowledge/CFM56系列手册.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "mech_erosion_compressor",
        "type": "mechanism",
        "title": "压气机叶片冲蚀 — 仅前缘/叶背，仅含颗粒气流",
        "content": "吸入硬质颗粒(砂尘/火山灰)对叶片前缘和叶背切削磨损。速率∝浓度×速度^2.5×攻角。前缘钝化→气动效率下降→压比降低→喘振裕度减小。多发于沙漠/火山区域运营。",
        "applicability": {
            "component": ["compressor_blade", "fan_blade"],
            "damage_type": ["erosion"],
            "location": ["leading_edge", "pressure_side"],
            "operating_environment": ["desert", "volcanic", "high_dust"],
        },
        "exclusions": {
            "component": ["turbine_blade", "combustor_liner"],
            "note": "涡轮叶片面临的是热腐蚀+氧化，不是机械冲蚀",
        },
        "source_ref": "knowledge/CFM56系列手册.pdf",
        "evidence_level": "B",
    },
    {
        "knowledge_id": "mech_hot_corrosion",
        "type": "mechanism",
        "title": "热腐蚀 — 仅涡轮叶片，仅含硫+盐环境",
        "content": "燃料硫+吸入盐分(Na/K)+高温→熔融硫酸盐→破坏氧化膜。Type I(>871°C)：晶界腐蚀尖峰。Type II(649-871°C)：低温点蚀。沿海/海上平台运营的发动机风险最高——内陆干燥环境无需担心此类腐蚀。",
        "applicability": {
            "component": ["turbine_blade", "turbine_vane"],
            "damage_type": ["corrosion"],
            "operating_environment": ["coastal", "offshore", "marine"],
            "temperature_range": "649-1000°C",
        },
        "exclusions": {
            "component": ["fan_blade", "compressor_blade"],
            "operating_environment": ["inland", "desert", "temperate"],
            "note": "非沿海/海上运营的发动机极少发生热腐蚀——不要误诊为普通氧化",
        },
        "source_ref": "Gas Turbine Engineering Handbook",
        "evidence_level": "B",
    },
    {
        "knowledge_id": "mech_tc4_crack",
        "type": "mechanism",
        "title": "TC4钛合金振动疲劳裂纹 — 仅TC4(Ti-6Al-4V)材料",
        "content": "TC4在振动疲劳下：裂纹从表面缺陷/FOD缺口/加工痕迹萌生→沿α/β相界扩展→Paris区→失稳。此机理和Paris参数(C≈10^-11,m≈3.5,R=0.1)仅对TC4有效——TC17(Ti-5Al-2Sn-2Zr-4Cr-4Mo)和IN718的裂纹扩展行为完全不同。",
        "applicability": {
            "component": ["fan_blade", "compressor_blade"],
            "damage_type": ["crack"],
            "material": ["TC4", "Ti-6Al-4V"],
            "engine_section": ["fan", "compressor"],
        },
        "exclusions": {
            "material": ["TC17", "IN718", "IN738", "nickel_superalloy", "stainless_steel"],
            "note": "Paris参数C=10^-11, m=3.5仅对TC4有效——其他材料需各自的C/m值",
        },
        "source_ref": "knowledge/离心力场对FOD钛合金叶片HCF的影响_贾旭.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "mech_fod_position",
        "type": "mechanism",
        "title": "FOD冲击位置影响 — 仅TC4风扇/压气机叶片",
        "content": "前缘FOD→HCF寿命降低50-70%。叶根FOD→静应力+振动应力叠加→最危险。锐角(30°)比正碰(90°)危害更大。建议前缘FOD叶片直接报废。数据来源：易子程(2024)TC4叶片实验。",
        "applicability": {
            "component": ["fan_blade", "compressor_blade"],
            "damage_type": ["FOD"],
            "material": ["TC4", "Ti-6Al-4V"],
        },
        "exclusions": {
            "material": ["nickel_superalloy", "IN718"],
            "note": "此结论基于TC4钛合金实验——镍基合金的FOD敏感性不同",
        },
        "source_ref": "knowledge/外物损伤冲击位置对TC4叶片振动疲劳强度的影响研究_易子程.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "mech_creep_fatigue_interaction",
        "type": "mechanism",
        "title": "蠕变-疲劳交互 — 仅镍基高温合金涡轮叶片",
        "content": "晶界碳化物膜→晶界脆化→循环载荷沿晶开裂。γ'退化→蠕变加速→又促进疲劳。超温触发恶性循环。此交互作用仅在高温(>800°C)镍基合金中出现——钛合金无此机制。",
        "applicability": {
            "component": ["turbine_blade"],
            "damage_type": ["creep", "fatigue", "crack"],
            "material": ["nickel_superalloy"],
            "temperature_range": ">800°C",
        },
        "exclusions": {
            "material": ["titanium_alloy", "aluminum_alloy"],
            "temperature_range": "<600°C",
            "note": "钛合金无蠕变-疲劳交互——此知识仅适用于高温镍基合金涡轮叶片",
        },
        "source_ref": "knowledge/CFM56系列手册.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "mech_centrifugal_fod",
        "type": "mechanism",
        "title": "离心力对FOD可用极限的影响 — 仅钛合金风扇/压气机叶片",
        "content": "离心拉应力增大平均应力→降低许用疲劳极限。转速越高→相同FOD的可用极限越低。裂纹型FOD(已有微裂纹)→任何转速下均不允许继续使用。研究数据：贾旭(2024)TC4/TC17叶片。",
        "applicability": {
            "component": ["fan_blade", "compressor_blade"],
            "damage_type": ["FOD", "crack"],
            "material": ["TC4", "TC17", "titanium_alloy"],
        },
        "exclusions": {
            "component": ["turbine_blade"],
            "note": "涡轮叶片的可用极限由蠕变+热疲劳决定，离心力不是主要控制因素",
        },
        "source_ref": "knowledge/离心力场对FOD钛合金叶片HCF的影响_贾旭.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "mech_fracture_surface",
        "type": "mechanism",
        "title": "疲劳断口特征 — 涡轮叶片vs压气机叶片需区分",
        "content": "涡轮叶片高温疲劳断口：沿晶断裂为主+氧化变色(源区深色)。压气机叶片HCF断口：穿晶解理+少量韧窝+无氧化色。混淆两者会导致误判失效模式。",
        "applicability": {
            "component": ["turbine_blade", "compressor_blade", "fan_blade"],
            "damage_type": ["crack", "fracture"],
        },
        "exclusions": {
            "note": "断口分析需区分涡轮(沿晶+氧化色)与压气机(穿晶+无氧化)——两者特征相反",
        },
        "source_ref": "民航发动机失效分析(专著)",
        "evidence_level": "A",
    },

    # ══════════════════════════════════════════════════════════════
    # 工程规则 (engineering_rule) — 严格限定机型/部件/条件
    # ══════════════════════════════════════════════════════════════
    {
        "knowledge_id": "rule_lpt_borescope_triggers",
        "type": "engineering_rule",
        "title": "LPT孔探触发条件 — 仅CFM56系列LPT",
        "content": "以下任一事件后必须对CFM56 LPT进行孔探：1)发动机喘振；2)FOD/鸟击；3)N1振动高；4)N1超转；5)超温；6)起动喷火。这6个条件仅适用于CFM56系列LPT——其他机型/发动机段有不同的触发条件。",
        "applicability": {
            "component": ["LPT_blade"],
            "engine_model": ["CFM56-3", "CFM56-5A", "CFM56-5B", "CFM56-5C", "CFM56-7B"],
            "engine_section": "low_pressure_turbine",
        },
        "exclusions": {
            "component": ["HPT_blade", "HPC_blade", "fan_blade"],
            "note": "此规则仅适用于LPT——HPT和HPC有各自的孔探触发条件",
        },
        "source_ref": "knowledge/CFM56系列手册.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "rule_damage_decision_abc",
        "type": "engineering_rule",
        "title": "叶片六级损伤决策 — 仅CFM56风扇/压气机叶片",
        "content": "A=允许不打磨。B=允许打磨。C=超容限报废。E=接近不打磨极限(紧急可飞1次)。F=接近打磨极限。仅平滑压痕允许——切口/尖锐边缘直接报废。此决策表仅适用于CFM56风扇/压气机叶片——涡轮叶片不可打磨修整。",
        "applicability": {
            "component": ["fan_blade", "compressor_blade"],
            "damage_type": ["FOD", "dent"],
            "engine_model": ["CFM56-3", "CFM56-5B", "CFM56-7B"],
            "damage_morphology": "smooth_indentation_only",
        },
        "exclusions": {
            "component": ["turbine_blade", "turbine_vane"],
            "damage_morphology": ["sharp_edge", "cut", "crack", "tear"],
            "note": "涡轮叶片不可打磨修整——任何损伤需工程评估后更换。有尖锐边缘/切口的损伤直接报废，不适用A/B决策。",
        },
        "source_ref": "knowledge/CFM56-7B风扇叶片损伤评判标准.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "rule_hpt_inspection_interval",
        "type": "engineering_rule",
        "title": "HPT孔探检查间隔 — 仅CFM56-7B HPT叶片",
        "content": "CFM56-7B HPT叶片孔探：高循环→3000循环/次；低循环/公务机→5000小时/次。此为CFM56-7B的MPD规定——CFM56-3/-5B有不同的间隔。",
        "applicability": {
            "component": ["HPT_blade"],
            "engine_model": ["CFM56-7B"],
            "engine_section": "high_pressure_turbine",
        },
        "exclusions": {
            "engine_model": ["CFM56-3", "CFM56-5A", "CFM56-5B", "CFM56-5C"],
            "note": "不同CFM56型号的HPT检查间隔不同——必须查对应型号的AMM/MPD",
        },
        "source_ref": "knowledge/CFM56系列手册.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "rule_crack_action",
        "type": "engineering_rule",
        "title": "裂纹处置决策 — 按部件等级区分",
        "content": "HPT叶片任何可见裂纹(>0.5mm)→立即停飞+工程评估。压气机叶片裂纹→AMM容限表，容限内可打磨。风扇叶片裂纹→任何裂纹均不允许，必须更换(CFM56-7B AMM 72-21-01)。三种叶片处置完全不同——不可混用。",
        "applicability": {
            "component": ["HPT_blade", "compressor_blade", "fan_blade"],
            "damage_type": ["crack"],
        },
        "exclusions": {
            "note": "HPT/压气机/风扇三种叶片对裂纹的处置标准完全不同。必须根据具体部件查阅对应AMM章节。",
        },
        "source_ref": "knowledge/CFM56-7B风扇叶片损伤评判标准.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "rule_overtemperature_indicators",
        "type": "engineering_rule",
        "title": "超温判断 — 仅镍基合金涡轮叶片",
        "content": "超温冶金指标：γ'完全溶解(>1313K)、晶粒异常粗大、晶界初熔、硬度降>10%。任一出现→叶片不可继续使用。此判断仅适用于镍基高温合金涡轮叶片——钛合金叶片的超温特征(氧化层颜色变化)完全不同。",
        "applicability": {
            "component": ["turbine_blade"],
            "damage_type": ["burn_mark", "overheating"],
            "material": ["nickel_superalloy"],
        },
        "exclusions": {
            "material": ["titanium_alloy"],
            "note": "钛合金超温表现为氧化层颜色(浅黄→蓝→灰白)，不是γ'溶解。两类材料的超温判断不可互换。",
        },
        "source_ref": "knowledge/CFM56系列手册.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "rule_inspection_interval_method",
        "type": "engineering_rule",
        "title": "初检+复检间隔计算 — 通用方法但材料参数需定制",
        "content": "Weibull失效分布→Bayesian-MCMC更新→Paris反演→设定P_h=0.001→确定初检T_ini和重复间隔ΔT。此方法框架通用，但具体的Weibull参数和Paris C/m值必须使用对应材料的实验数据。TC4的C/m值不能用于IN718叶片。",
        "applicability": {
            "component": ["blade", "disk"],
            "damage_type": ["crack"],
            "method": "damage_tolerance_based",
        },
        "exclusions": {
            "note": "方法框架通用，但输入参数(材料C/m/Weibull参数)必须匹配具体材料和部件",
        },
        "source_ref": "knowledge/虑及高循环疲劳的裂纹型外物损伤叶片的可用极限_贾旭.pdf",
        "evidence_level": "A",
    },

    # ══════════════════════════════════════════════════════════════
    # 标准条款 — 严格限定机型和AMM章节
    # ══════════════════════════════════════════════════════════════
    {
        "knowledge_id": "std_cfm56_7b_fan_72_21_01",
        "type": "standard_clause",
        "title": "CFM56-7B AMM 72-21-01 — 仅CFM56-7B风扇叶片",
        "content": "CFM56-7B风扇转子叶片检查标准：前缘弯曲/缺口/裂纹/叶尖磨损/涂层剥落的可接受限值。A/B/C三级决策。具体限值见AMM表格。注意：此AMM章节号仅适用CFM56-7B——CFM56-3对应AMM 72-21-00，CFM56-5B对应不同的章节。",
        "applicability": {
            "component": ["fan_blade"],
            "engine_model": ["CFM56-7B"],
            "document": "AMM 72-21-01",
        },
        "exclusions": {
            "engine_model": ["CFM56-3", "CFM56-5A", "CFM56-5B", "CFM56-5C"],
            "note": "不同CFM56型号的AMM章节号不同——不能交叉引用",
        },
        "source_ref": "knowledge/CFM56-7B风扇叶片损伤评判标准.pdf",
        "source_location": "AMM 72-21-01",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "std_cfm56_3_fan",
        "type": "standard_clause",
        "title": "CFM56-3 AMM 72-21-00 — 仅CFM56-3风扇叶片",
        "content": "CFM56-3风扇叶片检测：目视→清洗→涡流/渗透→测量L/l/h/a/b→对照AMM评判表→A/B/C/E/F决策。仅平滑压痕允许。",
        "applicability": {
            "component": ["fan_blade"],
            "engine_model": ["CFM56-3"],
            "document": "CFM56-3 AMM Chapter 72",
        },
        "exclusions": {
            "engine_model": ["CFM56-5B", "CFM56-7B"],
            "note": "仅适用于CFM56-3",
        },
        "source_ref": "knowledge/CFM56-3风扇转子叶片检测和评判.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "std_cfm56_5b_repair",
        "type": "standard_clause",
        "title": "CFM56-5B风扇修整程序 — 仅CFM56-5B风扇叶片",
        "content": "打磨深度<0.25mm，修整后Ra≤1.6μm。修整后NDT→重新涂覆→尺寸检验。仅适用于CFM56-5B风扇叶片。",
        "applicability": {
            "component": ["fan_blade"],
            "engine_model": ["CFM56-5B"],
            "damage_type": ["FOD", "dent", "scratch"],
            "document": "CFM56-5B AMM/SRM",
        },
        "exclusions": {
            "component": ["turbine_blade"],
            "engine_model": ["CFM56-3", "CFM56-7B"],
            "note": "打磨深度和粗糙度限值仅适用于CFM56-5B；其他型号有不同限值。涡轮叶片禁止打磨。",
        },
        "source_ref": "knowledge/CFM56-5B风扇转子叶片修整修理程序.pdf",
        "evidence_level": "A",
    },

    # ══════════════════════════════════════════════════════════════
    # 专家经验 — 严格限定场景和条件
    # ══════════════════════════════════════════════════════════════
    {
        "knowledge_id": "exp_borescope_technique",
        "type": "expert_experience",
        "title": "孔探检查操作经验 — 适用于所有叶片孔探",
        "content": "前缘最关键(65%疲劳裂纹)。检查顺序：前缘→叶盆→叶背→后缘→叶尖。侧光45°。可疑区≥3张多角度存档。同排相邻叶片作对比基准。",
        "applicability": {
            "component": ["fan_blade", "compressor_blade", "turbine_blade"],
            "inspection_method": "borescope",
        },
        "exclusions": {
            "note": "仅适用于孔探检查——不适用于分解检查/工业CT/涡流检测",
        },
        "source_ref": "knowledge/CFM56系列手册.pdf",
        "evidence_level": "B",
    },
    {
        "knowledge_id": "exp_vibration_thresholds",
        "type": "expert_experience",
        "title": "振动趋势监控阈值 — 适用于转子/轴承系统",
        "content": "1/rev突变>2×基线→立即检查。1/rev慢升>5%/100循环→计划孔探。BPF增加>3dB→检查FOD/裂纹。次同步振动→检查轴承/封严。多台同时异常→查传感器系统。",
        "applicability": {
            "component": ["rotor", "bearing", "blade"],
            "sensor_type": "vibration_accelerometer",
        },
        "exclusions": {
            "note": "振动阈值用于趋势监控，不是诊断结论——异常需孔探或其他手段确认",
        },
        "source_ref": "Vibration-based Condition Monitoring (Wiley, 2021)",
        "evidence_level": "B",
    },
    {
        "knowledge_id": "exp_fatigue_location",
        "type": "expert_experience",
        "title": "叶片疲劳失效位置分布 — 基于统计分析",
        "content": "叶根榫头40%→前缘根部25%→叶尖15%→缘板R角10%→FOD缺口10%。此分布基于王小庆(2024)统计——具体发动机型号可能有差异。",
        "applicability": {
            "component": ["blade"],
            "damage_type": ["fatigue", "crack"],
        },
        "exclusions": {
            "note": "统计分布仅供参考——实际失效位置需结合具体发动机型号和运行历史判断",
        },
        "source_ref": "knowledge/航空发动机叶片疲劳失效位置分析系统开发及应用_王小庆.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "exp_qar_trends",
        "type": "expert_experience",
        "title": "QAR趋势监控经验 — 适用于装配QAR的发动机",
        "content": "EGTM降>10°C/500循环→性能衰退。FF+3%→效率降。N1/N2振动>1.0 ips→机械异常。滑油耗>0.1 qt/hr→密封/轴承。双发EGT差>30°C→分别孔探。区分渐变(性能)vs突变(机械)。",
        "applicability": {
            "component": ["any"],
            "data_source": "QAR_flight_data",
            "engine_type": "turbofan",
        },
        "exclusions": {
            "note": "QAR趋势监控阈值为经验值——需根据具体发动机型号和运行环境校准",
        },
        "source_ref": "民航发动机失效分析(专著)",
        "evidence_level": "B",
    },
    {
        "knowledge_id": "exp_tc17_fod_hcf",
        "type": "expert_experience",
        "title": "TC17钛合金FOD HCF强度 — 仅TC17压气机叶片",
        "content": "FOD缺口深度与HCF强度负指数关系。K_t随缺口深度增大。Kitagawa-Takahashi图确定可用极限。数据仅对TC17(Ti-5Al-2Sn-2Zr-4Cr-4Mo)有效——不可用于TC4或其他合金。",
        "applicability": {
            "component": ["compressor_blade"],
            "damage_type": ["FOD"],
            "material": ["TC17"],
        },
        "exclusions": {
            "material": ["TC4", "Ti-6Al-4V", "IN718", "nickel_superalloy"],
            "note": "TC17的FOD/HCF关系不可用于TC4——两种钛合金的微观结构(α+β vs near-β)不同",
        },
        "source_ref": "knowledge/外物损伤TC17钛合金叶片高周疲劳强度预测方法研究_张钧贺.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "exp_tc4_fod_position",
        "type": "expert_experience",
        "title": "TC4 FOD位置影响 — 仅TC4风扇/压气机叶片",
        "content": "前缘FOD→疲劳极限降60%。叶根FOD→静应力高→总损伤大。锐角30°冲击>90°正碰。前缘FOD建议直接报废。数据仅对TC4有效。",
        "applicability": {
            "component": ["fan_blade", "compressor_blade"],
            "damage_type": ["FOD"],
            "material": ["TC4", "Ti-6Al-4V"],
        },
        "exclusions": {
            "material": ["TC17", "IN718"],
            "note": "前缘FOD报废建议仅对TC4钛合金叶片——镍基合金/TC17的损伤容限不同",
        },
        "source_ref": "knowledge/外物损伤冲击位置对TC4叶片振动疲劳强度的影响研究_易子程.pdf",
        "evidence_level": "A",
    },
    {
        "knowledge_id": "exp_tc4_paris_parameters",
        "type": "expert_experience",
        "title": "TC4 Paris参数 — 仅TC4(Ti-6Al-4V) R=0.1",
        "content": "TC4振动疲劳Paris参数：C≈10^-11, m≈3.5(R=0.1)。复检间隔取预测寿命1/3(安全系数3)。此参数仅对TC4在R=0.1条件下有效——其他应力比或材料需各自的实验数据。",
        "applicability": {
            "component": ["fan_blade", "compressor_blade"],
            "damage_type": ["crack"],
            "material": ["TC4", "Ti-6Al-4V"],
            "stress_ratio": "R=0.1",
        },
        "exclusions": {
            "material": ["TC17", "IN718", "IN738"],
            "stress_ratio": ["R<0", "R>0.5"],
            "note": "C=10^-11,m=3.5仅对TC4,R=0.1有效——其他条件需重新标定。IN718的C通常为10^-12量级。",
        },
        "source_ref": "knowledge/航空发动机叶片TC4钛合金振动疲劳裂纹扩展研究及剩余寿命预测_孙宇博.pdf",
        "evidence_level": "A",
    },
]
