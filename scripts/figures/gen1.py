# -*- coding: utf-8 -*-
import sys; sys.path.insert(0,'/tmp/svg')
from lib import *
OUT='/sessions/magical-fervent-turing/mnt/KW_VLA/deployment/assets/'
import os; os.makedirs(OUT,exist_ok=True)

# ---------- FIG 1: doc map ----------
def fig1():
    W,H=1020,470
    s=head(W,H,"文档地图")+defs()
    s.append(txt(24,34,"图 1 · 我该读哪一节（文档地图）",19,"#1f2328",weight="700"))
    s.append(box(24,62,190,60,"#1f6feb","#1f6feb",12))
    s+=boxtext(24,62,190,60,["我现在要","解决什么问题？"],14,"#ffffff","700")
    qs=[("有具体报错 / 想抄参数","#fff3bf","#f59f00",["§1 训练参数","§2 真机部署调试","§12.5 工程踩坑与 bug"]),
        ("要做选型决策","#d0ebff","#1c7ed6",["§9 上游选型","§12.6 硬件与 Sim2Real","§12.7 数据采集"]),
        ("判断模型/论文可不可信","#ffe3e3","#f03e3e",["§5 社区验证 vs 论文声称","§12.4 benchmark 鸿沟","§12.14 信念网络置信度"]),
        ("准备面试","#d3f9d8","#2f9e44",["§12.13 面试题库","§12.1 模型架构演进","§12.2 动作生成范式"]),
        ("了解行业判断与逆共识","#f3d9fa","#9c36b5",["§6 少数派观点","§12.12 产业与逆共识","§13 黑话辞典"])]
    y=62
    for q,fill,stroke,secs in qs:
        s.append(curve(214,92,268,y+28))
        s.append(box(268,y,250,56,fill,stroke,10))
        s+=boxtext(268,y,250,56,wrap(q,228,13),13,"#1f2328","600")
        s.append(arrow(518,y+28,566,y+28))
        s.append(box(566,y,300,56,"#f6f8fa","#d0d7de",10))
        s+=boxtext(566,y,300,56,secs,11.5)
        y+=74
    s.append(box(566,y+6,300,44,"#fff5f5","#f03e3e",10))
    s+=boxtext(566,y+6,300,44,["⚠ 读数字前先看 §12.4：","论文指标与真机复现存在系统性落差"],11.5,"#c92a2a","600")
    s.append('</svg>')
    open(OUT+'fig1-docmap.svg','w',encoding='utf-8').write('\n'.join(s))

# ---------- FIG 2: triage ----------
def fig2():
    rows=[("动作抖动 / 不平滑",["chunk 交接处导数突变 → RTC 三段式 · 帖315","lerobot 版 pi 隐藏 bug → 改用 openpi · 帖279 帖304","位置控制太硬 → 力矩/柔顺控制 · 帖350"]),
          ("抓取偏移 / 不准",["先查标定 / 坐标系 / TCP / 深度尺度 · 帖310 帖349","机械臂冷启动温度影响成功率 · 帖277"]),
          ("越训成功率越低",["训推预处理不一致 BGR/RGB · 帖299","quantile normalization fallback 静止臂统计炸裂 · 帖284"]),
          ("换场景就崩",["数据集碎片化 → spurious features · 帖300 帖239","评测本身是分布内 → 换 OOD setting · 帖303"]),
          ("跑不起来 / OOM / 太慢",["单卡策略：lora / fsdp2 / bs 调小 · 帖348","量化漂移是闭环累积 → 按漂移风险混合精度 · 帖296"]),
          ("长任务中途忘记做到哪",["历史当状态建模：SSM + 二阶动作桥 · 帖327"])]
    RH=32; pad=14
    heights=[max(58, len(c)*RH+18) for _,c in rows]
    H=90+sum(heights)+ (len(rows)-1)*12
    W=1080
    s=head(W,H,"故障诊断树")+defs()
    s.append(txt(24,34,"图 2 · 真机效果不对 → 根因 → 帖号",19,"#1f2328",weight="700"))
    s.append(txt(24,56,"左列是你看到的症状，右列是社区验证过的根因与出处",12.5,"#57606a"))
    y=78
    s.append(box(24,y,120,H-y-20,"#1f6feb","#1f6feb",12))
    s+=boxtext(24,y,120,H-y-20,["真机","效果","不对"],15,"#ffffff","700")
    for (sym,causes),bh in zip(rows,heights):
        s.append(curve(144,y+bh/2,196,y+bh/2))
        s.append(box(196,y,220,bh,"#fff3bf","#f59f00",10))
        s+=boxtext(196,y,220,bh,wrap(sym,200,13.5),13.5,"#1f2328","600")
        cy=y+9
        for c in causes:
            s.append(arrow(416,y+bh/2,458,cy+RH/2-4) if len(causes)>1 else arrow(416,y+bh/2,458,cy+RH/2-4))
            s.append(box(458,cy,598,RH-6,"#f6f8fa","#d0d7de",8,1))
            s+=boxtext(458,cy,598,RH-6,[c],12)
            cy+=RH
        y+=bh+12
    s.append('</svg>')
    open(OUT+'fig2-triage.svg','w',encoding='utf-8').write('\n'.join(s))
fig1(); fig2(); print("fig1,fig2 done")
