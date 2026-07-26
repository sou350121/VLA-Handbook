# -*- coding: utf-8 -*-
import sys; sys.path.insert(0,'/tmp/svg')
from lib import *
OUT='/sessions/magical-fervent-turing/mnt/KW_VLA/deployment/assets/'

# ---------- FIG 3: model lineage ----------
def fig3():
    W,H=1080,560
    s=head(W,H,"VLA 模型谱系")+defs()
    s.append(txt(24,34,"图 3 · 社区在用的 VLA 模型谱系",19,"#1f2328",weight="700"))
    s.append(txt(24,56,"虚线汇入处 = 社区最常拿来对照的基线；终点是本手册反复出现的主线矛盾",12.5,"#57606a"))
    fams=[("Physical Intelligence · π 系列","#d0ebff","#1c7ed6",78,
           [("π0","flow matching 出连续动作"),("FAST","动作 tokenizer"),("π0.5","泛化更强 · 社区主力 baseline"),("π0.6","RL 后训练"),("π0.7","reward + goal 扩数据")]),
          ("清华 TSAIL · RDT 系","#d3f9d8","#2f9e44",204,
           [("RDT-1B","双臂扩散基础模型"),("RDT2","UMI 数据 · 零样本跨本体")]),
          ("NVIDIA","#fff3bf","#f59f00",300,
           [("GR00T N1.5 / N1.7","只取第 12 层特征存争议")]),
          ("国产厂商开源","#f3d9fa","#9c36b5",396,
           [("LingBot-VLA","蚂蚁灵波"),("UnifoLM-VLA-0","宇树"),("星海图 G0.5","跨本体 action tokenizer"),("Qwen-VLA","硬件差异写成文本 prompt")]),
          ("轻量 / 社区线","#ffe8cc","#e8590c",492,
           [("ACT","适用桌面机械臂"),("SmolVLA","真机需 1w steps 起步")])]
    for name,fill,stroke,y,items in fams:
        s.append(txt(24,y+4,name,13,stroke,weight="700"))
        x=24
        for i,(t,sub) in enumerate(items):
            bw=max(150, w(t,14)+40, w(sub,10.5)+28)
            s.append(box(x,y+14,bw,52,fill,stroke,10))
            s.append(txt(x+bw/2,y+36,t,14,"#1f2328","middle","700"))
            s.append(txt(x+bw/2,y+53,sub,10.5,"#57606a","middle"))
            if i<len(items)-1: s.append(arrow(x+bw,y+40,x+bw+22,y+40))
            x+=bw+22
    # convergence
    s.append(box(760,300,150,52,"#f6f8fa","#8b949e",10))
    s+=boxtext(760,300,150,52,["复现对照基线"],13,"#1f2328","700")
    for yy in [130,256,326,448,544]:
        s.append('<path d="M700,%d C740,%d 740,326 758,326" stroke="#8b949e" stroke-width="1.4" fill="none" stroke-dasharray="4 3"/>'%(yy,yy))
    s.append(arrow(910,326,948,326))
    s.append(box(830,382,226,74,"#ffe3e3","#f03e3e",10))
    s+=boxtext(830,382,226,74,["§12.4 benchmark 鸿沟","论文分数 ≠ 真机可用"],13,"#c92a2a","700")
    s.append('<path d="M873,352 L873,380" stroke="#f03e3e" stroke-width="1.8" marker-end="url(#ah)"/>')
    s.append('</svg>')
    open(OUT+'fig3-lineage.svg','w',encoding='utf-8').write('\n'.join(s))

# ---------- FIG 4: slang panorama ----------
def fig4():
    cl=[("🪡 动作质量","#ffe3e3","#f03e3e",["VLA 帕金森","重新发明低通滤波","变树懒","smolvla 也是很抖","loss 0.002 真机寄"]),
        ("🔇 Language 不学派","#fff3bf","#f59f00",["不学 L 了","VLA = VA + 营销","VLM 三成功力都没用到","长瘤子的 VLM","vision-guide 才是真理"]),
        ("🧵 裁缝派 架构质疑","#ffe8cc","#e8590c",["大家普遍都是裁缝","从浆糊里抽一点","只有瘤子没有基座","大水池小水管","没有 next token 的 taste"]),
        ("🧠 小脑派之争","#e5dbff","#7048e8",["小脑受损的人","具身也许不需要小脑"]),
        ("📊 数据与过拟合","#d0ebff","#1c7ed6",["监督信号打架","1200 高 > 5688 普通","把人当 GPU 用","外包采数据的🤡"]),
        ("🌡 硬件 trap","#d3f9d8","#2f9e44",["纬度是超参数","珍爱生命远离 Piper","原来 lerobot = 不幸","mujoco 角速度陷阱"]),
        ("🎯 真机 RL 真相","#ffdeeb","#d6336c",["DAgger 拯救了世界","真机 RL = reward 加权 BC","组内物料都被干烂了","黑色孔抠图对准就好了"]),
        ("♻️ 跨代重复发明","#c5f6fa","#0c8599",["每一代人重新发明一遍","柔顺控制棺材板压不住","学计算机的重复发现机械","星宿老仙"]),
        ("📜 Paper factory","#f3d9fa","#9c36b5",["VLA 八股文","开辟新坑继续卷","小农经济学术界","288Hz 实测 14Hz"]),
        ("📈 刷分打假","#fff0f6","#a61e4d",["不是兄弟真的假的","分布内评测不考验泛化"]),
        ("💰 资本泡沫","#fff9db","#e67700",["左脚踩右脚","硬找场景","养蛊","半个行业建立在 Qwen 上","十年前的 VR / 元宇宙"]),
        ("🧪 验收标准梗","#ebfbee","#37b24d",["扫地机器人够好用了吗","空调测试","和工业自动化没本质区别"]),
        ("🎥 数采产业链","#e7f5ff","#1971c2",["真机采集就是财务诈骗","数采账本 支出550收入300","所采即所得"]),
        ("🍊 科研生活","#fff4e6","#f76707",["大年初一掉进局部最优","一调一个不吱声","赛博夹橙子","激动到流泪","究竟是谁在上岸啊"])]
    cols=4; cw=252; gap=14
    rows=(len(cl)+cols-1)//cols
    heights=[]
    for r in range(rows):
        grp=cl[r*cols:(r+1)*cols]
        heights.append(max(len(g[3]) for g in grp)*20+52)
    W=24*2+cols*cw+(cols-1)*gap
    H=104+sum(heights)+(rows-1)*gap
    s=head(W,H,"黑话全景")+defs()
    s.append(txt(24,36,"图 4 · VLA 社区黑话全景（听懂哪几类 = 踩过哪几种坑）",20,"#1f2328",weight="700"))
    s.append(txt(24,60,"14 类 · 93 条词条；完整原话、作者、赞数与直链见 §13.1–13.4",12.5,"#57606a"))
    y=80
    for r in range(rows):
        x=24; bh=heights[r]
        for g in cl[r*cols:(r+1)*cols]:
            name,fill,stroke,items=g
            s.append(box(x,y,cw,bh,fill,stroke,12))
            s.append(txt(x+12,y+22,name,13.5,stroke,weight="700"))
            for i,it in enumerate(items):
                s.append(txt(x+16,y+44+i*20,"· "+it,11.8,"#1f2328"))
            x+=cw+gap
        y+=bh+gap
    s.append('</svg>')
    open(OUT+'fig4-slang-map.svg','w',encoding='utf-8').write('\n'.join(s))

# ---------- FIG 5: decoder ----------
def fig5():
    rows=[("VLA 帕金森 / 重新发明低通滤波","action chunk 衔接与平滑性；控制频率不足需插值","§1 参数 · §12.2 动作生成"),
          ("不学 L 了 / VLA = VA + 营销","语言模态贡献存疑，视觉主导，底座是否可去","§12.3 泛化根因 · §6 少数派"),
          ("π0.5 纯纯过拟合 / 监督信号打架","小数据微调不出泛化；多来源监督互相冲突","§12.5 踩坑 · §12.7 数据"),
          ("纬度是超参数 / 远离 Piper","硬件一致性与标定：温度、零点、仿真约定","§9 选型 · §12.6 部署硬件"),
          ("DAgger 拯救了世界 / 物料被干烂","真机 RL 成本极高，本质常退化为 DAgger","§12.10 RL 后训练"),
          ("VLA 八股文 / 288Hz 实测 14Hz","指标不可复现，论文数字与真机落差","§12.4 benchmark 鸿沟 · §5"),
          ("触觉是残差不是主决策","高频脆弱信号定位错，不该承担主决策","§12.9 触觉与力控"),
          ("左脚踩右脚 / 真机采集是财务诈骗","商业模型与数采成本算不过账","§12.12 产业与逆共识")]
    RH=54; W=1100; H=118+len(rows)*RH
    s=head(W,H,"黑话解码器")+defs()
    s.append(txt(24,36,"图 5 · 黑话解码器（梗 → 真实痛点 → 该读哪节）",20,"#1f2328",weight="700"))
    s.append(txt(24,60,"黑话的价值不在好笑，而在它是真实技术痛点的压缩包",12.5,"#57606a"))
    hs=[("你听到的梗",24,330,"#fff3bf","#f59f00"),("它指向的真实痛点",378,400,"#ffe3e3","#f03e3e"),("去读这里",800,276,"#d3f9d8","#2f9e44")]
    for t,x,bw,fill,stroke in hs:
        s.append(box(x,80,bw,30,fill,stroke,8,1))
        s.append(txt(x+bw/2,100,t,13,stroke,"middle","700"))
    y=120
    for joke,pain,sec in rows:
        s.append(box(24,y,330,RH-8,"#fffbe6","#f0c36d",9,1))
        s+=boxtext(24,y,330,RH-8,wrap(joke,306,12.5),12.5)
        s.append(arrow(354,y+(RH-8)/2,376,y+(RH-8)/2))
        s.append(box(378,y,400,RH-8,"#fff5f5","#f5a3a3",9,1))
        s+=boxtext(378,y,400,RH-8,wrap(pain,376,12.5),12.5)
        s.append(arrow(778,y+(RH-8)/2,798,y+(RH-8)/2))
        s.append(box(800,y,276,RH-8,"#f4fce3","#a9d977",9,1))
        s+=boxtext(800,y,276,RH-8,wrap(sec,256,12),12)
        y+=RH
    s.append('</svg>')
    open(OUT+'fig5-decoder.svg','w',encoding='utf-8').write('\n'.join(s))
fig3(); fig4(); fig5(); print("fig3-5 done")
