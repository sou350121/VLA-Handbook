# -*- coding: utf-8 -*-
import sys,math; sys.path.insert(0,'/tmp/svg')
from lib import *
OUT='/sessions/magical-fervent-turing/mnt/KW_VLA/deployment/assets/'

# ---------- FIG 8: timeline ----------
def fig8():
    stages=[("2025 H2","调参期","#37b24d",["VLA 帕金森","变树懒","重新发明低通滤波"],"我的机械臂在抖"),
            ("2026 Q1","祛魅期","#1c7ed6",["不学 L 了","VLA = VA + 营销","长瘤子的 VLM","纬度是超参数","loss 0.002 真机寄"],"语言模态到底有没有用"),
            ("2026 Q2","架构质疑期","#e8590c",["大家普遍都是裁缝","从浆糊里抽一点","大水池小水管","柔顺控制棺材板压不住","不是兄弟真的假的"],"这套架构本身是不是错的"),
            ("2026 H1","泡沫期","#f03e3e",["左脚踩右脚","养蛊","半个行业建立在 Qwen 上","真机采集就是财务诈骗","空调测试"],"这门生意算不算得过来账")]
    W=1180; CW=262; gap=24; H=520
    s=head(W,H,"黑话演化史")+defs()
    s.append(txt(24,36,"图 8 · 黑话演化史：吐槽重心的迁移",20,"#1f2328",weight="700"))
    s.append(txt(24,60,"一年时间，社区的关注点从「我的机械臂在抖」升级到「这门生意算不算得过来账」——梗的迁移就是行业阶段的迁移",12.5,"#57606a"))
    y0=150
    s.append('<line x1="40" y1="%d" x2="%d" y2="%d" stroke="#d0d7de" stroke-width="3"/>'%(y0,W-40,y0))
    x=40
    for i,(per,name,color,items,mood) in enumerate(stages):
        cx=x+CW/2
        s.append('<circle cx="%.1f" cy="%d" r="11" fill="%s" stroke="#fff" stroke-width="3"/>'%(cx,y0,color))
        s.append(txt(cx,y0-30,per,15,color,"middle","700"))
        s.append(txt(cx,y0-12,name,12.5,"#57606a","middle"))
        bh=48+len(items)*22
        s.append(box(x,y0+28,CW,bh,"#ffffff",color,12))
        s.append('<rect x="%.1f" y="%d" width="%.1f" height="26" rx="12" fill="%s" opacity="0.12"/>'%(x,y0+28,CW,color))
        s.append(txt(x+CW/2,y0+46,"关键词",12,color,"middle","700"))
        for k,it in enumerate(items):
            s.append(txt(x+16,y0+70+k*22,"· "+it,12.5,"#1f2328"))
        s.append(box(x,y0+40+bh,CW,54,"#f6f8fa","#d0d7de",10,1))
        s+=boxtext(x,y0+40+bh,CW,54,["这一阶段的核心焦虑"]+wrap(mood,236,12.5),12,"#57606a")
        if i<len(stages)-1:
            s.append(arrow(x+CW+2,y0,x+CW+gap-4,y0,"#adb5bd",2))
        x+=CW+gap
    s.append('</svg>')
    open(OUT+'fig8-timeline.svg','w',encoding='utf-8').write('\n'.join(s))

# ---------- FIG 9: pie ----------
def fig9():
    data=[("模型/架构质疑（裁缝派·不学L派·小脑派）",20,"#f03e3e"),
          ("工程与 Sim2Real 踩坑",19,"#1c7ed6"),
          ("学术圈 / paper factory / 重复发明",17,"#e8590c"),
          ("产业·资本·数采·智驾",17,"#9c36b5"),
          ("科研生活梗",9,"#37b24d"),
          ("评测打假与验收标准",6,"#0c8599"),
          ("真机 RL 与 DAgger 真相",5,"#d6336c")]
    total=sum(d[1] for d in data)
    W,H=1000,470; cx,cy,r,ir=250,250,150,78
    s=head(W,H,"黑话主题分布")+defs()
    s.append(txt(24,36,"图 9 · 93 条黑话的主题分布",20,"#1f2328",weight="700"))
    s.append(txt(24,60,"按 §13 实际词条逐条统计（合计 93 条）",12.5,"#57606a"))
    a0=-math.pi/2
    for name,v,color in data:
        a1=a0+2*math.pi*v/total
        large=1 if (a1-a0)>math.pi else 0
        x1,y1=cx+r*math.cos(a0),cy+r*math.sin(a0); x2,y2=cx+r*math.cos(a1),cy+r*math.sin(a1)
        xi1,yi1=cx+ir*math.cos(a1),cy+ir*math.sin(a1); xi2,yi2=cx+ir*math.cos(a0),cy+ir*math.sin(a0)
        s.append('<path d="M%.2f,%.2f A%d,%d 0 %d,1 %.2f,%.2f L%.2f,%.2f A%d,%d 0 %d,0 %.2f,%.2f Z" fill="%s" stroke="#fff" stroke-width="2"/>'%(x1,y1,r,r,large,x2,y2,xi1,yi1,ir,ir,large,xi2,yi2,color))
        am=(a0+a1)/2; lx,ly=cx+(r+ir)/2*math.cos(am),cy+(r+ir)/2*math.sin(am)
        s.append(txt(lx,ly+5,str(v),15,"#ffffff","middle","700"))
        a0=a1
    s.append(txt(cx,cy-6,"93",34,"#1f2328","middle","700"))
    s.append(txt(cx,cy+18,"条黑话",13,"#57606a","middle"))
    ly=110
    for name,v,color in data:
        s.append('<rect x="470" y="%d" width="16" height="16" rx="4" fill="%s"/>'%(ly-12,color))
        s.append(txt(496,ly+1,name,13,"#1f2328"))
        s.append(txt(966,ly+1,"%d 条 · %.0f%%"%(v,100.0*v/total),12.5,"#57606a","end"))
        ly+=32
    s.append(box(470,ly+6,496,70,"#f6f8fa","#d0d7de",10,1))
    s+=boxtext(470,ly+6,496,70,["架构质疑 + 工程踩坑 ≈ 42%（技术痛点）","学术圈 + 产业资本 ≈ 37%（生态痛点）","社区火力基本对半分"],12.5,"#57606a")
    s.append('</svg>')
    open(OUT+'fig9-pie.svg','w',encoding='utf-8').write('\n'.join(s))
fig8(); fig9(); print("fig8,9 done")
