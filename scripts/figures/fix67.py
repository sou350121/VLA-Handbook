# -*- coding: utf-8 -*-
import sys,math; sys.path.insert(0,'/tmp/svg')
from lib import *
OUT='/sessions/magical-fervent-turing/mnt/KW_VLA/deployment/assets/'

def fig6():
    pts=[("读到 pi0 论文",5,"入坑期"),("3400 元 + MacBook 起步",5,"入坑期"),("没有机械臂玩个锤子",2,"入坑期"),
         ("第一个 ACT 复现成功 激动到流泪",5,"跑通期"),("赛博夹橙子 超级解压",5,"跑通期"),("loss 0.002 以为自己是天才",4,"跑通期"),
         ("上真机 全程 VLA 帕金森",1,"真机期"),("loss 0.002 真机寄",1,"真机期"),("一调一个不吱声",1,"真机期"),
         ("发现是 BGR / RGB 搞反了",2,"真机期"),("纬度是超参数 冷启动也影响",1,"真机期"),
         ("论文 288Hz 实测 14Hz",2,"祛魅期"),("原来大家普遍都是裁缝",2,"祛魅期"),("真机 RL = reward 加权 BC",2,"祛魅期"),
         ("半个行业建立在 Qwen 上",3,"祛魅期"),
         ("不存在没有意义的复现",4,"和解期"),("触觉是残差不是主决策",4,"和解期"),("数据质量远比数量重要",5,"和解期")]
    colors={"入坑期":"#37b24d","跑通期":"#1c7ed6","真机期":"#f03e3e","祛魅期":"#e8590c","和解期":"#7048e8"}
    W=1180; L=78; R=W-40; T=132; B=372
    ncol=3; rows=(len(pts)+ncol-1)//ncol
    H=B+96+rows*22+30
    s=head(W,H,"心态曲线")+defs()
    s.append(txt(24,36,"图 6 · 具身炼丹师的一年（心态曲线 · 全程用黑话标注）",20,"#1f2328",weight="700"))
    s.append(txt(24,60,"纵轴 = 心情（1 想退学 → 5 天下无敌）。曲线的低谷正是黑话最密集的地方——梗的密度 = 痛点的密度",12.5,"#57606a"))
    secs=[]
    for i,(_,_,sec) in enumerate(pts):
        if not secs or secs[-1][0]!=sec: secs.append([sec,i,i])
        else: secs[-1][2]=i
    n=len(pts); dx=(R-L)/(n-1)
    for sec,a,b in secs:
        x1=L+a*dx-dx*0.45; x2=L+b*dx+dx*0.45
        s.append('<rect x="%.1f" y="%d" width="%.1f" height="%d" rx="8" fill="%s" opacity="0.08"/>'%(x1,T-22,x2-x1,B-T+44,colors[sec]))
        s.append(txt((x1+x2)/2,T-30,sec,14,colors[sec],"middle","700"))
    for v in range(1,6):
        y=B-(v-1)*(B-T)/4
        s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e6e8eb" stroke-width="1"/>'%(L-14,y,R,y))
        s.append(txt(L-22,y+4,str(v),12,"#8b949e","end"))
    d=["%.1f,%.1f"%(L+i*dx,B-(v-1)*(B-T)/4) for i,(_,v,_) in enumerate(pts)]
    s.append('<polyline points="%s" fill="none" stroke="#1f6feb" stroke-width="2.6" stroke-linejoin="round"/>'%(' '.join(d)))
    for i,(lab,v,sec) in enumerate(pts):
        x=L+i*dx; y=B-(v-1)*(B-T)/4
        s.append('<circle cx="%.1f" cy="%.1f" r="10" fill="%s" stroke="#fff" stroke-width="2"/>'%(x,y,colors[sec]))
        s.append(txt(x,y+4,str(i+1),11,"#ffffff","middle","700"))
    # legend
    ly=B+72
    s.append(txt(24,ly-20,"图例（编号对应曲线上的点）",13,"#1f2328",weight="700"))
    colw=(W-48)/ncol
    for i,(lab,v,sec) in enumerate(pts):
        c=i//rows; r=i%rows
        x=24+c*colw; y=ly+r*22
        s.append('<circle cx="%.1f" cy="%.1f" r="8" fill="%s"/>'%(x+8,y-4,colors[sec]))
        s.append(txt(x+8,y,str(i+1),10.5,"#ffffff","middle","700"))
        s.append(txt(x+22,y,lab,12.3,"#1f2328"))
    s.append('</svg>')
    open(OUT+'fig6-journey.svg','w',encoding='utf-8').write('\n'.join(s))

def fig7():
    P=[("真机采集就是财务诈骗",.55,.955),("半个行业建立在 Qwen 上",.72,.90),("空调测试",.80,.845),
       ("扫地机器人够好用了吗",.60,.80),("左脚踩右脚涨估值",.32,.885),("大家普遍都是裁缝",.80,.70),
       ("论文 288Hz 实测 14Hz",.90,.60),("真机 RL = reward 加权 BC",.93,.44),("大水池小水管",.85,.33),
       ("不学 L 了",.88,.51),("VLA 帕金森",.68,.28),("柔顺控制棺材板压不住",.57,.21),
       ("星宿老仙",.30,.20),("一调一个不吱声",.44,.13),("大年初一掉进局部最优",.14,.11),
       ("究竟是谁在上岸啊",.10,.34),("赛博夹橙子",.20,.055)]
    W,H=1180,700; L=110; R=760; T=112; B=H-96
    s=head(W,H,"黑话四象限")+defs()
    s.append(txt(24,36,"图 7 · 黑话四象限（技术含量 × 出圈杀伤力）",20,"#1f2328",weight="700"))
    s.append(txt(24,60,"右上角 = 又硬又出圈，想用一句话打动外行就从这里挑",12.5,"#57606a"))
    mx=(L+R)/2; my=(T+B)/2
    for x,y,name,fill,tx,ty,anc in [(L,T,"行业级暴论","#fff0f6",L+10,T+20,"start"),
                                     (mx,T,"出圈金句","#fff9db",R-10,T+20,"end"),
                                     (L,my,"日常怨念","#f1f3f5",L+10,B-10,"start"),
                                     (mx,my,"内行黑话","#e7f5ff",R-10,B-10,"end")]:
        s.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s"/>'%(x,y,(R-L)/2,(B-T)/2,fill))
        s.append(txt(tx,ty,name,13,"#adb5bd",anc,"700"))
    s.append('<rect x="%d" y="%d" width="%.1f" height="%.1f" fill="none" stroke="#d0d7de"/>'%(L,T,R-L,B-T))
    s.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#d0d7de" stroke-dasharray="5 4"/>'%(mx,T,mx,B))
    s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#d0d7de" stroke-dasharray="5 4"/>'%(L,my,R,my))
    s.append(txt((L+R)/2,B+52,"技术含量：纯情绪吐槽 → 硬核技术判据",13,"#57606a","middle","600"))
    s.append('<text x="36" y="%.1f" font-size="13" fill="#57606a" font-weight="600" text-anchor="middle" transform="rotate(-90 36 %.1f)">出圈杀伤力：圈内自嗨 → 出圈</text>'%(my,my))
    s.append(arrow(L,B+22,R,B+22,"#adb5bd",1.4)); s.append(arrow(L-26,B,L-26,T,"#adb5bd",1.4))
    for lab,x,y in P:
        px=L+x*(R-L); py=B-y*(B-T)
        hot = x>.5 and y>.65
        s.append('<circle cx="%.1f" cy="%.1f" r="%d" fill="%s" opacity="0.92"/>'%(px,py,7 if hot else 5,"#f03e3e" if hot else "#1f6feb"))
        right_half = px > mx
        if right_half:
            s.append(txt(px-11,py+4,lab,11.6,"#1f2328" if hot else "#57606a","end","700" if hot else "400"))
        else:
            s.append(txt(px+11,py+4,lab,11.6,"#1f2328" if hot else "#57606a","start","700" if hot else "400"))
    s.append(box(R+40,T,340,164,"#fff5f5","#f03e3e",10))
    s.append(txt(R+58,T+30,"🔥 出圈三件套",14,"#c92a2a",weight="700"))
    for i,t in enumerate(["半个具身行业建立在 Qwen VLM 上","真机采集就是机器人公司的财务诈骗","空调测试：机器人上门装空调才算成"]):
        s.append(txt(R+58,T+62+i*32,"· "+t,12.3,"#c92a2a"))
    s.append(box(R+40,T+186,340,150,"#f6f8fa","#d0d7de",10))
    s.append(txt(R+58,T+216,"读法",13.5,"#1f2328",weight="700"))
    for i,t in enumerate(["红点 = 又硬又出圈，可直接引用","蓝点 = 圈内向，需要背景才好笑","坐标为编者主观标注，非量化数据","完整原话与直链见 §13.1–13.4"]):
        s.append(txt(R+58,T+246+i*26,"· "+t,12,"#57606a"))
    s.append('</svg>')
    open(OUT+'fig7-quadrant.svg','w',encoding='utf-8').write('\n'.join(s))
fig6(); fig7(); print("rebuilt 6,7")
