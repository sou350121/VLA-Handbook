# -*- coding: utf-8 -*-
import sys,math; sys.path.insert(0,'/tmp/svg')
from lib import *
OUT='/sessions/magical-fervent-turing/mnt/KW_VLA/deployment/assets/'

# (label, mood, phase, 帖号)
P=[("读到 pi0 论文，感觉世界的大门打开了",5,"入坑期",""),
   ("3400 元 + MacBook 就能起步",5,"入坑期",""),
   ("想啥呢？没有机械臂玩个锤子",2,"入坑期",""),
   ("求推荐机械臂，评论区先帮你排除一家",3,"入坑期","帖349"),
   ("第一个 ACT 复现成功，激动到流泪",5,"跑通期","帖347"),
   ("赛博夹橙子，超级解压",5,"跑通期","帖346"),
   ("ALOHA 仿真跑通（Record/Eval 初始化坑除外）",4,"跑通期","帖343"),
   ("loss 0.002，以为自己是天才",5,"跑通期",""),
   ("等等，YCB 物体基本无物理属性可言",3,"跑通期","帖302"),
   ("上真机：全程 VLA 帕金森",1,"真机期","帖172"),
   ("loss 0.002 真机寄",1,"真机期",""),
   ("一调一个不吱声",1,"真机期","帖340"),
   ("机械臂冷启动温度也影响成功率",1,"真机期","帖277"),
   ("查半个月，是 quantile normalization fallback",1,"真机期","帖284"),
   ("原来训练和推理的 BGR / RGB 搞反了",2,"真机期","帖299"),
   ("4090 跑双臂直接 OOM",2,"真机期","帖348"),
   ("论文 288Hz，实测 14Hz",2,"祛魅期",""),
   ("照着论文复现，指标 20% vs 论文一大截",1,"祛魅期","帖292"),
   ("原来大家普遍都是裁缝",2,"祛魅期","帖360"),
   ("真机 RL 本质是 reward 加权的 BC",2,"祛魅期","帖184"),
   ("robotwin 那套评测本来就是分布内",2,"祛魅期","帖303"),
   ("半个具身行业建立在 Qwen VLM 上",3,"祛魅期","帖359"),
   ("不存在没有意义的复现",4,"和解期","帖347"),
   ("数据质量 > 数量：1200 条把 40% 拉到 90%",5,"和解期","帖206"),
   ("触觉要架构隔离：TACO 38% → 82%",4,"和解期","帖307"),
   ("介入式 RL：柔顺插接 30% → 90%",5,"和解期","帖301"),
   ("先把 Sim2Real 四个坑补上，再上 RL",4,"和解期","帖295"),
   ("把历史建成状态：记忆任务 0% → 78%",5,"和解期","帖327")]

PH=[("入坑期","一切皆有可能","#37b24d","我需要买什么"),
    ("跑通期","仿真里我无敌","#1c7ed6","为什么 demo 这么顺"),
    ("真机期","现实开始收利息","#f03e3e","到底是哪一环在骗我"),
    ("祛魅期","论文与真机的鸿沟","#e8590c","这些数字能信吗"),
    ("和解期","知道该信什么了","#7048e8","什么才是真的有用")]
CO={p[0]:p[2] for p in PH}

CALL=[(13,"解药：LeRobot 双臂采集时静止臂统计炸裂，\n用固定值替代静止臂数据 · 帖284",  -1),
      (17,"解药：换成 OOD setting（clean2rand / Libero-plus）\n再看分数，别信分布内评测 · 帖303", 1),
      (14,"解药：保存每次 eval 的输入输出逐一对比，\n先查图像预处理链 · 帖299", 1)]

W=1420; L=92; R=W-46; T=228; B=596
ncol=4; rows=(len(P)+ncol-1)//ncol
H=B+360+rows*23
s=head(W,H,"具身炼丹师的一年 · 心态曲线")+defs()
s.append(txt(26,42,"图 6 · 具身炼丹师的一年：期望 vs 现实（全程用黑话标注）",22,"#1f2328",weight="700"))
s.append(txt(26,68,"纵轴 = 心情（1 想退学 → 5 天下无敌）。灰色虚线是入行时以为的样子，蓝线是真实发生的样子。",13,"#57606a"))
s.append(txt(26,88,"曲线最深的三个坑都配了「解药 + 帖号」——本手册的价值就在这些坑到解药的连线上。",13,"#57606a"))

n=len(P); dx=(R-L)/(n-1)
# phase bands
idx={}
for i,(_,_,ph,_) in enumerate(P): idx.setdefault(ph,[i,i])[1]=i
for name,tag,color,anx in PH:
    a,b=idx[name]
    x1=L+a*dx-dx*0.46; x2=L+b*dx+dx*0.46
    s.append('<rect x="%.1f" y="%d" width="%.1f" height="%d" rx="10" fill="%s" opacity="0.075"/>'%(x1,T-88,x2-x1,B-T+130,color))
    s.append(txt((x1+x2)/2,T-66,name,15.5,color,"middle","700"))
    s.append(txt((x1+x2)/2,T-48,tag,12,color,"middle"))
    s.append(txt((x1+x2)/2,T-28,"“"+anx+"”",11.5,"#868e96","middle"))
# grid
for v in range(1,6):
    y=B-(v-1)*(B-T)/4
    s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e6e8eb" stroke-width="1"/>'%(L-16,y,R,y))
    s.append(txt(L-26,y+4,str(v),12.5,"#8b949e","end"))
# expectation curve
exp=[]
for i in range(n):
    v=4.4+0.6*math.sin(i/ (n-1) * math.pi*0.9)
    exp.append("%.1f,%.1f"%(L+i*dx, B-(v-1)*(B-T)/4))
s.append('<polyline points="%s" fill="none" stroke="#adb5bd" stroke-width="2.2" stroke-dasharray="7 5"/>'%(' '.join(exp)))
s.append(txt(L+2*dx, B-(4.9-1)*(B-T)/4-14,"你以为的样子",12.5,"#868e96","start","700"))
# reality curve
pts=[(L+i*dx, B-(v-1)*(B-T)/4) for i,(_,v,_,_) in enumerate(P)]
s.append('<polyline points="%s" fill="none" stroke="#1f6feb" stroke-width="3" stroke-linejoin="round"/>'%(' '.join("%.1f,%.1f"%p for p in pts)))
s.append(txt(L+9.6*dx, pts[10][1]+30,"真实发生的样子",12.5,"#1f6feb","start","700"))
for i,((lab,v,ph,pid),(px,py)) in enumerate(zip(P,pts)):
    s.append('<circle cx="%.1f" cy="%.1f" r="11" fill="%s" stroke="#fff" stroke-width="2"/>'%(px,py,CO[ph]))
    s.append(txt(px,py+4.5,str(i+1),11.5,"#ffffff","middle","700"))
# callouts -> dedicated "解药" strip below the chart (3 non-overlapping columns)
sy=B+34
s.append(txt(26,sy+2,"三个最深的坑 → 社区给出的解药",13,"#1f2328",weight="700"))
cw=(R-L-2*18)/3
for k,(i,text,_side) in enumerate(CALL):
    px,py=pts[i]
    bx=L+k*(cw+18); by=sy+16; lines=text.split('\n'); bh=len(lines)*19+30
    s.append('<path d="M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f" stroke="#f59f00" stroke-width="1.5" stroke-dasharray="3 3" fill="none"/>'%(px,py+13,px,by-10,bx+cw/2,by))
    s.append(box(bx,by,cw,bh,"#fff9db","#f59f00",9,1.4))
    s.append(txt(bx+14,by+21,"坑 #%d"%(i+1),12,"#e8590c",weight="700"))
    for m,l in enumerate(lines):
        s.append(txt(bx+14,by+41+m*19,l,11.8,"#8a6d00"))

# slang density bar
dy=B+156
s.append(txt(26,dy-14,"黑话密度（每阶段落在词典里的梗数量）",12.5,"#1f2328",weight="700"))
cnt={"入坑期":6,"跑通期":5,"真机期":24,"祛魅期":21,"和解期":7}
mx=max(cnt.values())
for name,tag,color,anx in PH:
    a,b=idx[name]; x1=L+a*dx-dx*0.46; x2=L+b*dx+dx*0.46
    ww=(x2-x1); hh=42*cnt[name]/mx
    s.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="5" fill="%s" opacity="0.75"/>'%(x1,dy+42-hh,ww,hh,color))
    s.append(txt((x1+x2)/2,dy+40-hh-5,str(cnt[name])+" 条",11.5,color,"middle","700"))
s.append(txt(26,dy+72,"→ 梗最密集的地方就是坑最密集的地方：真机期 + 祛魅期贡献了全部黑话的一半以上。",12.5,"#57606a"))

# legend
ly=dy+126
s.append(txt(26,ly-16,"图例（编号对应曲线上的点；带帖号的可在 §12 回溯原帖）",13,"#1f2328",weight="700"))
colw=(W-52)/ncol
for i,(lab,v,ph,pid) in enumerate(P):
    c=i//rows; r=i%rows
    x=26+c*colw; y=ly+8+r*23
    s.append('<circle cx="%.1f" cy="%.1f" r="8.5" fill="%s"/>'%(x+9,y-4,CO[ph]))
    s.append(txt(x+9,y,str(i+1),10.5,"#ffffff","middle","700"))
    t=lab if not pid else lab+"  ("+pid+")"
    s.append(txt(x+24,y,t[:30],11.6,"#1f2328"))
s.append('</svg>')
open(OUT+'fig6-journey.svg','w',encoding='utf-8').write('\n'.join(s))
print("fig6 deep version written")
