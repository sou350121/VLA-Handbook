# -*- coding: utf-8 -*-
FONT="-apple-system,BlinkMacSystemFont,'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif"
def w(s,fs):
    t=0
    for ch in s:
        t+= fs*1.0 if ord(ch)>0x2e80 else fs*0.56
    return t
def esc(s):
    return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
def head(W,H,title):
    return ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" font-family="%s">'%(W,H,W,H,FONT),
            '<title>%s</title>'%esc(title),
            '<rect width="%d" height="%d" fill="#ffffff"/>'%(W,H)]
def txt(x,y,s,fs=14,fill="#1f2328",anchor="start",weight="400",op=1):
    return '<text x="%.1f" y="%.1f" font-size="%d" fill="%s" text-anchor="%s" font-weight="%s" opacity="%s">%s</text>'%(x,y,fs,fill,anchor,weight,op,esc(s))
def box(x,y,bw,bh,fill,stroke,r=10,sw=1.5):
    return '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%d" fill="%s" stroke="%s" stroke-width="%s"/>'%(x,y,bw,bh,r,fill,stroke,sw)
def arrow(x1,y1,x2,y2,color="#8b949e",wd=1.8,dash=None):
    d=' stroke-dasharray="4 3"' if dash else ''
    return '<path d="M%.1f,%.1f L%.1f,%.1f" stroke="%s" stroke-width="%s" fill="none" marker-end="url(#ah)"%s/>'%(x1,y1,x2,y2,color,wd,d)
def curve(x1,y1,x2,y2,color="#8b949e",wd=1.8):
    mx=(x1+x2)/2
    return '<path d="M%.1f,%.1f C%.1f,%.1f %.1f,%.1f %.1f,%.1f" stroke="%s" stroke-width="%s" fill="none" marker-end="url(#ah)"/>'%(x1,y1,mx,y1,mx,y2,x2,y2,color,wd)
def defs(color="#8b949e"):
    return ['<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker></defs>'%color]
def wrap(s,maxw,fs):
    out=[];cur=''
    for ch in s:
        if w(cur+ch,fs)>maxw and cur:
            out.append(cur);cur=ch
        else: cur+=ch
    if cur: out.append(cur)
    return out
def boxtext(x,y,bw,bh,lines,fs=13,fill="#1f2328",weight="400"):
    n=len(lines); lh=fs*1.45
    y0=y+bh/2-(n-1)*lh/2+fs*0.35
    return [txt(x+bw/2,y0+i*lh,l,fs,fill,"middle",weight) for i,l in enumerate(lines)]
