"""Rotation-independent junction collars on observed foreground.

Connectivity is defined on eight-neighbour raster paths, including diagonal
strokes. A three-arm collar is a junction at every rotation. Four-arm ink needs
one declared drawing convention; this is never guessed from circuit equations.
"""
import math
import cv2
import numpy as np

CONVENTIONS={'explicit_marks','continuous_ink'}

def collar_arms(ink, x, y, stroke_radius, *, thin=False):
    inner=max(3,int(math.ceil(stroke_radius*2)))
    outer=max(inner+6,int(math.ceil(stroke_radius*5)),12)
    if thin:
        inner=max(inner,outer-max(2,int(math.ceil(2*stroke_radius))))
    h,w=ink.shape
    left,top=max(0,x-outer-1),max(0,y-outer-1)
    right,bottom=min(w,x+outer+2),min(h,y+outer+2)
    yy,xx=np.mgrid[top:bottom,left:right]
    radius=np.hypot(xx-x,yy-y)
    ring=((radius>=inner)&(radius<=outer)&(ink[top:bottom,left:right]!=0)).astype(np.uint8)
    count,labels=cv2.connectedComponents(ring,connectivity=8)
    arms=[]
    for i in range(1,count):
        ys,xs=np.nonzero(labels==i)
        r=radius[ys,xs]
        if not (np.any(r<=inner+2) and np.any(r>=outer-2)):
            continue
        edge=r>=outer-2
        dx=float(np.mean(xs[edge]+left-x));dy=float(np.mean(ys[edge]+top-y))
        length=math.hypot(dx,dy)
        if length>inner:
            occupied=np.column_stack((xs[edge]+left,ys[edge]+top))
            index=int(np.argmin(np.sum((occupied-np.array([x+dx,y+dy]))**2,axis=1)))
            arms.append({'vector':[dx/length,dy/length],
                'point_px':(occupied[index].astype(float)+.5).tolist()})
    return sorted(arms,key=lambda a:math.atan2(a['vector'][1],a['vector'][0]))


def junction_kind(arms, *, filled_dot, convention):
    if convention not in CONVENTIONS:
        raise ValueError('Unknown junction convention')
    count=len(arms)
    if filled_dot and count>=3:return 'filled_dot'
    if count==3:return 'continuous_three_way'
    if count==4 and convention=='continuous_ink':return 'continuous_four_way'
    if count<=2:return 'continuous_bend_or_end'
    return None


def within_complete_junction(point, region, events, regions):
    """A thinning endpoint inside a fully observed joined collar is not a gap.

    Every native arm must already belong to the SAME predicted foreground
    region. Missing arms or a second region prevent suppression.
    """
    h,w=regions.shape
    for event in events:
        arms=event.get('observed_arms',[])
        if event['state']!='joined' or len(arms)<3 or math.dist(point,event['point_px'])>12:
            continue
        bound=[]
        for arm in arms:
            x,y=[int(v) for v in arm['point_px']]
            occupied=set(map(int,regions[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)].ravel()))-{0}
            bound.append(occupied=={region})
        if all(bound):return True
    return False
