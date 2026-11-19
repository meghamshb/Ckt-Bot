"""Associate skeleton ends with measured lead contacts using occupied paths.

No net is joined here. An endpoint is a contact only when the contact lies
inside the measured stroke footprint, or behind its continuing tangent, and
a short local SAM path actually connects it.
The corridor width scales with the observed stroke; offset length alone is
not a rejection criterion. A missing pixel path still prevents association.
"""
import math
from collections import deque


def contact_endpoint_proof(point, region, contacts, regions, neighbors, distance):
    x,y=[int(v) for v in point]
    origin=(x,y);path=[origin]
    while len(path)<8:
        options=[p for p in neighbors.get(path[-1],[]) if len(path)<2 or p!=path[-2]]
        if len(options)!=1:break
        path.append(options[0])
    if len(path)<2:return None
    width=2*max(float(distance[b,a]) for a,b in path)
    last=path[min(len(path)-1,max(1,int(math.ceil(width))))]
    tangent=(last[0]-x,last[1]-y);norm=math.hypot(*tangent)
    if not norm:return None
    h,w=regions.shape
    for contact,owner_region in contacts:
        if owner_region!=region:continue
        dx,dy=contact[0]-point[0],contact[1]-point[1];offset=math.hypot(dx,dy)
        if offset>2*width and (dx*tangent[0]+dy*tangent[1])/(offset*norm)>-.25:continue
        target=(int(contact[0]),int(contact[1]))
        if not (0<=target[0]<w and 0<=target[1]<h) or regions[target[1],target[0]]!=region:continue
        queue=deque([origin]);previous={origin:None};found=False
        while queue:
            current=queue.popleft()
            if current==target:found=True;break
            for a in (-1,0,1):
                for b in (-1,0,1):
                    p=(current[0]+a,current[1]+b)
                    if not(a or b) or p in previous or not(0<=p[0]<w and 0<=p[1]<h):continue
                    if regions[p[1],p[0]]!=region:continue
                    vx,vy=p[0]+.5-point[0],p[1]+.5-point[1]
                    if offset:
                        along=(vx*dx+vy*dy)/offset
                        across=abs(vx*dy-vy*dx)/offset
                        if not -width<=along<=offset+width or across>width:continue
                    previous[p]=current;queue.append(p)
        if not found:continue
        trail=[];current=target
        while current is not None:trail.append([current[0]+.5,current[1]+.5]);current=previous[current]
        return {'point_px':list(point),'contact_px':list(contact),'observed_path_px':trail[::-1],
            'stroke_width_px':width,'offset_stroke_widths':offset/width,
            'alignment':'contact_footprint' if offset<=2*width else 'opposes_tangent',
            'evidence':'same_native_region_local_contact_corridor'}
    return None


def observed_multiarm_endpoint(point, region, source, regions, stroke_radius):
    """Register medial-axis spurs only inside a connected source stroke footprint."""
    from .wire_junctions import collar_arms
    import cv2
    import numpy as np
    x,y=map(int,point);h,w=regions.shape
    reach=max(12,int(math.ceil(5*stroke_radius)))
    neighbourhood=source[max(0,y-reach):min(h,y+reach+1),max(0,x-reach):min(w,x+reach+1)]
    feature_radius=max(stroke_radius,float(cv2.distanceTransform(neighbourhood.astype(np.uint8),cv2.DIST_L2,3).max()))
    arms=collar_arms(source,x,y,feature_radius,thin=True)
    if len(arms)<2:return None
    radius=max(12,int(math.ceil(5*feature_radius)))+2
    left,top=max(0,x-radius),max(0,y-radius);right,bottom=min(w,x+radius+1),min(h,y+radius+1)
    native=source[top:bottom,left:right].astype(np.uint8)
    _,labels=cv2.connectedComponents(native,8);groups=[]
    for arm in arms:
        a,b=[int(v) for v in arm['point_px']];a-=left;b-=top
        occupied=set(regions[max(0,b+top-1):min(h,b+top+2),max(0,a+left-1):min(w,a+left+2)].ravel())-{0}
        if occupied!={region}:return None
        choices=set(labels[max(0,b-1):min(bottom-top,b+2),max(0,a-1):min(right-left,a+2)].ravel())-{0}
        if len(choices)!=1:return None
        groups.extend(choices)
    if len(set(groups))!=1:return None
    return {'point_px':list(point),'source_arm_count':len(arms),'source_feature_radius_px':feature_radius,'evidence':'same_native_stroke_footprint_medial_axis_spur','arms':arms}
