"""Trace source-observed crossover curves in continuous orientation space."""

import cv2
import numpy as np


def normal(axis):
    return np.array([-axis[1], axis[0]])



def samples(image, points):
    x, y = np.rint(points).astype(int).T
    return image[y, x]


def verified_crossover(ink, blocked, crossing):
    """Fit two observed axes plus a curved detour at arbitrary angle and scale.

    The curve and all four leads must be occupied in the source image. A
    declared junction convention never overrides an observed bridge curve.
    """
    x,y=crossing;h,w=ink.shape
    width=cv2.distanceTransform(ink,cv2.DIST_L2,3)
    stroke=max(1.,float(width[y,x]))
    reach=min(96,max(24,int(np.ceil(28*stroke))))
    left,top=max(0,x-reach),max(0,y-reach);right,bottom=min(w,x+reach+1),min(h,y+reach+1)
    patch=ink[top:bottom,left:right].astype(np.uint8);origin=np.array([x-left,y-top],float)
    lines=cv2.HoughLinesP(patch,1,np.pi/360,max(5,round(4*stroke)),minLineLength=max(6,round(6*stroke)),maxLineGap=max(1,round(stroke)))
    if lines is None:return None
    groups=[]
    for a,b,c,d in lines.reshape(-1,4):
        endpoints=np.array([[a,b],[c,d]],float)-origin;v=endpoints[1]-endpoints[0];v/=np.linalg.norm(v)
        if v[0]<0:v=-v
        offset=float(endpoints.mean(axis=0)@normal(v))
        group=next((g for g in groups if abs(g['axis']@v)>.99 and abs(g['offset']-offset)<2*stroke),None)
        if group is None:groups.append({'axis':v,'offset':offset,'points':[endpoints]})
        else:group['points'].append(endpoints)
    axes=[]
    for group in groups:
        points=np.concatenate(group['points']);dx,dy,cx,cy=cv2.fitLine(points.astype(np.float32),cv2.DIST_L2,0,.01,.01).ravel();v=np.array([dx,dy],float)
        if v[0]<0:v=-v
        axes.append((v,float(np.array([cx,cy])@normal(v))))
    if len(axes)>24:return None
    empty=cv2.distanceTransform(1-patch,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    yy,xx=np.nonzero(patch);points=np.column_stack((xx,yy))-origin
    solutions=[]
    def occupied(path):
        pix=np.rint(path+origin).astype(int)
        if (pix<0).any() or (pix[:,0]>=patch.shape[1]).any() or (pix[:,1]>=patch.shape[0]).any():return False
        return bool(np.max(empty[pix[:,1],pix[:,0]])<=max(1.15,.9*stroke))
    for under,offset in axes:
        if abs(offset)>2*stroke:continue
        for over,over_offset in axes:
            if abs(under@over)>.94 or abs(over_offset)<2*stroke:continue
            center=np.linalg.solve([normal(under),normal(over)],[offset,over_offset])
            estimate=float(np.linalg.norm(center))
            if estimate<3*stroke or estimate>reach*.6:continue
            transverse=normal(over);direction=1 if (-center)@transverse>0 else -1
            for radius in np.arange(max(3*stroke,estimate-1.5*stroke),estimate+1.5*stroke+.1,.5*stroke):
                theta=np.linspace(0,np.pi,max(25,round(6*radius)))
                arc=center+radius*(np.cos(theta[:,None])*over+direction*np.sin(theta[:,None])*transverse)
                straight=center+np.linspace(-radius-5*stroke,radius+5*stroke,max(30,round(6*radius)))[:,None]*under
                limbs=np.array([under,-under,over,-over]);paths=center+limbs[:,None,:]*(radius+np.linspace(2.5*stroke,5*stroke,8))[None,:,None]
                if not all(occupied(path) for path in [arc,straight,paths.reshape(-1,2)]):continue
                relative=points-center;distance=np.linalg.norm(relative,axis=1);along=relative@over;side=relative@transverse
                modeled=np.minimum.reduce([abs(relative@normal(under)),np.where(abs(along)>=radius-stroke,abs(side),np.inf),np.where(direction*side>=-stroke,abs(distance-radius),np.inf)])
                near=distance<radius+2*stroke
                if not near.any() or np.max(modeled[near])>2*stroke:continue
                cy,cx=np.ogrid[:patch.shape[0],:patch.shape[1]];center_px=center+origin;rad=np.hypot(cx-center_px[0],cy-center_px[1]);cut=rad<=radius+2*stroke;collar=(rad>radius+2*stroke)&(rad<=radius+5*stroke)
                if np.any(blocked[top:bottom,left:right]&(cut|collar)):continue
                _,labels=cv2.connectedComponents((patch&collar.astype(np.uint8)).astype(np.uint8),8)
                arms=center_px+limbs*(radius+4*stroke)
                choices=[]
                for a,b in arms:
                    i,j=round(a),round(b)
                    if not(1<=i<patch.shape[1]-1 and 1<=j<patch.shape[0]-1):break
                    choices.append(set(labels[j-1:j+2,i-1:i+2].ravel())-{0})
                if len(choices)!=4 or any(len(c)!=1 for c in choices) or len(set.union(*choices))!=4:continue
                error=float(np.mean([np.mean(samples(empty,path+origin)) for path in [arc,straight]]))
                solutions.append((error,(left,top,right,bottom),~cut,arms+np.array([left,top]),center,radius))
    if not solutions:return None
    solutions.sort(key=lambda s:s[0]);best=solutions[0]
    # Multiple numerical fits of the same observed curve are one hypothesis.
    if any(np.linalg.norm(s[4]-best[4])>3*stroke for s in solutions[1:]):return None
    return best[1:4]+({'center':(best[4]+np.array([x+.5,y+.5])).tolist(),'inner_radius':best[5]+2*stroke,'outer_radius':best[5]+5*stroke,'stroke_radius':stroke},)



def separate_crossovers(ink, blocked, candidates, junctions):
    """Exclude each verified crossing disk from CC analysis and pair its visible arms."""
    touching = [candidate for candidate in candidates if len(candidate['ink_regions']) == 1]
    if not touching:
        return None
    if junctions:
        blocked = blocked.copy()
        for left, top, right, bottom in junctions:
            blocked[top:bottom, left:right] = 1
    screened = ink.copy()
    removed = np.zeros_like(ink, dtype=bool)
    proposals = []
    for candidate in touching:
        crossing = candidate['point_px']
        proof = verified_crossover(ink, blocked, crossing)
        if proof is None:
            continue
        (left, top, right, bottom), keep, arms = proof[:3]
        window = np.s_[top:bottom, left:right]
        if np.any(removed[window] & ~keep):
            return None
        removed[window] |= ~keep
        screened[window] &= keep
        proposals.append((crossing, arms))
    if not proposals:
        return None

    count, labels, stats, _ = cv2.connectedComponentsWithStats(screened, connectivity=8)
    offsets = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
    pairs = []
    for crossing, arms in proposals:
        observed = [set(samples(labels, arm + offsets)) - {0} for arm in arms]
        if any(len(choices) != 1 for choices in observed) or len(set().union(*observed)) != 4:
            return None
        regions = [int(next(iter(choices))) for choices in observed]
        pairs.append((crossing, arms, regions))

    roots = np.arange(count)
    for _, _, (a, b, c, d) in pairs:
        roots[roots == roots[b]] = roots[a]
        roots[roots == roots[d]] = roots[c]
    verified = []
    for crossing, arms, (a, _, c, _) in pairs:
        if roots[a] == roots[c]:
            return None
        verified.append({'point_px': crossing, 'arm_points_px': arms.round(3).tolist(),
                         'paired_ink_regions': [int(roots[a]), int(roots[c])]})
    for index, root in enumerate(roots):
        if index != root:
            stats[root, cv2.CC_STAT_AREA] += stats[index, cv2.CC_STAT_AREA]
    return screened, count, roots[labels].astype(labels.dtype), stats, verified


def prove_underpass(labels, x, y, left, right, over, before, after):
    """Require straight observed endpoints and one continuous arcing ink region."""
    if len({over, before, after}) != 3:
        return None
    top, bottom = y - 27, y + 28
    start, end = min(x, left) - 25, max(x, right) + 25
    if top < 0 or start < 0 or bottom > labels.shape[0] or end > labels.shape[1]:
        return None
    patch = labels[top:bottom, start:end]
    left, right, x, y = left - start, right - start, x - start, y - top
    reference_columns = slice(x - 4, x + 5)
    upper = np.nonzero(patch[y-24:y-18, reference_columns] == over)[1]
    lower = np.nonzero(patch[y+18:y+24, reference_columns] == over)[1]
    if min(len(upper), len(lower)) < 6:
        return None
    upper_x, lower_x = (x - 4 + np.median(points) for points in (upper, lower))
    if abs(upper_x - lower_x) > 1:
        return None
    axis_x = (upper_x + lower_x) / 2
    ends = (patch[y-4:y+5, left-3] == before) & (patch[y-4:y+5, right+2] == after)
    rows = np.flatnonzero(ends)
    if len(rows) < 2 or rows[-1] - rows[0] + 1 != len(rows):
        return None
    axis_y = y - 4 + np.median(rows)
    gap_x = (left + right - 1) / 2
    direction = 1 if gap_x > axis_x else -1
    estimated_radius = abs(gap_x - axis_x)
    if not 7 <= estimated_radius <= 20:
        return None
    distance = cv2.distanceTransform(np.uint8(patch != over), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ink_y, ink_x = np.nonzero(patch)
    along, across = direction * (ink_x - axis_x), ink_y - axis_y
    radii = np.hypot(along, across)
    theta = np.linspace(0, np.pi, 25)
    for radius in np.arange(estimated_radius - 2, estimated_radius + 2.1, .5):
        arc_x = np.rint(axis_x + direction * radius * np.sin(theta)).astype(int)
        arc_y = np.rint(axis_y + radius * np.cos(theta)).astype(int)
        if distance[arc_y, arc_x].max() > 1:
            continue
        nearby = radii <= radius + 5
        on_arc = (along >= -2) & (abs(radii - radius) <= 3.6)
        on_vertical = (abs(along) <= 3) & (abs(across) >= radius - 2.5)
        on_underpass = (abs(across) <= 3) & ((ink_x < left + 1) | (ink_x >= right - 1))
        on_route = np.where(patch[ink_y, ink_x] == over, on_arc | on_vertical, on_underpass)
        if np.all(on_route[nearby]):
            return {'chord': [float(axis_x + start), float(axis_y + top)],
                    'radius_px': float(radius), 'regions': [before, after, over]}
    return None
