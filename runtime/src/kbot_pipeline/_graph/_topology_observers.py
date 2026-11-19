"""Exact low-level geometric observers; no old reader/OCR/solver entrypoint.

Copied from reader-next raster_wires.py90416e01; scope/limitations are retained.
Current adapter accepts class names explicitly and supplies canonical RGB masks.
"""
import math
import cv2
import numpy as np
from ._topology_bridges import prove_underpass
_DIAGONAL_SLOPES=(.67,.75,.83,.91,1.0,1.10,1.21,1.33,1.49)
SIDE_PAIRS=({'left','right'},{'top','bottom'})
def _runs(line):
    changes = np.flatnonzero(np.diff(np.pad(np.asarray(line, dtype=np.int8), 1)))
    return list(zip(changes[::2].tolist(), changes[1::2].tolist()))


def _contact(side, box, hlines, vlines, labels):
    x0, y0, x1, y1 = box
    along_x = side in ('left', 'right')
    sign = -1 if side in ('left', 'top') else 1
    first = {'left': x0-1, 'right': x1, 'top': y0-1, 'bottom': y1}[side]
    low, high = (y0, y1) if along_x else (x0, x1)
    scan = hlines.T if along_x else vlines
    raw = labels.T if along_x else labels
    for gap in range(3):
        edge = first + sign*gap
        if not 0 <= edge < scan.shape[0]:
            continue
        direction = [edge+sign*d for d in range(6) if 0 <= edge+sign*d < scan.shape[0]]
        if len(direction) < 4:
            continue
        support = scan[direction, low:high]
        proposed = (support[0] != 0) & (support.sum(axis=0) >= len(direction)-1)
        if not np.any(proposed):
            continue
        bands = []
        for lo, hi in _runs(proposed):
            center = low + (lo + hi - 1) // 2
            region = int(raw[edge, center])
            if region == 0:
                continue
            if bands:
                a, b, earlier, earlier_center = bands[-1]
                # A one-pixel fringe of the same stroke can be separated by
                # a thresholded antialias row. Keep physically distinct pins.
                if (region == earlier and lo - b == 1 and min(b - a, hi - lo) <= 2
                        and hi - a <= 12):
                    bands[-1] = (a, hi, region, earlier_center if b - a >= hi - lo else center)
                    continue
            bands.append((lo, hi, region, center))
        options = [{
            'side': side,
            'point_px': [edge, center] if along_x else [center, edge],
            'ink_region': region,
            'edge_contact_width_px': hi - lo,
            'boundary_gap_px': gap,
            'consecutive_wire_support_px': len(direction),
        } for lo, hi, region, center in bands]
        if options:
            return options
    return []


def _unit(x, y):
    length = math.hypot(x, y)
    return np.array([x / length, y / length])


def _diagonal_masks(ink, length):
    def kernel(slope, sign, size):
        radius = size // 2
        t = np.arange(-radius, radius + 1)
        dx = t if slope <= 1 else np.rint(t / slope).astype(int)
        dy = np.rint(t * slope).astype(int) if slope <= 1 else t
        result = np.zeros((size, size), np.uint8)
        result[radius + sign * dy, radius + dx] = 1
        return result

    directions = []
    for sign in (1, -1):
        supported = np.zeros_like(ink)
        expanded = np.zeros_like(ink)
        for slope in _DIAGONAL_SLOPES:
            opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel(slope, sign, length))
            supported |= opened
            if opened.any():
                expanded |= cv2.dilate(opened, kernel(slope, sign, max(23, length + 10)))
        directions.append((supported, expanded))
    return directions


def _samples(image, points):
    xs, ys = np.moveaxis(np.rint(points).astype(int), -1, 0)
    height, width = image.shape
    valid = (0 <= xs) & (xs < width) & (0 <= ys) & (ys < height)
    return np.where(valid, image[ys.clip(0, height - 1), xs.clip(0, width - 1)], 0)


def _diagonal_rays(box, labels, axis, normal, centers, direction):
    lo = np.array(box[:2]) - .5
    hi = np.array(box[2:]) - .5
    facing = axis * direction
    edge = np.where(facing > 0, hi, lo)
    distance = np.min((edge - centers) / facing, axis=1)
    # Native centerline samples tolerate one pixel of stair-step quantization.
    points = centers[:, None, :] + (distance[:, None, None] + np.arange(1, 16)[None, :, None]) * facing
    strip = points[:, :, None, :] + np.arange(-1, 2)[None, None, :, None] * normal
    sampled = _samples(labels, strip)
    rays = []
    for i, sample in enumerate(sampled):
        counts = np.bincount(sample[:4].ravel())
        if len(counts) <= 1:
            continue
        counts[0] = 0
        region = int(np.argmax(counts))
        hits = np.any(sample == region, axis=1)
        if hits.sum() < 13 or hits[-5:].sum() < 4:
            continue
        first = int(np.flatnonzero(hits)[0])
        for offset in (0, -1, 1):
            if sample[first, offset + 1] == region:
                px = np.rint(points[i, first] + offset * normal).astype(int)
                break
        transverse = _samples(labels, px + np.arange(-7, 8)[:, None] * normal) == region
        width = max((b - a for a, b in _runs(transverse) if a <= 7 < b), default=0)
        if not 1 <= width <= 12:
            continue
        rays.append((i, int(hits.sum()), {
            'point_px': px.tolist(), 'ink_region': region, 'edge_contact_width_px': width,
            'boundary_gap_px': first, 'consecutive_wire_support_px': int(hits.sum()),
            'direction_from_body': np.round(facing, 4).tolist(), 'contact_axis': 'diagonal',
        }))
    return rays


def _diagonal_contacts(box, labels, diagonal_masks, *, single=False):
    x0, y0, x1, y1 = box
    breadth = min(x1 - x0, y1 - y0)
    if breadth < 10:
        return []
    center = (np.array(box[:2]) + np.array(box[2:]) - 1) / 2
    radius = max(3, min(10, round(.22 * breadth)))
    offsets = np.arange(-radius, radius + 1)
    align_tolerance = min(4, max(2, round(.075 * breadth)))
    possibilities = {}
    for sign, (support, _) in zip((1, -1), diagonal_masks, strict=True):
        height, width = labels.shape
        if not support[max(0, y0 - 16):min(height, y1 + 16),
                       max(0, x0 - 16):min(width, x1 + 16)].any():
            continue
        for slope in _DIAGONAL_SLOPES:
            axis = _unit(1, sign * slope)
            normal = np.array([-axis[1], axis[0]])
            centers = center + offsets[:, None] * normal
            low = _diagonal_rays(box, labels, axis, normal, centers, -1)
            high = _diagonal_rays(box, labels, axis, normal, centers, 1)
            lo_side, hi_side = ('top_left', 'bottom_right') if sign > 0 else ('bottom_left', 'top_right')
            if single:
                for side, rays in ((lo_side, low), (hi_side, high)):
                    for index, hits, contact in rays:
                        key = (side, contact['ink_region'])
                        score = (hits, -contact['boundary_gap_px'], -abs(offsets[index]), -abs(slope - 1))
                        if key not in possibilities or possibilities[key][0] < score:
                            possibilities[key] = (score, [dict(contact, side=side)])
                continue
            for li, ls, a in low:
                for hi, hs, b in high:
                    if abs(li - hi) > align_tolerance:
                        continue
                    score = (ls + hs, -a['boundary_gap_px'] - b['boundary_gap_px'],
                             -abs(li - hi), -abs(offsets[li] + offsets[hi]), -abs(slope - 1))
                    key = (sign, a['ink_region'], b['ink_region'])
                    if key in possibilities and possibilities[key][0] >= score:
                        continue
                    possibilities[key] = (score, [dict(a, side=lo_side), dict(b, side=hi_side)])
    if len(possibilities) != 1:
        return []
    return next(iter(possibilities.values()))[1]


def _local_crossings(ink, hlines, vlines, labels, mask, run_length):
    """Confirm filled dots at continuous four-way contacts, flag bare ones."""
    bits = (hlines & vlines).astype(np.uint8)
    n, _, stats, centers = cv2.connectedComponentsWithStats(bits, connectivity=8)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radius = max(8, run_length+3)
    out = []
    height, width = ink.shape
    for (left, top, wide, high, area), (cx, cy) in zip(stats[1:n], centers[1:n], strict=True):
        x, y = round(cx), round(cy)
        if area > 300 or not (radius < x < width-radius and radius < y < height-radius):
            continue
        if mask[y-radius:y+radius+1, x-radius:x+radius+1].any():
            continue
        arm_coords = (range(x-radius,x-radius+4), range(x+radius-3,x+radius+1),
                      range(y-radius,y-radius+4), range(y+radius-3,y+radius+1))
        arms = [
            hlines[y-2:y+3, list(arm_coords[0])].max(axis=0),
            hlines[y-2:y+3, list(arm_coords[1])].max(axis=0),
            vlines[list(arm_coords[2]), x-2:x+3].max(axis=1),
            vlines[list(arm_coords[3]), x-2:x+3].max(axis=1),
        ]
        if any(np.count_nonzero(arm)<3 for arm in arms):
            continue
        widths = []
        for coords, signal in zip(arm_coords, (ink, ink, ink.T, ink.T), strict=True):
            cross_axis = y if signal is ink else x
            for i in coords:
                span = signal[cross_axis-7:cross_axis+8,i]
                choices = [hi-lo for lo,hi in _runs(span) if lo<=7<hi or abs((lo+hi-1)/2-7)<=2]
                if choices:
                    widths.append(min(choices))
        if len(widths) < 12:
            continue
        stroke = float(np.median(widths))
        peak = float(distance[y-2:y+3,x-2:x+3].max())
        off = max(2, round(stroke*.78))
        full_quadrants = sum(bool(ink[y+dy*off,x+dx*off]) for dx in (-1,1) for dy in (-1,1))
        dot = peak/stroke >= 1.10 and full_quadrants >= 3
        region = int(labels[y,x])
        if not region:
            continue
        out.append({
            'kind': 'joined_filled_dot' if dot else 'ambiguous_four_way',
            'point_px': [x,y],
            'ink_regions': [region],
            'stroke_width_px': stroke,
            'junction_radius_px': round(peak,3),
            'filled_quadrants': full_quadrants,
        })
    return out


def _crossing_arms(center, support, labels, vectors, radius):
    distances = np.arange(radius - 3, radius + 4)
    for axis in vectors:
        normal = np.array([-axis[1], axis[0]])
        strips = (center + np.array([-1, 1])[:, None, None, None]
                  * distances[None, :, None, None] * axis
                  + np.arange(-1, 2)[None, None, :, None] * normal)
        values = _samples(labels, strips) * _samples(support, strips)
        arm_regions = []
        for arm in values:
            counts = np.bincount(arm.ravel())
            counts[0] = 0
            region = int(np.argmax(counts))
            if region and np.any(arm == region, axis=1).sum() >= 6:
                arm_regions.append(region)
        if len(arm_regions) == 2:
            return arm_regions
    return None


def _diagonal_crossings(ink, hlines, vlines, diagonals, labels, mask, line_run):
    # Candidate dilation also finds a short interrupted arm near an overpass.
    radii = (max(10, line_run + 3), max(18, line_run + 9), max(23, line_run + 14))
    extension = max(23, line_run + 12)
    horizontal = (hlines, cv2.dilate(hlines, np.ones((1, extension), np.uint8)), (_unit(1, 0),))
    vertical = (vlines, cv2.dilate(vlines, np.ones((extension, 1), np.uint8)), (_unit(0, 1),))
    positive, negative = [(*masks, tuple(_unit(1, sign * slope) for slope in _DIAGONAL_SLOPES))
                          for sign, masks in zip((1, -1), diagonals, strict=True)]
    result = []
    height, width = ink.shape
    for first, second in ((positive, negative), (horizontal, positive), (horizontal, negative),
                          (vertical, positive), (vertical, negative)):
        support_a, expand_a, angles_a = first
        support_b, expand_b, angles_b = second
        # The center may be white at a printed overpass. Demand continuous
        # support on all four outward arms before treating it as a crossing.
        common = expand_a & expand_b
        if not common.any():
            continue
        count, _, stats, centers = cv2.connectedComponentsWithStats(common, connectivity=8)
        for (_, _, _, _, area), (cx, cy) in zip(stats[1:count], centers[1:count], strict=True):
            x, y = round(cx), round(cy)
            if (area > 340 or not (radii[0] + 4 < x < width - radii[0] - 4
                                    and radii[0] + 4 < y < height - radii[0] - 4)):
                continue
            if any(max(abs(x - c['point_px'][0]), abs(y - c['point_px'][1])) <= 6 for c in result):
                continue
            for dx, dy in ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)):
                point = np.array([x + dx, y + dy])
                if mask[point[1], point[0]]:
                    continue
                arms_a = next((arms for radius in radii
                               if (arms := _crossing_arms(point, support_a, labels, angles_a, radius))), None)
                if not arms_a:
                    continue
                arms_b = next((arms for radius in radii
                               if (arms := _crossing_arms(point, support_b, labels, angles_b, radius))), None)
                if not arms_b:
                    continue
                result.append({'kind': 'ambiguous_diagonal_four_way', 'point_px': point.tolist(),
                               'ink_regions': sorted(set(arms_a + arms_b)),
                               'arm_ink_regions': [arms_a, arms_b]})
                break
    return result


def _possible_bridges(along, across, labels, mask, swapped, *, verify_isolated=False):
    """Look for interrupted straight underpasses beside a continuous curved overpass."""
    results = []
    height, width = along.shape
    for y in range(25, height-25):
        segments = _runs(along[y])
        for (a,b),(c,d) in zip(segments,segments[1:]):
            if b-a < 12 or d-c < 12 or not 3 <= c-b <= 19:
                continue
            lregion, rregion = int(labels[y,b-1]), int(labels[y,c])
            if not lregion or not rregion or b < 25 or c > width-25:
                continue
            if mask[y-19:y+20,b-22:c+23].any():
                continue
            # An arc above and below the underpass must be the same raw-ink
            # component, separate from both straight wire arms.
            here = set(map(int, labels[y-2:y+3,b:c].ravel())) - {0,lregion,rregion}
            if not here:
                continue
            xlo,xhi = b-22,c+22
            upper = across[y-24:y-18,xlo:xhi].sum(axis=0)
            lower = across[y+19:y+25,xlo:xhi].sum(axis=0)
            xs = np.flatnonzero((upper>=4)&(lower>=4))+xlo
            for x in xs.tolist():
                over = int(labels[y-21,x])
                if over not in here or int(labels[y+21,x])!=over:
                    continue
                p = [y,x] if swapped else [x,y]
                if any(max(abs(p[0]-a['point_px'][0]),abs(p[1]-a['point_px'][1]))<=6 for a in results):
                    break
                record = {'kind':'possible_wire_bridge','point_px':p,
                          'ink_regions':sorted({lregion,rregion,over}),
                          'under_gap_px':c-b,
                          'under_axis':'vertical' if swapped else 'horizontal'}
                if verify_isolated:
                    proof = prove_underpass(labels, x, y, b, c, over, lregion, rregion)
                    if proof is not None:
                        if swapped:
                            proof['chord'].reverse()
                        record.update(kind='verified_isolated_crossover', proof=proof)
                results.append(record)
                break
    return results


def _ground_end_rail_box(box, ink, line_run, reach):
    """Keep a rail that ends at a verified three-bar ground stem."""
    x0, y0, x1, y1 = box
    width = x1 - x0
    center = (x0 + x1) // 2
    margin = max(4, round(.16 * width))
    left, right = x0 - reach, x1 + reach
    rail_rows = []
    for y in range(max(0, y0 - 5), min(y1, y0 + 6)):
        for start, stop in _runs(ink[y, left:right]):
            if start == 0 and abs(left + stop - center) <= margin:
                rail_rows.append((y, left, left + stop, 'left'))
            elif stop == right - left and abs(left + start - center) <= margin:
                rail_rows.append((y, left + start, right, 'right'))
    if not rail_rows or len({side for _, _, _, side in rail_rows}) != 1:
        return box
    rail_y, rail_left, rail_right, _ = rail_rows[-1]
    if rail_y - rail_rows[0][0] + 1 != len(rail_rows) or len(rail_rows) > line_run:
        return box
    top = rail_y + line_run + 2
    if not y0 < top < (y0 + y1) / 2:
        return box

    body = ink[top:y1, x0:x1]
    row_widths = body.sum(axis=1)
    bands = _runs(row_widths >= max(10, .19 * width))
    if len(bands) != 3 or bands[0][0] < 3:
        return box
    stem = body[:bands[0][0]]
    stems = _runs(stem.all(axis=0))
    if (len(stems) != 1 or stems[0][1] - stems[0][0] > 10
            or np.any(stem.sum(axis=1) != stems[0][1] - stems[0][0])):
        return box
    stem_start, stem_end = stems[0]
    stem_x = (stem_start + stem_end - 1) / 2
    if (abs(x0 + stem_x - center) > margin
            or not rail_left <= x0 + stem_start < x0 + stem_end <= rail_right
            or not ink[rail_y:top + 1, x0 + stem_start:x0 + stem_end].all()):
        return box

    lengths = []
    for first, last in bands:
        row = first + int(np.argmax(row_widths[first:last]))
        runs = _runs(body[row])
        if (last - first > line_run or len(runs) != 1
                or abs((runs[0][0] + runs[0][1] - 1) / 2 - stem_x) > max(2, .08 * width)
                or ink[top + first:top + last, left:x0].any()
                or ink[top + first:top + last, x1:right].any()):
            return box
        lengths.append(int(row_widths[row]))
    if not (lengths[0] >= .50 * width and lengths[1] <= .86 * lengths[0]
            and lengths[2] <= .86 * lengths[1]):
        return box
    if (body[bands[-1][1]:].any()
            or any(body[before[1]:after[0]].any() for before, after in zip(bands, bands[1:]))):
        return box
    return x0, top, x1, y1


def _ground_body_box(box, ink, line_run):
    """Keep a demonstrably connected rail above the dedicated ground stem."""
    x0, y0, x1, y1 = box
    height, width = ink.shape
    reach = max(8, line_run)
    if x0 < reach or x1 + reach > width:
        return box
    # A ground body bar stays inside its box. A wire rail continues across both
    # sides and can touch the top of an overinclusive ground box.
    start, stop = max(0, y0 - 3), min(height, y0 + 7)
    rails = np.flatnonzero(ink[start:stop, x0 - reach:x1 + reach].mean(axis=1) >= .98)
    if not rails.size:
        return _ground_end_rail_box(box, ink, line_run, reach)
    rail_bottom = start + int(rails[-1]) + 1
    top = rail_bottom + line_run + 1
    if top >= (y0 + y1) / 2 or top + 2 >= height:
        return box
    center = (x0 + x1) // 2
    margin = max(3, round(.13 * (x1 - x0)))
    strip = ink[rail_bottom - 1:top + 2, center - margin:center + margin + 1]
    stems = _runs(strip.all(axis=0))
    if (len(stems) != 1 or stems[0][1] - stems[0][0] > 10
            or ink[top, x0:x1].sum() > max(9, .22 * (x1 - x0))):
        return box
    return (x0, top, x1, y1)


def _box(bounds, width, height, padding=0):
    x0, y0, x1, y1 = map(float, bounds)
    if not all(map(math.isfinite, (x0, y0, x1, y1))) or x0 >= x1 or y0 >= y1:
        raise ValueError(f'Invalid pixel box {bounds!r}')
    return (max(0, math.floor(x0)-padding), max(0, math.floor(y0)-padding),
            min(width, math.ceil(x1)+padding), min(height, math.ceil(y1)+padding))


def _gray(image):
    raster = np.asarray(image)
    if raster.dtype != np.uint8:
        raise ValueError('Use a uint8 raster at its native dimensions')
    if raster.ndim == 2:
        return raster
    if raster.ndim == 3 and raster.shape[2] in (3, 4):
        # A channel mean is invariant to PIL RGB versus OpenCV BGR ordering.
        return np.rint(raster[..., :3].mean(axis=2)).astype(np.uint8)
    raise ValueError('Expected a grayscale or RGB/BGR image')


def _earth_ground_down(image, bounds, *, observe=False):
    """Fit a three-bar glyph and its stem independently of neighbouring ink.

    Horizontal stroke projection separates the bars from vertical wires near
    the detector box. All shape coordinates still refer to original pixels.
    """
    gray=_gray(image);height,width=gray.shape
    left,top,right,bottom=_box(bounds,width,height,padding=2)
    reach=max(3,round(.4*(right-left)))
    x0,x1=max(0,left-reach),min(width,right+reach)
    y1=min(height,bottom+reach);region=gray[top:y1,x0:x1]
    paper=float(np.percentile(gray,90));signatures=[]
    for delta in (48,60):
        ink=(region<=paper-delta).astype(np.uint8)
        longest=int(ink.sum(axis=1).max())
        if longest<8:return None
        # Select opening length from native stroke thickness. A neighbouring
        # long rail must not erase the glyph's short third bar.
        distance=cv2.distanceTransform(ink,cv2.DIST_L2,5)
        ridge=(distance>0)&(distance>=cv2.dilate(distance,np.ones((3,3),np.uint8)))
        radius=float(np.median(distance[ridge])) if ridge.any() else 1.
        opening=max(3,2*round(radius)+1)
        horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,opening),np.uint8))
        bands=_runs(horizontal.sum(axis=1)>0)
        candidates=[]
        for offset in range(len(bands)-2):
            candidate=_ground_bar_triplet(ink,horizontal,bands[offset:offset+3],gray,top,x0,x1,reach,delta,paper,height,observe=observe)
            if candidate is not None:candidates.append(candidate)
        if len(candidates)!=1:return None
        signatures.append(candidates[0])
    shape=[s['signature'] for s in signatures] if observe else signatures
    return signatures[0] if np.max(np.abs(np.subtract(*shape)))<=.08 else None


def _ground_bar_triplet(ink,horizontal,bands,gray,top,x0,x1,reach,delta,paper,height,*,observe=False):
    lengths=[];centers=[];spans=[]
    for start,end in bands:
        columns=_runs(np.any(horizontal[start:end],axis=0))
        if len(columns)!=1:return None
        a,b=columns[0];spans.append((a,b));lengths.append(b-a);centers.append((a+b)/2)
    thickness=max(end-start for start,end in bands)
    if not (lengths[0]>lengths[1]>lengths[2]>=2*thickness):return None
    if thickness>.2*lengths[0] or max(centers)-min(centers)>max(thickness,.08*lengths[0]):return None
    # A bar footprint, not the entire enlarged detector neighbourhood,
    # must be empty between bars. Adjacent circuit wires are not the glyph.
    a,b=spans[0]
    if any(not np.any(~np.any(ink[before[1]:after[0],a:b],axis=1)) for before,after in zip(bands,bands[1:])):return None
    if not np.any(~np.any(ink[bands[-1][1]:,spans[-1][0]:spans[-1][1]],axis=1)):return None
    bar_top=top+bands[0][0];stem_rows=0
    clearance=min(after[0]-before[1] for before,after in zip(bands,bands[1:]))
    found=False
    for y in range(bar_top-1,max(-1,bar_top-reach-1),-1):
        runs=_runs(gray[y,x0:x1]<=paper-delta)
        central=[(a,b) for a,b in runs if abs((a+b)/2-centers[0])<=max(thickness,.08*lengths[0])]
        narrow=len(central)==1 and central[0][1]-central[0][0]<=max(2*thickness,.18*lengths[0])
        if narrow:stem_rows+=1;found=True
        elif found:break
        elif bar_top-y>clearance:break
    if not stem_rows or top+bands[-1][1]>=height:return None
    signature=[lengths[1]/lengths[0],lengths[2]/lengths[0],(bands[2][0]-bands[0][0])/lengths[0]]
    if observe:
        native=[]
        for (a,b),(start,end) in zip(spans,bands):
            runs=[(left,right) for left,right in _runs(np.any(ink[start:end],axis=0)) if left<=(a+b)/2<right]
            if len(runs)!=1:return None
            left,right=runs[0];native.append([x0+left,top+start,x0+right,top+end])
        return {'signature':signature,'bar_boxes_px':native}
    return signature


def earth_ground_shape(image, bounds):
    """Observe the three bars and stem in any orthogonal orientation.

    Rotation changes neither the pixels nor the bar/stem requirements.
    This is one geometric recognizer, not a second model or an alias guess.
    """
    raster=np.asarray(image)
    box=list(bounds);matches=[]
    for rotation in range(4):
        proof=_earth_ground_down(raster,box)
        if proof is not None:matches.append(proof)
        width=raster.shape[1]
        x0,y0,x1,y1=box
        box=[y0,width-x1,y1,width-x0]
        raster=np.rot90(raster)
    return matches[0] if len(matches)==1 else None
