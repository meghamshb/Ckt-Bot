"""Deterministic views expose the same pin/net identities consumed by Qwen."""
from __future__ import annotations

import math
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


COLORS = ['#b2182b', '#2166ac', '#1b7837', '#762a83', '#d66000', '#008080']


def font(size=16):
    return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', size)


def label(draw, xy, text, color, size=16):
    f = font(size)
    box = draw.textbbox(xy, text, font=f)
    draw.rectangle((box[0]-2, box[1]-2, box[2]+2, box[3]+2), fill='white')
    draw.text(xy, text, font=f, fill=color)


def overlay(source, circuit, path):
    image = Image.open(source).convert('RGB')
    draw = ImageDraw.Draw(image)
    fragments = {f['id']: f for f in circuit.get('wire_fragments', [])}
    for i, net in enumerate(circuit['nets']):
        if not net['pins']:
            continue
        lines = [fragments[f]['polyline_px'] for f in net.get('fragment_ids', []) if f in fragments]
        color = COLORS[i % len(COLORS)]
        for line in lines:
            if len(line) > 1:
                draw.line([tuple(p) for p in line], fill=color, width=2)
        if lines:
            line = max(lines, key=lambda x: sum(math.dist(a, b) for a, b in zip(x, x[1:])))
            a, b = line[len(line)//2-1], line[len(line)//2]
            label(draw, ((a[0]+b[0])/2+5, (a[1]+b[1])/2+5), net['id'], color, 14)
    for c in circuit['components']:
        draw.rectangle(c['bbox'], outline='#1d4ed8', width=2)
        label(draw, (c['bbox'][0], max(0, c['bbox'][1]-20)), c['id']+':'+c['type'], '#1d4ed8', 14)
    for p in circuit['pins']:
        if p.get('anchor_px'):
            x, y = p['anchor_px']
            draw.ellipse((x-3, y-3, x+3, y+3), fill='#c00000')
            label(draw, (x+4, y+4), p['id'] + (':'+p['role'] if p.get('role') else ''), '#c00000', 13)
    for x in circuit.get('intersections', []):
        if x['state']!='unresolved' or not x.get('relevant_to_components',True):
            continue
        px, py = x['point_px']
        draw.rectangle((px-5, py-5, px+5, py+5), outline='#7e22ce', width=2)
        label(draw, (px+7, py-20), x['id'], '#7e22ce', 14)
    for arrow in circuit.get('reference_arrows',[]):
        box=arrow['bbox_px'];draw.rectangle(box,outline='#a21caf',width=2)
        label(draw,(box[2]+3,box[1]),arrow['id'],'#a21caf',13)
        for wid,choice in arrow['choices'].items():
            if 'point_px' not in choice:continue
            x,y=choice['point_px'];draw.ellipse((x-3,y-3,x+3,y+3),outline='#a21caf',width=2)
            label(draw,(x+4,y+4),arrow['id']+'.'+wid,'#a21caf',12)
    image.save(path)


def reading_sheet(source, candidate, path):
    image = Image.open(source).convert('RGB')
    overview_path = Path(path).with_name('candidate-overlay.png')
    overlay(source, candidate, overview_path)
    overview = Image.open(overview_path).convert('RGB')
    width = 1536
    top = min(760, overview.height + 58)
    columns = min(4, max(1, len(candidate['components'])))
    rows = max(1, math.ceil(len(candidate['components'])/columns))
    cell_h = 230
    canvas = Image.new('RGB', (width, top + cell_h*rows), 'white')
    draw = ImageDraw.Draw(canvas)
    overview.thumbnail((width-24, top-40))
    canvas.paste(overview, ((width-overview.width)//2, 30))
    draw.text((12, 6), 'REFERENCE IDS ONLY — original pixels are in the first image', font=font(18), fill='black')
    cell_w = width//columns
    tiles=Path(path).with_name(Path(path).stem+'-components');tiles.mkdir(exist_ok=True)
    for index, c in enumerate(candidate['components']):
        x, y = (index % columns)*cell_w, top+(index//columns)*cell_h
        a,b,d,e = c['bbox']
        bw,bh=d-a,e-b
        # Upright ratings beside vertical bodies need more horizontal room;
        # continuation-wire neighbours should not dominate the target crop.
        if bh>1.5*bw:px,py=max(48,round(1.4*bh)),max(32,round(.25*bh))
        elif bw>1.5*bh:px,py=max(32,round(.25*bw)),max(32,round(.65*bw))
        else:px,py=max(64,round(1.5*max(bw,bh))),max(32,round(.7*max(bw,bh)))
        box = (max(0,int(a-px)), max(0,int(b-py)), min(image.width,int(d+px)), min(image.height,int(e+py)))
        crop = image.crop(box)
        scale = min((cell_w-16)/crop.width, (cell_h-36)/crop.height, 3)
        crop = crop.resize((max(1,round(crop.width*scale)), max(1,round(crop.height*scale))))
        crop_draw=ImageDraw.Draw(crop)
        target=[(a-box[0])*scale,(b-box[1])*scale,(d-box[0])*scale,(e-box[1])*scale]
        crop_draw.rectangle(target,outline='#1450bb',width=1)
        # A body-sized neighborhood cannot bound an arbitrarily long rating.
        # Every fixed reader input therefore contains the complete, unchanged
        # source frame plus the outlined target detail. No text is discarded,
        # and generated IDs/captions remain outside the literal image.
        gap=24
        literal=Image.new('RGB',(max(image.width,crop.width),image.height+gap+crop.height),'white')
        literal.paste(image,(0,0))
        detail_x=(literal.width-crop.width)//2;detail_y=image.height+gap
        literal.paste(crop,(detail_x,detail_y))
        literal.save(tiles/(c['id']+'.png'))
        (tiles/(c['id']+'.json')).write_text(json.dumps({
            'original_frame_px':[0,0,image.width,image.height],
            'detail_frame_px':[detail_x,detail_y,detail_x+crop.width,detail_y+crop.height],
            'target_bbox_source_px':list(c['bbox']),
            'detail_context_source_px':list(box),
            'source_pixels_preserved':'complete unscaled original prefix; target outline only in detail'
        },indent=2)+'\n')
        for pin in candidate['pins']:
            if pin['component_id']!=c['id'] or not pin.get('anchor_px'):continue
            px=(pin['anchor_px'][0]-box[0])*scale;py=(pin['anchor_px'][1]-box[1])*scale
            side=pin.get('geometric_side','') or ''
            bounds=crop_draw.textbbox((0,0),pin['id'],font=font(15));tw=bounds[2]-bounds[0]
            if 'left' in side:position=(max(1,px-tw-8),py-9)
            elif 'right' in side:position=(min(crop.width-tw-2,px+8),py-9)
            elif side=='top':position=(px-tw/2,max(1,py-25))
            else:position=(px-tw/2,min(crop.height-20,py+7))
            crop_draw.ellipse((px-2,py-2,px+2,py+2),fill='#1450bb')
            label(crop_draw,position,pin['id'],'#1450bb',15)
        draw.rectangle((x,y,x+cell_w-1,y+cell_h-1), outline='#cccccc')
        draw.text((x+8,y+5), c['id']+' '+c['type']+' | original neighborhood', font=font(15), fill='blue')
        canvas.paste(crop, (x+8,y+30))
        canvas.crop((x,y,x+cell_w,y+cell_h)).save(tiles/(c['id']+'-annotated.png'))
    canvas.save(path)


def measurement_sheet(source,candidate,path):
    """Keep original electrical text legible while naming physical sites."""
    from .measurements import sites
    import numpy as np
    original=Image.open(source).convert('RGB');scale=2;image=original.resize((original.width*scale,original.height*scale));draw=ImageDraw.Draw(image);ink=np.asarray(image).mean(2)<225;occupied=np.zeros(ink.shape,bool)
    for site in sites(candidate):
        x,y=[v*scale for v in site['point_px']];text=site['id'];box=draw.textbbox((0,0),text,font=font(17));w,h=box[2]-box[0]+6,box[3]-box[1]+6;positions=[]
        for radius in (12,28,45):
            for dx,dy in ((1,1),(1,-1),(-1,1),(-1,-1),(0,1),(0,-1)):
                a=max(0,min(image.width-w,round(x+dx*radius-(w if dx<0 else w/2 if dx==0 else 0))));b=max(0,min(image.height-h,round(y+dy*radius-(h if dy<0 else 0))));cost=float(ink[b:b+h,a:a+w].mean())+2*float(occupied[b:b+h,a:a+w].mean())+radius/10000;positions.append((cost,a,b))
        _,a,b=min(positions);occupied[b:b+h,a:a+w]=1;draw.line((x,y,a+w/2,b+h/2),fill='#729ecc',width=1);draw.ellipse((x-2,y-2,x+2,y+2),fill='#1450bb');label(draw,(a+3,b+1),text,'#1450bb',17)
    from .voltage_pairs import polarity_pairs
    pairs=polarity_pairs(candidate)
    if pairs:
        panel=Image.new('RGB',(image.width,image.height+28*(len(pairs)+1)),'white');panel.paste(image,(0,0));d=ImageDraw.Draw(panel)
        for i,pair in enumerate(pairs):d.text((12,image.height+8+28*i),pair['id']+' : + '+pair['positive_site_id']+'   /   - '+pair['negative_site_id']+'  (observed polarity pair)',fill='#1450bb',font=font(19))
        image=panel
    image.save(path)


def enumerated_reading_view(source,candidate,path):
    """One source-sized ID overview; the separate original stays untouched."""
    import numpy as np
    from .measurements import sites
    image=Image.open(source).convert('RGB');draw=ImageDraw.Draw(image)
    ink=np.asarray(image).mean(2)<225;occupied=np.zeros(ink.shape,bool)
    def place(point,text,color,size=12):
        x,y=point;bounds=draw.textbbox((0,0),text,font=font(size));w,h=bounds[2]-bounds[0]+6,bounds[3]-bounds[1]+6
        positions=[]
        for radius in (9,18,30):
            for dx,dy in ((1,1),(1,-1),(-1,1),(-1,-1),(0,1),(0,-1)):
                a=max(0,min(image.width-w,round(x+dx*radius-(w if dx<0 else w/2 if dx==0 else 0))))
                b=max(0,min(image.height-h,round(y+dy*radius-(h if dy<0 else 0))))
                region=ink[b:b+h,a:a+w];used=occupied[b:b+h,a:a+w]
                if not region.size:continue
                cost=float(region.mean())+2*float(used.mean())+radius/10000
                positions.append((cost,a,b))
        if not positions:return
        _,a,b=min(positions);occupied[b:b+h,a:a+w]=True
        draw.line((x,y,a+w/2,b+h/2),fill=color,width=1)
        label(draw,(a+3,b+1),text,color,size)
    for component in candidate['components']:
        a,b,c,d=component['bbox'];draw.rectangle((a,b,c,d),outline='#1d4ed8',width=1)
        place((a,b),component['id'],'#1d4ed8',13)
    for site in sites(candidate):
        point=site['point_px'];x,y=point;draw.ellipse((x-2,y-2,x+2,y+2),fill='#12766f');place(point,site['id'],'#12766f')
    for arrow in candidate.get('reference_arrows',[]):
        a,b,c,d=arrow['bbox_px'];draw.rectangle((a,b,c,d),outline='#a21caf',width=1);place((c,b),arrow['id'],'#a21caf')
        for wid,choice in arrow['choices'].items():
            if choice.get('point_px') is not None:place(choice['point_px'],arrow['id']+'.'+wid,'#a21caf')
    for crossing in candidate.get('intersections',[]):
        if crossing['state']=='unresolved' and crossing.get('relevant_to_components',True):place(crossing['point_px'],crossing['id'],'#8b5d20')
    image.save(path)
