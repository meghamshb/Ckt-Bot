"""One fixed local path with exact linear calculation. No alternate backend or retries."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import load, save, sha256, check_models


def canonical_image(source, output):
    from PIL import Image, ImageOps
    with Image.open(source) as image:
        image=ImageOps.exif_transpose(image).convert('RGBA')
        white=Image.new('RGBA',image.size,'white')
        image=Image.alpha_composite(white,image).convert('RGB')
        scale=min(max(1,768/max(image.size)),math.sqrt(1_900_000/(image.width*image.height)))
        if scale!=1:
            image=image.resize((max(1,int(image.width*scale)),max(1,int(image.height*scale))),Image.Resampling.LANCZOS)
        image.save(output)
        return {'original_path':str(Path(source).resolve()),'original_sha256':sha256(source),
            'canonical_size':list(image.size),'scale':scale,'policy':'EXIF, white RGB, min768 long side/max1.9MP; one shared frame'}


def publish_answer(output,answer):
    """Readers see a complete stage result while its explanation is enriched."""
    pending=output/'.answer.json.tmp'
    save(pending,answer)
    pending.replace(output/'answer.json')
    pending=output/'.answer.md.tmp'
    pending.write_text(answer_markdown(answer))
    pending.replace(output/'answer.md')


def answer_markdown(answer):
    lines=[f"Status: {answer['status']}", '', answer['answer']]
    if answer.get('reasoning'):lines+=['','Reasoning: '+answer['reasoning']]
    explanation=answer.get('explanation') or {}
    if answer.get('explanation_status')=='pending':
        lines+=['','Worked explanation: still being generated.']
    if explanation.get('status')=='available':
        lines+=['','Worked solution: '+explanation['method']]
        for index,step in enumerate(explanation.get('steps',[]),1):
            lines+=['',f"{index}. {step['title']}",step['text']]
            lines += ['$$'+row['latex']+'$$' if row.get('latex') else row.get('text','')
                for row in step.get('equations',[])]
    elif explanation:
        lines+=['','Worked explanation unavailable: '+explanation.get('error','No checked explanation was produced.')]
    if answer['equations']:
        lines+=['','Working:']+[f'- {x}' for x in answer['equations']]
    if answer['quantities']:
        lines+=['','Quantities:']
        for x in answer['quantities']:
            if x['kind']=='phasor':
                value=f"{x['real']:g} {x['imaginary']:+g}j; {x['magnitude']:.6g}∠{x['phase_degrees']:.6g}°"
            elif x['kind']=='scalar':
                value=f"{x['real']:g}"
            else:
                value=x['expression']
            lines.append(f"- {x['name']}: {value} {x['unit']} ({x['reference_direction']})")
    if answer['assumptions']:
        lines+=['','Assumptions:']+[f'- {x}' for x in answer['assumptions']]
    if answer['missing_evidence']:
        lines+=['','Unresolved:']+[f'- {x}' for x in answer['missing_evidence']]
    return '\n'.join(lines)+'\n'


def crosscheck_simulation(state, simulation):
    """Compare numerical results after both engines ran, allowing reference gauges."""
    from .linear import expression, CalculationError
    import sympy as sp
    checked = []; mismatches = []; skipped = []
    def exact(row):
        if not row or not row.get('identifiable'):
            return None
        try:
            return complex(sp.N(expression(row['exact_value'], {'I': sp.I}), 17))
        except (CalculationError, TypeError, ValueError):
            return None
    def check(name, expected, observed):
        if expected is None:
            skipped.append(name); return
        number = complex(observed['real'], observed['imaginary'])
        error = abs(number-expected)
        limit = 1e-7 + 1e-7*abs(expected)
        checked.append(name)
        if error > limit:
            mismatches.append({'quantity':name, 'absolute_error':error, 'tolerance':limit})
    voltages = state.get('node_voltages', {})
    for gauge in simulation.get('mapping', {}).get('gauges', []):
        offset = exact(voltages.get(gauge['net_id']))
        for net in gauge['island_net_ids']:
            value = exact(voltages.get(net)); observed = simulation['net_voltages'].get(net)
            if observed is not None:
                check('V('+net+')', None if value is None or offset is None else value-offset, observed)
    for name, observed in simulation.get('terminal_currents', {}).items():
        check('I('+name+')', exact(state.get('terminal_currents', {}).get(name)), observed)
    return {'status':'mismatch' if mismatches else 'passed' if checked else 'not_checked',
            'checked_quantities':len(checked), 'mismatches':mismatches, 'skipped_quantities':skipped,
            'absolute_tolerance':1e-7, 'relative_tolerance':1e-7,
            'scope':'Agreement between independent numerical engines on the same extracted circuit; not image/topology validation.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True,type=Path)
    parser.add_argument('--question',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    config_path=root/'config/pipeline.json';config=load(config_path)
    if not args.image.is_file():
        parser.error('Input image does not exist')
    try:
        check_models(config)
    except (ValueError,RuntimeError,OSError) as error:
        parser.error(str(error))
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    run={'status':'running','config_sha256':sha256(config_path),'question':args.question,
        'vision_model':config['vision'],'reasoning_model':config['reasoning'],'stages':[]}
    run['source_files']={str(p.relative_to(root)):sha256(p) for p in sorted((root/'src').rglob('*.py'))}
    run['source_sha256']=hashlib.sha256(json.dumps(run['source_files'],sort_keys=True).encode()).hexdigest()
    started=time.monotonic();stage='prepare'
    try:
        # This is the existing shared lock, opened read-only. No home-directory
        # files are created, and no unrelated Ollama model is unloaded.
        with Path(config['lock_file']).open('r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            source=output/'source.png'
            run['image']=canonical_image(args.image,source)
            env=os.environ.copy()
            env.update(PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1',
                YOLO_CONFIG_DIR=str(root/'work/ultralytics'),MPLCONFIGDIR=str(root/'work/matplotlib'))
            for stage in ('detect','segment','graph'):
                print(f'{stage}...',flush=True);tick=time.monotonic()
                with (output/f'{stage}.log').open('w') as log:
                    subprocess.run([config['python'],'-B','-m','kbot_pipeline.workers',stage,
                        str(config_path),str(root),str(source),str(output)],env=env,
                        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
                run['stages'].append({'stage':stage,'seconds':time.monotonic()-tick})
                save(output/'run.json',run)
            from .views import reading_sheet, overlay
            from .vision import read
            from .graph import finalize
            from .reasoning import solve
            candidate=load(output/'candidate.json')
            stage='vision';print('Reading circuit diagram...',flush=True);tick=time.monotonic()
            sheet=output/'reading-sheet.png';reading_sheet(source,candidate,sheet)
            readings=read(config,source,sheet,candidate,args.question,output)
            save(output/'readings.json',readings)
            run['stages'].append({'stage':stage,'seconds':time.monotonic()-tick})
            stage='assemble';tick=time.monotonic()
            circuit=finalize(candidate,readings)
            save(output/'circuit.json',circuit)
            overlay(source,circuit,output/'circuit.png')
            run['stages'].append({'stage':stage,'seconds':time.monotonic()-tick})
            stage='reasoning'
            print('Electrical interpretation incomplete; numerical inference skipped.' if circuit['electrical_issues']
                  else 'Interpreting question and calculating...',flush=True)
            tick=time.monotonic()
            answer=solve(config,circuit,args.question,output)
            save(output/'answer-linear.json',answer)
            run['stages'].append({'stage':stage,'seconds':time.monotonic()-tick,'model_inference':(output/'reasoning-request.json').exists(),'linear_calculation':(output/'calculation.json').exists()})
            save(output/'run.json',run)
            stage='spice';print('Running SPICE simulation...',flush=True);tick=time.monotonic()
            from .spice import simulate
            simulation=simulate(circuit,args.question,output)
            if simulation['status']=='completed' and (output/'linear-state.json').exists():
                simulation['crosscheck']=crosscheck_simulation(load(output/'linear-state.json'),simulation)
                save(output/'spice-result.json',simulation)
                if simulation['crosscheck']['status']=='mismatch':
                    answer={'status':'verification_failed','answer':'The independent numerical calculations disagree.',
                        'reasoning':'The original linear result and SPICE record are preserved for inspection.',
                        'equations':[],'assumptions':[],'quantities':[],
                        'missing_evidence':['Numerical disagreement on the extracted circuit; this run is not a verified answer.']}
            run['stages'].append({'stage':stage,'seconds':time.monotonic()-tick,'status':simulation['status']})
            run['spice_status']=simulation['status']
            # Numerical verification is finished. Publish its real result now;
            # the separately visible worked explanation may still be running.
            answer['numerical_verification']={'spice_status':simulation['status'],
                'crosscheck_status':simulation.get('crosscheck',{}).get('status','not_checked')}
            answer['explanation_status']='pending'
            publish_answer(output,answer)
            stage='explain';print('Explaining the checked circuit solution...',flush=True);tick=time.monotonic()
            save(output/'run.json',run)
            from .explanation import explain
            calculation=load(output/'calculation.json') if (output/'calculation.json').exists() else None
            # Optional prose has a bounded wait; it cannot invalidate the
            # already published numerical result or lock the demo for minutes.
            explanation_config={**config,'reasoning':{**config['reasoning'],
                'timeout':min(config['reasoning']['timeout'],60)}}
            explanation=explain(explanation_config,circuit,args.question,answer,calculation,simulation,output)
            save(output/'explanation.json',explanation)
            answer['explanation']=explanation
            answer['explanation_status']=explanation['status']
            run['stages'].append({'stage':stage,'seconds':time.monotonic()-tick,'status':explanation['status']})
            publish_answer(output,answer)
            run.update(status='completed',answer_status=answer['status'],seconds=time.monotonic()-started)
            save(output/'run.json',run)
            print(answer['answer']);print(f'Results: {output}')
    except Exception as error:
        run.update(status='failed',failed_stage=stage,error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-started)
        save(output/'run.json',run)
        print(f"Failed at {stage}: {error}. See {output}",file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
