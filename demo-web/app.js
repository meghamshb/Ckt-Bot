'use strict';

const $ = id => document.getElementById(id);
const app = {samples:[], current:null, snapshot:null, timer:null, health:null, selectedComponent:null, selectedNet:null, overlay:'final', jsonPath:null, jsonVersion:null, jsonToken:0, submitting:false, localPreview:null, navigation:0};
app.visual='input';app.inspector='results';app.follow=true;
const stageLabels = {input:'Input',detect:'YOLO',segment:'SAM mask',graph:'Initial graph',vision:'Diagram reading',assemble:'Final graph',prepare:'Linear state',target:'Question target',calculate:'Calculation',spice:'SPICE',explain:'Worked solution',answer:'Answer'};
const stageAnchors = {input:'visual-section',detect:'visual-section',segment:'visual-section',graph:'topology-section',vision:'components-section',assemble:'topology-section',prepare:'analysis-section',target:'analysis-section',calculate:'analysis-section',spice:'spice-panel',explain:'worked-solution',answer:'answer-panel'};
const majorJSON = ['detections.json','candidate.json','readings.json','circuit.json','linear-state.json','reasoning-plan.json','calculation.json','answer.json','spice-result.json','run.json'];
const jsonLabels = {'detections.json':'YOLO','candidate.json':'Initial graph','readings.json':'Readings','circuit.json':'Final graph','linear-state.json':'Linear state','reasoning-plan.json':'Question target','calculation.json':'Calculation','answer.json':'Answer','spice-result.json':'SPICE','run.json':'Run record'};
const ns='http://www.w3.org/2000/svg';
const colors=['#326e91','#47794e','#8b5d84','#a76d38','#2f7b79','#6b6898'];

function show(node, visible=true){node.hidden=!visible;}
function setText(node,value){const text=value==null?'':String(value);if(node.textContent!==text)node.textContent=text;}
function pretty(value){return JSON.stringify(value,null,2);}
function artifact(snapshot,path){return(snapshot.artifacts||[]).find(a=>a.path===path);}
function versioned(a){return a.url+(a.url.includes('?')?'&':'?')+'v='+encodeURIComponent(a.version??'');}
function statusTitle(s){if(s.mode==='saved')return 'Saved rehearsal';if(s.status==='failed')return 'Run failed';if(s.answer_status==='insufficient_evidence')return 'Not answered';if(s.status==='completed')return s.answer_status==='answered'?'Run completed':'Run finished';return s.status==='queued'?'Queued':'Processing';}
function duration(s){const n=Math.max(0,Math.floor(Number(s)||0));return Math.floor(n/60)+':'+String(n%60).padStart(2,'0');}
function terminal(s){return s.mode==='saved'||s.status==='completed'||s.status==='failed';}

async function request(url,options={}){
  const response=await fetch(url,{cache:'no-store',...options});
  if(!response.ok){let message;try{const d=await response.json();message=d.error||d.message;}catch{message=await response.text().catch(()=>null);}throw new Error(message||`Request failed (${response.status})`);}
  return response.json();
}

function error(message,form=false){const target=form?$('form-error'):$('connection-error');setText(target,message);show(target,Boolean(message));}
function updateStartButton(){const selected=Boolean($('image-input').files.length||$('sample-select').value),busy=Boolean(app.health?.active_run_id)||Boolean(app.snapshot&&app.snapshot.mode==='live'&&!terminal(app.snapshot));$('run-button').disabled=app.submitting||!app.health?.ready||!selected||!$('question-input').value.trim()||busy;$('run-button').title=busy?'A circuit is still running. Open it in History to see its progress.':'';}

async function health(){
  try{const data=await request('/api/health');app.health=data;const node=$('health');node.className='health '+(data.ready?'ready':'unready');setText(node,data.ready?'Pipeline ready':(data.message||'Pipeline not ready'));node.title=data.ready?'Configured pipeline checks passed.':data.message||'';

    updateStartButton();if(!data.ready||data.active_run_id)setTimeout(health,4000);
  }catch(e){app.health={ready:false};setText($('health'),'Demo server unavailable');$('health').className='health unready';updateStartButton();setTimeout(health,5000);}
}

function imagePreview(url,caption){const p=$('upload-preview');p.src=url;show(p);$('drop-zone').classList.add('has-image');setText($('file-caption'),caption);show($('file-caption'));}
function setUpload(file){if(app.localPreview)URL.revokeObjectURL(app.localPreview);app.localPreview=null;if(file){$('sample-select').value='';show($('prepared-note'),false);app.localPreview=URL.createObjectURL(file);imagePreview(app.localPreview,file.name);}updateStartButton();}

$('image-input').addEventListener('change',()=>setUpload($('image-input').files[0]));
$('question-input').addEventListener('input',updateStartButton);
$('sample-select').addEventListener('change',()=>{
  const sample=app.samples.find(s=>s.id===$('sample-select').value);if(!sample){updateStartButton();return;}
  $('image-input').value='';if(app.localPreview)URL.revokeObjectURL(app.localPreview);app.localPreview=null;
  $('question-input').value=sample.question;imagePreview(sample.image_url,sample.title+' · known prepared example');show($('prepared-note'));error('',true);updateStartButton();
});
for(const name of ['dragenter','dragover'])$('drop-zone').addEventListener(name,e=>{e.preventDefault();$('drop-zone').classList.add('dragging');});
for(const name of ['dragleave','drop'])$('drop-zone').addEventListener(name,e=>{e.preventDefault();$('drop-zone').classList.remove('dragging');});
$('drop-zone').addEventListener('drop',e=>{const file=e.dataTransfer.files[0];if(!file)return;const list=new DataTransfer();list.items.add(file);$('image-input').files=list.files;setUpload(file);});
$('upload-preview').addEventListener('click',()=>zoom($('upload-preview').src,'Selected input circuit'));
function newCircuit(updateHistory=true){
  app.navigation++;
  clearTimeout(app.timer);app.current=null;app.snapshot=null;app.jsonToken++;
  if(app.localPreview)URL.revokeObjectURL(app.localPreview);app.localPreview=null;
  $('run-form').reset();$('history-select').value='';$('backup-select').value='';
  $('upload-preview').removeAttribute('src');show($('upload-preview'),false);
  $('drop-zone').classList.remove('has-image','dragging');show($('file-caption'),false);show($('prepared-note'),false);
  $('input-section').classList.remove('collapsed');show($('input-section'));show($('run-section'),false);
  error('');error('',true);
  if(updateHistory&&(location.search||location.hash))history.pushState(null,'',location.pathname);
  updateStartButton();health();window.scrollTo({top:0,behavior:'instant'});
}
$('new-input').addEventListener('click',()=>newCircuit());
window.addEventListener('popstate',()=>{
  const params=new URL(location.href).searchParams,backup=params.get('backup'),run=params.get('run');
  if(backup||run){$('input-section').classList.add('collapsed');openRun(backup||run,backup?'saved':'live');}
  else newCircuit(false);
});

$('run-form').addEventListener('submit',async e=>{
  e.preventDefault();error('',true);app.submitting=true;updateStartButton();
  const navigation=app.navigation;
  try{
    const payload=new FormData();payload.append('question',$('question-input').value.trim());
    const sample=$('sample-select').value;if(sample)payload.append('sample_id',sample);else if($('image-input').files[0])payload.append('image',$('image-input').files[0]);else throw new Error('Choose an image or prepared example.');
    const result=await request('/api/runs',{method:'POST',body:payload});
    if(app.navigation!==navigation){refreshHistory();health();return;}
    await openRun(result.id,'live');$('input-section').classList.add('collapsed');$('run-section').scrollIntoView({block:'start'});refreshHistory();
  }catch(e){error(e.message,true);}finally{app.submitting=false;updateStartButton();}
});

async function openRun(id,mode='live'){
  app.navigation++;
  clearTimeout(app.timer);app.current={id,mode};app.snapshot=null;app.selectedComponent=null;app.selectedNet=null;app.jsonPath=null;app.jsonVersion=null;app.overlay='final';app.jsonToken++;app.visual='input';app.inspector='results';app.follow=true;
  for(const node of document.querySelectorAll('[data-signature]'))delete node.dataset.signature;
  $('json-view').textContent='';$('artifact-select').replaceChildren();$('json-shortcuts').replaceChildren();
  for(const f of document.querySelectorAll('[data-image]')){show(f,false);f.querySelector('img').removeAttribute('src');delete f.querySelector('img').dataset.artifactSrc;}
  for(const id of ['components-section','topology-section','analysis-section','json-section','target-panel','linear-panel','equations-panel','answer-panel','spice-panel','worked-solution'])show($(id),false);
  $('component-table').tBodies[0].replaceChildren();$('net-table').tBodies[0].replaceChildren();$('topology-frame').replaceChildren();
  const url=new URL(location.href);url.search='';url.searchParams.set(mode==='saved'?'backup':'run',id);history.replaceState(null,'',url);if(mode==='live')localStorage.setItem('kbot-demo-run',id);
  error('');show($('run-section'));await poll();
}

async function poll(){
  const current=app.current;if(!current)return;
  try{
    const snapshot=await request(current.mode==='saved'?'/api/backups/'+encodeURIComponent(current.id):'/api/runs/'+encodeURIComponent(current.id));
    if(app.current!==current)return;
    error('');app.snapshot=snapshot;render(snapshot);updateStartButton();
    if(!terminal(snapshot))app.timer=setTimeout(poll,1000);else refreshHistory();
  }catch(e){if(app.current===current){error('Connection or run error: '+e.message+'. Last received artifacts remain visible.');app.timer=setTimeout(poll,3000);}}
}

function render(s){
  setText($('mode-badge'),s.mode==='saved'?'SAVED BACKUP':'LIVE RUN');$('mode-badge').className='badge '+(s.mode==='saved'?'saved':'');
  setText($('run-id'),s.id);setText($('run-heading'),s.label||'Circuit run');setText($('run-question'),s.question||'');setText($('elapsed'),duration(s.elapsed_seconds));setText($('run-status'),statusTitle(s));
  show($('saved-warning'),s.mode==='saved');show($('run-error'),Boolean(s.error));setText($('run-error'),s.error||'');
  renderStages(s);renderImages(s);renderReaderImages(s);renderComponents(s);renderAnalysis(s);renderWorkedSolution(s);renderSpice(s);renderArtifacts(s);
  const con=$('console-output'),nearEnd=con.scrollHeight-con.scrollTop-con.clientHeight<35;setText(con,s.console||'');if(nearEnd)con.scrollTop=con.scrollHeight;
  applyWorkbench(s);
  const p=s.reader_progress,reading=s.stages?.find(stage=>stage.id==='vision')?.status==='running';show($('reader-progress'),reading&&Boolean(p?.mode));if(reading&&p){setText($('reader-progress'),p.mode.includes('single')?'Reading diagram…':(p.total?`Diagram reading: ${p.completed} / ${p.total} component observations complete.`:'Diagram reading in progress.'));}
}

function renderStages(s){
  const rail=$('stage-rail'),byId=new Map((s.stages||[]).map(x=>[x.id,x]));
  const visibleStages=Object.entries(stageLabels).filter(([id])=>id!=='explain'||byId.has(id)),signature=visibleStages.map(([id])=>id).join('|');if(rail.dataset.stageSignature!==signature){rail.dataset.stageSignature=signature;rail.replaceChildren();for(const [id,label]of visibleStages){const li=document.createElement('li');li.dataset.stage=id;const a=document.createElement('a');a.href='#'+stageAnchors[id];a.textContent=label;a.style.color='inherit';a.style.textDecoration='none';li.append(a,document.createElement('small'));rail.append(li);}}
  for(const li of rail.children){const st=byId.get(li.dataset.stage)||{status:'pending'};li.className=st.status||'pending';li.title=st.label||stageLabels[li.dataset.stage];li.setAttribute('aria-label',(st.label||stageLabels[li.dataset.stage])+': '+(st.status||'pending'));setText(li.querySelector('small'),st.seconds!=null?Number(st.seconds).toFixed(2)+' s':({pending:'Waiting',running:'Running',ready:'Artifact ready',skipped:'Not run',failed:'Failed'}[st.status]||'Waiting'));}
}

function renderImages(s){
  const source=artifact(s,'source.png');
  const initial=artifact(s,'geometry/pin-net-overlay.png')||artifact(s,'candidate-overlay.png');const final=artifact(s,'circuit.png');
  const graph=app.overlay==='initial'?initial:(final||initial);
  const images={source:source||{url:s.input_url,version:s.id},detect:artifact(s,'detections-overlay.png'),mask:artifact(s,'sam/mask.png'),graph};
  for(const [key,a]of Object.entries(images)){const figure=document.querySelector(`[data-image="${key}"]`);if(!a?.url){show(figure,false);continue;}const img=figure.querySelector('img'),src=versioned(a);if(img.dataset.artifactSrc!==src){img.dataset.artifactSrc=src;img.src=src;}show(figure);figure.querySelector('.image-button').onclick=()=>zoom(src,figure.querySelector('strong').textContent);}
  const isFinal=Boolean(final&&app.overlay!=='initial');setText($('graph-image-label'),isFinal?'Final pin/net overlay':'Initial pin/net contacts');setText($('graph-image-caption'),isFinal?'Finalized after diagram reading':'Observed contacts before visual reading');
  show($('overlay-choice'),Boolean(initial&&final));for(const b of $('overlay-choice').children)b.classList.toggle('active',b.dataset.overlay===(isFinal?'final':'initial'));
}
for(const b of $('overlay-choice').children)b.addEventListener('click',()=>{app.overlay=b.dataset.overlay;if(app.snapshot)renderImages(app.snapshot);});

function renderReaderImages(s){
  const mainPaths=new Set();
  const files=(s.artifacts||[]).filter(a=>a.kind==='image'&&!mainPaths.has(a.path));show($('reader-inputs'),Boolean(files.length));
  for(const figure of document.querySelectorAll('[data-reader-image]')){const a=files.find(x=>x.path===figure.dataset.readerImage);show(figure,Boolean(a));if(!a)continue;const img=figure.querySelector('img'),src=versioned(a);if(img.dataset.artifactSrc!==src){img.dataset.artifactSrc=src;img.src=src;}figure.querySelector('button').onclick=()=>zoom(src,figure.querySelector('strong').textContent);}
  const extras=files.filter(a=>!['reading-sheet.png','measurement-sites.png'].includes(a.path)),select=$('image-artifact-select');show($('extra-image-control'),Boolean(extras.length));
  const signature=extras.map(a=>a.path).join('|');if(select.dataset.signature!==signature){const previous=select.value;select.dataset.signature=signature;select.replaceChildren(...extras.map(a=>new Option(a.path,a.path)));if(extras.some(a=>a.path===previous))select.value=previous;}
  const selected=extras.find(a=>a.path===select.value);show($('extra-image-figure'),Boolean(selected));if(selected){const img=$('extra-image-preview'),src=versioned(selected);if(img.dataset.artifactSrc!==src){img.dataset.artifactSrc=src;img.src=src;}setText($('extra-image-caption'),selected.path);$('extra-image-button').onclick=()=>zoom(src,selected.path);}
}
$('image-artifact-select').addEventListener('change',()=>{if(app.snapshot)renderReaderImages(app.snapshot);});

function cell(row,text,cls){const td=document.createElement('td');if(cls)td.className=cls;td.textContent=text==null?'—':String(text);row.append(td);return td;}
function renderComponents(s){
  const data=s.data||{},graph=data.circuit||data.candidate,det=data.detections,components=graph?.components||det?.components||[];
  if(app.selectedNet&&!graph?.nets?.some(n=>n.id===app.selectedNet))app.selectedNet=null;
  if(app.selectedComponent&&!components.some(c=>c.id===app.selectedComponent))app.selectedComponent=null;
  show($('components-section'),components.length>0);if(!components.length)return;
  const pins=new Map((graph?.pins||[]).map(p=>[p.id,p]));const readings=new Map([...Object.entries(data.reader_observations||{}).map(([id,c])=>[id,c]),...(data.readings?.components||[]).map(c=>[c.id,c])]);const detects=new Map((det?.components||[]).map(c=>[c.id,c]));
  const signature=JSON.stringify([components,graph?.pins,data.readings,data.reader_observations,app.selectedComponent]);const tbody=$('component-table').tBodies[0];
  if(tbody.dataset.signature!==signature){tbody.dataset.signature=signature;tbody.replaceChildren();for(const c of components){const d=detects.get(c.id)||{},r=readings.get(c.id)||{},row=document.createElement('tr');row.classList.toggle('selected-row',app.selectedComponent===c.id);const idCell=cell(row,'');const button=document.createElement('button');button.type='button';button.className='component-select';button.textContent=c.id;button.addEventListener('click',()=>selectComponent(c.id));idCell.append(button);cell(row,(c.type||'').replaceAll('_',' '));const confidence=c.confidence??d.confidence;cell(row,confidence==null?'—':(confidence*100).toFixed(1)+'%');const value=c.value??r.value,unit=c.unit??r.unit;cell(row,value==null?'Not yet read':String(value)+(unit&&!String(value).includes(String(unit))?' '+unit:''),value==null?'null':'value-cell');
      const pcell=cell(row,'');const owned=(graph?.pins||[]).filter(p=>p.component_id===c.id);if(!owned.length)pcell.textContent='Not yet assembled';for(const p of owned){const span=document.createElement('span');span.className='pin';span.textContent=p.id+((p.role||r.pin_roles?.[p.id])?' · '+(p.role||r.pin_roles?.[p.id]):'')+' → '+(p.net_id||'unresolved');pcell.append(span);}tbody.append(row);}}
  const finalized=Boolean(data.circuit);setText($('graph-state'),graph?(finalized?'Finalized · '+(graph.status||'unverified'):'Initial candidate'):'Detections only');
  const nets=graph?.nets||[];setText($('net-count'),nets.length?'('+nets.length+')':'');show($('nets-details'),Boolean(nets.length));const nt=$('net-table').tBodies[0];const nsig=JSON.stringify(nets);if(nt.dataset.signature!==nsig){nt.dataset.signature=nsig;nt.replaceChildren();for(const n of nets){const row=document.createElement('tr');cell(row,n.id);cell(row,n.ground?'Yes':'No');cell(row,(n.pins||[]).join(', ')||'No component pins');nt.append(row);}}
  const issues=data.circuit?.electrical_issues||[];show($('issues-panel'),Boolean(issues.length));setText($('issues-text'),issues.length?pretty(issues):'');
  show($('topology-section'),Boolean(graph?.pins?.length&&nets.length));if(graph?.pins?.length&&nets.length)renderTopology(graph,finalized,s);
}

function selectComponent(id){app.selectedNet=null;app.selectedComponent=app.selectedComponent===id?null:id;show($('clear-selection'),Boolean(app.selectedComponent));if(app.snapshot){renderComponents(app.snapshot);app.visual='connections';app.follow=false;applyWorkbench(app.snapshot);}}
$('clear-selection').addEventListener('click',()=>{app.selectedComponent=null;app.selectedNet=null;show($('clear-selection'),false);if(app.snapshot){renderComponents(app.snapshot);app.visual='connections';app.follow=false;applyWorkbench(app.snapshot);}});
function svgElement(tag,attrs={},text){const node=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))node.setAttribute(k,String(v));if(text!=null)node.textContent=text;return node;}
function renderTopology(graph,finalized,s){
  const candidate=s.data?.candidate||{},source=artifact(s,'source.png'),sourceURL=source?versioned(source):s.input_url;
  const fragments=graph.wire_fragments?.length?graph.wire_fragments:(candidate.wire_fragments||[]),size=graph.image?.size||candidate.image?.size;
  if(!size?.length||!sourceURL){show($('topology-section'),false);return;}
  const signature=JSON.stringify([graph.components,graph.pins,graph.nets,fragments,app.selectedComponent,app.selectedNet,finalized,sourceURL]);
  const frame=$('topology-frame');if(frame.dataset.signature===signature)return;frame.dataset.signature=signature;
  const width=size[0],height=size[1],pins=graph.pins||[],components=graph.components||[],nets=graph.nets||[],fragmentMap=new Map(fragments.map(f=>[f.id,f])),candidateNetMap=new Map((candidate.nets||[]).map(n=>[n.id,n]));
  const ownIds=n=>n.fragment_ids?.length?n.fragment_ids:(n.source_net_ids||[]).flatMap(id=>candidateNetMap.get(id)?.fragment_ids||[]);
  const selectedComponentNets=new Set(pins.filter(p=>p.component_id===app.selectedComponent).map(p=>p.net_id));
  const active=n=>!app.selectedNet&&!app.selectedComponent||app.selectedNet===n.id||app.selectedComponent&&selectedComponentNets.has(n.id);
  const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':(finalized?'Final':'Initial candidate')+' net connections on the source circuit',preserveAspectRatio:'xMidYMid meet'});
  svg.append(svgElement('image',{href:sourceURL,x:0,y:0,width,height,opacity:app.selectedNet||app.selectedComponent ? .36 : .68}));
  const scale=width/850,stroke=Math.max(2,3*scale),radius=Math.max(3,4*scale);
  nets.forEach((net,i)=>{const color=colors[i%colors.length],group=svgElement('g',{opacity:active(net)?1:.10});
    for(const id of ownIds(net)){const fragment=fragmentMap.get(id);if(!fragment?.polyline_px?.length)continue;const path=svgElement('polyline',{points:fragment.polyline_px.map(p=>p.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':stroke,'stroke-linecap':'round','stroke-linejoin':'round',class:'net-wire'});path.append(svgElement('title',{},net.id+' · '+id));path.addEventListener('click',()=>selectNet(net.id));group.append(path);}
    for(const pin of pins.filter(p=>p.net_id===net.id)){const point=pin.anchor_px;if(!point?.length)continue;const dot=svgElement('circle',{cx:point[0],cy:point[1],r:radius,fill:'white',stroke:color,'stroke-width':Math.max(1.5,2*scale),class:'net-pin'});dot.append(svgElement('title',{},pin.id+' · '+pin.component_id+(pin.role?' · '+pin.role:'')+' → '+net.id));dot.addEventListener('click',()=>selectNet(net.id));group.append(dot);}
    svg.append(group);
  });
  if(app.selectedNet||app.selectedComponent){const joined=new Set(pins.filter(p=>app.selectedNet?p.net_id===app.selectedNet:p.component_id===app.selectedComponent).map(p=>p.component_id));for(const component of components){if(!joined.has(component.id)||!component.bbox?.length)continue;const [x1,y1,x2,y2]=component.bbox,color=app.selectedNet?colors[Math.max(0,nets.findIndex(n=>n.id===app.selectedNet))%colors.length]:'#176b68';svg.append(svgElement('rect',{x:x1-3,y:y1-3,width:x2-x1+6,height:y2-y1+6,fill:'none',stroke:color,'stroke-width':stroke,'stroke-dasharray':`${4*scale} ${2*scale}`}));const label=svgElement('text',{x:x1,y:Math.max(13*scale,y1-7*scale),fill:color,'font-size':Math.max(10,13*scale),'font-weight':700,'paint-order':'stroke',stroke:'white','stroke-width':3*scale},component.id);svg.append(label);}}
  frame.replaceChildren(svg);
  const legend=$('net-legend');legend.replaceChildren();for(const [i,net]of nets.entries()){const owners=[...new Set(pins.filter(p=>p.net_id===net.id).map(p=>p.component_id))],button=document.createElement('button');button.type='button';button.className='net-chip'+(app.selectedNet===net.id?' active':'');button.style.setProperty('--net-color',colors[i%colors.length]);button.setAttribute('aria-pressed',String(app.selectedNet===net.id));const dot=document.createElement('span');dot.className='net-swatch';const label=document.createElement('span');label.textContent=net.id+(net.ground?' · ground':'')+' → '+(owners.join(', ')||'no component pins');button.append(dot,label);if(!ownIds(net).some(id=>fragmentMap.has(id))){const note=document.createElement('small');note.textContent='No recorded wire path';button.append(note);}button.addEventListener('click',()=>selectNet(net.id));legend.append(button);}
  show($('clear-selection'),Boolean(app.selectedNet||app.selectedComponent));
  setText($('topology-caption'),(finalized?'Final graph':'Initial contacts')+' · '+((graph.status||'unverified').includes('unverified')?'Unverified':(graph.status||'unverified').replaceAll('_',' ')));
}
function selectNet(id){app.selectedComponent=null;app.selectedNet=app.selectedNet===id?null:id;if(app.snapshot){renderComponents(app.snapshot);app.visual='connections';app.follow=false;applyWorkbench(app.snapshot);}}

function renderMath(latex,node){
  if(!latex||!window.katex)return false;
  if(node.dataset.latex!==latex){window.katex.render(latex,node,{throwOnError:false,displayMode:true,strict:'ignore'});node.dataset.latex=latex;}
  return true;
}
function renderTargetPlan(plan){
  const record=$('target-record'),signature=JSON.stringify(plan);if(record.dataset.signature===signature)return;record.dataset.signature=signature;record.replaceChildren();
  const targets=plan.targets||[];if(targets.length){const table=document.createElement('table'),head=document.createElement('thead'),header=document.createElement('tr');for(const label of ['Requested quantity','Interpretation','Reference']){const th=document.createElement('th');th.textContent=label;header.append(th);}head.append(header);const body=document.createElement('tbody');for(const target of targets){const row=document.createElement('tr');cell(row,target.label||target.name||target.id||'—');cell(row,(target.kind||'').replaceAll('_',' '));const reference=Object.entries(target).filter(([key])=>!['label','name','id','kind'].includes(key)).map(([key,value])=>key.replaceAll('_',' ')+': '+(typeof value==='object'?JSON.stringify(value):value)).join(' · ');cell(row,reference||'—');body.append(row);}table.append(head,body);record.append(table);}
  if(plan.assumptions?.length){const p=document.createElement('p');p.className='small';p.textContent='Assumptions: '+plan.assumptions.join(' · ');record.append(p);}
}

function renderAnalysis(s){
  const d=s.data||{},plan=d.reasoning_plan,linear=d.linear_state,calculation=d.calculation,answer=d.answer,math=d.display_math||{};
  show($('analysis-section'),Boolean(plan||linear||calculation||answer));show($('target-panel'),Boolean(plan));if(plan){setText($('target-method'),plan.method||'');renderTargetPlan(plan);}
  show($('linear-panel'),Boolean(linear));if(linear){const bits=[['node potentials',Object.keys(linear.node_voltages||{}).length],['terminal currents',Object.keys(linear.terminal_currents||{}).length],['printed references',Object.keys(linear.printed_references||{}).length]];const sig=JSON.stringify(bits);if($('linear-summary').dataset.signature!==sig){$('linear-summary').dataset.signature=sig;$('linear-summary').replaceChildren(...bits.map(([label,n])=>{const span=document.createElement('span');span.textContent=n+' '+label;return span;}));}}
  const analyses=calculation?.analyses||[];show($('equations-panel'),Boolean(analyses.length));const eqSig=JSON.stringify([analyses,math.equations,Boolean(window.katex)]);if($('equations').dataset.signature!==eqSig){$('equations').dataset.signature=eqSig;$('equations').replaceChildren();for(const analysis of analyses){const equations=analysis.system?.equations||analysis.equations||[],latexGroup=(math.equations||[]).find(g=>g.target_id===analysis.target_id),latexEquations=latexGroup?.equations||[];if(!equations.length)continue;const group=document.createElement('div');group.className='equation-group';const heading=document.createElement('h4');heading.textContent='Target '+(analysis.target_id||'')+(analysis.exact_result!=null?' · exact result '+analysis.exact_result:'');group.append(heading);if(latexEquations.length&&window.katex){const block=document.createElement('div');block.className='latex-equations';for(const equation of latexEquations){const line=document.createElement('div');renderMath(equation,line);block.append(line);}group.append(block);}else{const pre=document.createElement('pre');pre.textContent=equations.join('\n');group.append(pre);}if(analysis.system?.exact_residuals_zero===true){const p=document.createElement('p');p.className='residual';p.textContent='Exact equation residuals: zero';group.append(p);}$('equations').append(group);}}
  show($('answer-panel'),Boolean(answer));if(answer){const ok=answer.status==='answered';$('answer-panel').classList.toggle('not-answered',!ok);setText($('answer-status'),answer.status||s.answer_status||'');$('answer-status').className='badge '+(ok?'':'warning');setText($('answer-text'),answer.answer||'');const rendered=renderMath(math.answer,$('answer-math'));show($('answer-math'),rendered);show($('answer-text'),!rendered);const qs=answer.quantities||[],signature=JSON.stringify(qs);if($('quantities').dataset.signature!==signature){$('quantities').dataset.signature=signature;$('quantities').replaceChildren();if(qs.length){const table=document.createElement('table');const head=document.createElement('thead');const row=document.createElement('tr');for(const label of ['Quantity','Value','Unit','Reference direction']){const th=document.createElement('th');th.textContent=label;row.append(th);}head.append(row);const tbody=document.createElement('tbody');for(const q of qs){const tr=document.createElement('tr');let value=q.expression??q.real??q.value??'—';if(q.imaginary) value=String(q.real??0)+(q.imaginary>=0?' + ':' − ')+Math.abs(q.imaginary)+'j';cell(tr,q.name||q.id);cell(tr,value);cell(tr,q.unit);cell(tr,q.reference_direction);tbody.append(tr);}table.append(head,tbody);$('quantities').append(table);}}const missing=answer.missing_evidence||[];show($('missing-evidence'),Boolean(missing.length));setText($('missing-evidence').querySelector('pre'),missing.length?pretty(missing):'');}
}


function renderWorkedSolution(s){
  const explanation=s.data?.explanation,stage=s.stages?.find(row=>row.id==='explain'),active=!terminal(s),hasCalculation=Boolean(s.data?.calculation),panel=$('worked-solution');
  const visible=Boolean(explanation||hasCalculation||stage?.status==='running'||terminal(s));show(panel,visible);if(!visible)return;show($('analysis-section'));
  const signature=JSON.stringify([explanation,active,hasCalculation,stage?.status,s.mode]);if(panel.dataset.signature===signature)return;panel.dataset.signature=signature;
  const badge=$('worked-status'),message=$('worked-message'),content=$('worked-content');show(content,false);show(message,true);setText(message,'');badge.className='badge neutral';
  if(!explanation){
    if(active){setText(badge,'Pending');setText(message,'Preparing the worked explanation…');}
    else{setText(badge,'Not generated');setText(message,stage&&stage.status!=='skipped'&&s.mode!=='saved'?'Worked explanation was not generated for this run.':'Worked explanation was not generated for this earlier run.');}
    return;
  }
  if(explanation.status!=='available'){setText(badge,'Unavailable');badge.className='badge warning';setText(message,explanation.error||'Worked solution is unavailable for this run.');return;}
  setText(badge,'Available');badge.className='badge';show(message,false);show(content);setText($('worked-method'),explanation.method||'');show($('worked-method'),Boolean(explanation.method));
  const list=(id,items)=>{const values=[...new Set((items||[]).filter(value=>typeof value==='string'&&value.trim()))],block=$(id),ul=block.querySelector('ul');show(block,Boolean(values.length));ul.replaceChildren(...values.map(value=>{const li=document.createElement('li');li.textContent=value;return li;}));};
  list('worked-assumptions',explanation.assumptions);list('worked-checks',explanation.checks);list('worked-limitations',explanation.limitations);
  const steps=$('worked-steps');steps.replaceChildren();for(const step of explanation.steps||[]){const li=document.createElement('li');if(step.title){const title=document.createElement('h3');title.textContent=step.title;li.append(title);}if(step.text){const text=document.createElement('p');text.textContent=step.text;li.append(text);}for(const equation of step.equations||[]){if(typeof equation.latex!=='string'||!equation.latex.trim()){if(equation.text){const evidence=document.createElement('p');evidence.className='worked-evidence';evidence.textContent=equation.text.replace(/; observed value None\s*;/g,';').replaceAll(' -> ',' → ').replaceAll('ground;','ground connection;');li.append(evidence);}continue;}const block=document.createElement('div');block.className='worked-equation';if(equation.id)block.dataset.equationId=equation.id;if(equation.source)block.title=String(equation.source);if(!renderMath(equation.latex,block)){block.textContent=equation.latex;block.classList.add('literal-math');}li.append(block);}steps.append(li);}
}

function renderSpice(s){
  const result=s.data?.spice_result;show($('spice-panel'),Boolean(result));if(!result)return;show($('analysis-section'));
  const status=result.status||'Recorded',ok=status==='completed';setText($('spice-status'),status.replaceAll('_',' '));$('spice-status').className='badge '+(ok?'':status==='failed'?'error':'warning');
  const analysis=result.analysis||{},mode=analysis.mode,seconds=result.seconds;
  setText($('spice-summary'),[mode?mode.toUpperCase()+' analysis':null,analysis.frequency_hz!=null?formatNumber(analysis.frequency_hz)+' Hz'+(analysis.frequency_basis==='normalized_single_frequency_impedances'?' (normalized)':''):null,seconds!=null?Number(seconds).toFixed(2)+' s':null].filter(Boolean).join(' · '));
  const crosscheck=result.crosscheck;show($('spice-crosscheck'),Boolean(crosscheck));if(crosscheck){const status=crosscheck.status,checked=crosscheck.checked_quantities;let text=status==='passed'?'Numerical results agree'+(checked!=null?' · '+checked+' checked quantities':''):status==='mismatch'?'Numerical results disagree — review the calculation and simulation.':'Numerical agreement not checked.';if(status==='mismatch'&&crosscheck.mismatches?.length)text+=' '+crosscheck.mismatches.map(row=>row.quantity).filter(Boolean).join(', ');setText($('spice-crosscheck'),text);$('spice-crosscheck').className=status==='mismatch'?'notice warning':'small numerical-agreement';}
  const notes=[result.error,result.reason,result.message,analysis.frequency_basis==='normalized_single_frequency_impedances'?'Normalized reference frequency for supplied phasor impedances.':null,ok?'Simulation of the extracted graph; image interpretation is not independently verified.':null].filter(Boolean);setText($('spice-note'),notes.join(' '));show($('spice-note'),Boolean(notes.length));
  renderSpiceValues('spice-voltages',result.net_voltages||{},mode,'net');
  const branches={},directions={};for(const [cid,probe]of Object.entries(result.mapping?.component_probes||{})){const value=result.terminal_currents?.[cid+'.'+probe.primary_terminal];if(!value)continue;branches[cid]=value;directions[cid]=cid+' · '+probe.primary_terminal+(probe.secondary_terminal?' → '+probe.secondary_terminal:'');}renderSpiceValues('spice-currents',branches,mode,'component',directions);
  renderSourceParameters(result.source_parameters||{});
  const netlist=(s.artifacts||[]).find(a=>a.path==='circuit.spice')||(s.artifacts||[]).find(a=>/\.(cir|spice)$/i.test(a.path));show($('spice-netlist'),Boolean(netlist));if(netlist){$('spice-netlist').href=netlist.url+'?download=1';$('spice-netlist').setAttribute('download',netlist.path.split('/').at(-1));}
}
function numericParts(value){const real=Number(value.real),imaginary=Number(value.imaginary??0);return {real,imaginary,magnitude:Math.hypot(real,imaginary),phase:Math.atan2(imaginary,real)*180/Math.PI};}
function renderSpiceValues(id,values,mode,kind,labels={}){
  const container=$(id),signature=JSON.stringify([values,labels,mode,app.selectedNet,app.selectedComponent]);if(container.dataset.signature===signature)return;container.dataset.signature=signature;container.replaceChildren();
  const rows=Object.entries(values);if(!rows.length){const p=document.createElement('p');p.className='small';p.textContent='No recorded results.';container.append(p);return;}
  const table=document.createElement('table'),head=document.createElement('thead'),header=document.createElement('tr'),ac=mode==='ac';for(const name of [kind==='net'?'Net':'Component branch',...(ac?['Magnitude','Phase']:['Value'])]){const th=document.createElement('th');th.textContent=name;header.append(th);}head.append(header);const body=document.createElement('tbody');
  for(const [name,value]of rows){const row=document.createElement('tr'),component=name.split('.')[0],selected=kind==='net'?app.selectedNet===name:app.selectedComponent===component;row.classList.toggle('selected-row',selected);const label=cell(row,''),button=document.createElement('button');button.type='button';button.className='component-select';button.textContent=labels[name]||name;button.addEventListener('click',()=>{if(kind==='net')selectNet(name);else selectComponent(component);if(app.snapshot)renderSpice(app.snapshot);});label.append(button);const n=numericParts(value);if(ac){cell(row,formatNumber(n.magnitude)+' '+(value.unit||''));cell(row,n.magnitude?formatNumber(n.phase)+'°':'—');}else cell(row,formatNumber(n.real)+' '+(value.unit||''));body.append(row);}table.append(head,body);container.append(table);
}
function renderSourceParameters(parameters){
  const rows=Object.entries(parameters);show($('source-parameters'),Boolean(rows.length));const container=$('source-parameter-table'),signature=JSON.stringify(parameters);if(container.dataset.signature===signature)return;container.dataset.signature=signature;container.replaceChildren();if(!rows.length)return;
  const table=document.createElement('table'),head=document.createElement('thead'),header=document.createElement('tr');for(const name of ['Source','Observed kind','SPICE value / magnitude','Phase','Angular frequency','Convention']){const th=document.createElement('th');th.textContent=name;header.append(th);}head.append(header);const body=document.createElement('tbody');for(const [name,source]of rows){const row=document.createElement('tr'),n=source.phasor?numericParts(source.phasor):null;cell(row,name);cell(row,source.kind);cell(row,n?formatNumber(source.kind==='dc'?n.real:n.magnitude)+' '+(source.phasor.unit||''):'—');cell(row,n&&source.kind!=='dc'?formatNumber(n.phase)+'°':'—');cell(row,source.omega_rad_s!=null?formatNumber(source.omega_rad_s)+' rad/s':'—');cell(row,source.convention||'unspecified');body.append(row);}table.append(head,body);container.append(table);
}

function formatNumber(value){return typeof value==='number'?Number(value.toPrecision(6)).toString():String(value??'—');}

function renderArtifacts(s){
  const files=(s.artifacts||[]).filter(a=>a.kind==='json'||a.kind==='text'||/\.(json|md|log)$/.test(a.path));show($('json-section'),Boolean(files.length));if(!files.length)return;
  const select=$('artifact-select'),signature=files.map(a=>a.path).join('|');if(select.dataset.signature!==signature){select.dataset.signature=signature;select.replaceChildren(...files.map(a=>{const opt=document.createElement('option');opt.value=a.path;opt.textContent=a.path;return opt;}));}
  const shortcuts=$('json-shortcuts');const shorts=majorJSON.filter(p=>files.some(a=>a.path===p));const ssig=shorts.join('|');if(shortcuts.dataset.signature!==ssig){shortcuts.dataset.signature=ssig;shortcuts.replaceChildren(...shorts.map(path=>{const button=document.createElement('button');button.type='button';button.dataset.path=path;button.textContent=jsonLabels[path];button.addEventListener('click',()=>selectArtifact(path));return button;}));}
  if(!app.jsonPath||!files.some(a=>a.path===app.jsonPath))app.jsonPath=files.some(a=>a.path==='detections.json')?'detections.json':files[0].path;
  select.value=app.jsonPath;for(const button of shortcuts.children)button.classList.toggle('active',button.dataset.path===app.jsonPath);
  const current=files.find(a=>a.path===app.jsonPath);if(current&&app.jsonVersion!==current.path+':'+current.version){app.jsonVersion=current.path+':'+current.version;loadArtifact(current);}
}
function selectArtifact(path){app.jsonPath=path;app.jsonVersion=null;if(app.snapshot)renderArtifacts(app.snapshot);}
$('artifact-select').addEventListener('change',()=>selectArtifact($('artifact-select').value));
async function loadArtifact(a){const token=++app.jsonToken;setText($('artifact-path'),a.path);$('download-json').href=a.url+(a.url.includes('?')?'&':'?')+'download=1';$('download-json').setAttribute('download',a.path.split('/').at(-1));try{const response=await fetch(versioned(a),{cache:'no-store'});if(!response.ok)throw new Error('Artifact is not yet readable');let text=await response.text();if(a.path.endsWith('.json')){try{text=pretty(JSON.parse(text));}catch{}}if(token===app.jsonToken){const pre=$('json-view'),scroll=pre.scrollTop;setText(pre,text);pre.scrollTop=scroll;}}catch(e){if(token===app.jsonToken){setText($('json-view'),e.message);app.jsonVersion=null;}}}
$('copy-json').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('json-view').textContent);setText($('copy-json'),'Copied');setTimeout(()=>setText($('copy-json'),'Copy'),1300);}catch{setText($('copy-json'),'Select text to copy');}});

function zoom(src,title){$('zoom-image').src=src;setText($('zoom-title'),title);$('image-dialog').showModal();}
$('close-dialog').addEventListener('click',()=>$('image-dialog').close());$('image-dialog').addEventListener('click',e=>{if(e.target===$('image-dialog'))$('image-dialog').close();});

async function refreshHistory(){try{const response=await request('/api/runs'),runs=Array.isArray(response)?response:(response.runs||[]);const select=$('history-select'),current=select.value;select.replaceChildren(new Option('Recent live runs…',''),...runs.map(r=>new Option((r.label||r.id)+' · '+r.status,r.id)));if([...select.options].some(o=>o.value===current))select.value=current;}catch{}}
$('history-select').addEventListener('change',()=>{if($('history-select').value){openRun($('history-select').value,'live');$('input-section').classList.add('collapsed');}});
$('backup-select').addEventListener('change',()=>{if($('backup-select').value){openRun($('backup-select').value,'saved');$('input-section').classList.add('collapsed');}});


function applyWorkbench(s){
  if(app.follow){const overlay=s.data?.circuit?'final':'initial';if(app.overlay!==overlay){app.overlay=overlay;renderImages(s);}const graph=s.data?.circuit||s.data?.candidate;if(graph?.pins?.length)app.visual='connections';else if(artifact(s,'sam/mask.png'))app.visual='mask';else if(artifact(s,'detections-overlay.png'))app.visual='detect';else app.visual='input';const interpreting=s.data?.explanation||s.data?.linear_state||s.data?.reasoning_plan||s.data?.calculation||s.data?.answer;app.inspector=interpreting?'results':s.data?.detections?'components':'results';}
  $('image-grid').className='comparison-stage visual-'+app.visual;
  for(const button of document.querySelectorAll('[data-visual]')){const on=button.dataset.visual===app.visual;button.classList.toggle('active',on);button.setAttribute('aria-pressed',String(on));}
  const selected=app.visual==='connections'?(!document.querySelector('[data-image="graph"]').hidden?document.querySelector('[data-image="graph"]'):$('topology-section')):document.querySelector('[data-image="'+(app.visual==='input'?'source':app.visual)+'"]');
  show($('canvas-empty'),!selected||selected.hidden);setText($('canvas-empty-text'),{input:'Waiting for the image',detect:'Waiting for detection',mask:'Waiting for the wire mask',connections:'Waiting for connections'}[app.visual]);
  const content=document.querySelector('.inspector-content');content.className='inspector-content inspect-'+app.inspector;
  for(const button of document.querySelectorAll('[data-inspector]')){const on=button.dataset.inspector===app.inspector;button.classList.toggle('active',on);button.setAttribute('aria-pressed',String(on));}
  const available=app.inspector==='results'?!$('analysis-section').hidden:app.inspector==='components'?!$('components-section').hidden:!$('nets-details').hidden;show($('inspector-empty'),!available);setText($('inspector-empty'),{results:'Waiting for results',components:'Waiting for component detections',nets:'Waiting for net connections'}[app.inspector]);
  show($('calculation-details'),!$('linear-panel').hidden||!$('equations-panel').hidden);show($('spice-details'),!$('spice-panel').hidden);
  $('follow-run').classList.toggle('active',app.follow);$('follow-run').setAttribute('aria-pressed',String(app.follow));
}
for(const button of document.querySelectorAll('[data-visual]'))button.addEventListener('click',()=>{app.follow=false;app.visual=button.dataset.visual;if(app.snapshot)applyWorkbench(app.snapshot);});
for(const button of document.querySelectorAll('[data-inspector]'))button.addEventListener('click',()=>{app.follow=false;app.inspector=button.dataset.inspector;if(app.snapshot)applyWorkbench(app.snapshot);});
$('follow-run').addEventListener('click',()=>{app.follow=!app.follow;if(app.snapshot)applyWorkbench(app.snapshot);});
$('stage-rail').addEventListener('click',event=>{const stage=event.target.closest('[data-stage]')?.dataset.stage;if(!stage)return;event.preventDefault();app.follow=false;if(stage==='graph'||stage==='assemble'){app.overlay=stage==='graph'?'initial':'final';if(app.snapshot)renderImages(app.snapshot);}app.visual={input:'input',detect:'detect',segment:'mask',graph:'connections',assemble:'connections'}[stage]||app.visual;app.inspector=['vision','assemble','detect'].includes(stage)?'components':['graph'].includes(stage)?'nets':'results';if(['prepare','calculate'].includes(stage))$('calculation-details').open=true;if(stage==='spice')$('spice-details').open=true;if(app.snapshot)applyWorkbench(app.snapshot);if(stage==='explain')$('worked-solution').scrollIntoView({block:'start',behavior:'smooth'});});
$('open-technical').addEventListener('click',()=>$('technical-drawer').showModal());$('close-technical').addEventListener('click',()=>$('technical-drawer').close());$('technical-drawer').addEventListener('click',event=>{if(event.target===$('technical-drawer'))$('technical-drawer').close();});
for(const id of ['history-select','backup-select'])$(id).addEventListener('change',()=>{$('history-menu').open=false;});

async function init(){
  health();refreshHistory();
  try{const response=await request('/api/samples');app.samples=Array.isArray(response)?response:(response.samples||[]);for(const s of app.samples){$('sample-select').append(new Option(s.title,s.id));$('backup-select').append(new Option(s.title+' · saved',s.id));}}catch(e){error('Could not load prepared sample list: '+e.message,true);}
  const url=new URL(location.href),backup=url.searchParams.get('backup'),run=url.searchParams.get('run');
  if(backup){$('input-section').classList.add('collapsed');openRun(backup,'saved');}
  else if(run){$('input-section').classList.add('collapsed');openRun(run,'live');}
}
init();
