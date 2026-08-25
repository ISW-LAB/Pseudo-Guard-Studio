/* PG-Label UI behaviour: canvas editing, API calls, training panel.
   Served at /static/js/app.js and loaded at the end of <body>, so the DOM already exists. */
const PALETTE=["#6366f1","#f59e0b","#22c55e","#ef4444","#a855f7","#06b6d4","#eab308","#84cc16","#ec4899","#f97316"];
/* canvas cannot resolve CSS var() — use literal hex so AI-box band colors actually render */
const BAND={green:"#34d399",amber:"#fbbf24",red:"#f87171"};
/* human-drawn / human-reviewed boxes are always BLACK — the palette stays for the class legend only */
const HUMAN_COL="#000000";
let CLASSES=[],IMAGES=[],HAS_AI=false,CAN_TRAIN=false,idx=0,trainPoll=null,COUNTS={},OWNER={},humanDirty=false,DEFAULTS={};
let IMG_CLASSES={},AI_CLASSES={},FILTER_CLASS=null;   // saved + AI-inferred class-ids + active filter
let ADAPT_CONT=true;   // server's live containment decision — it flips as the human seed grows
function mark(){humanDirty=true;}   // a genuine human edit → this image becomes human-owned seed on save
let img=new Image(),imgW=1,imgH=1,scale=1;
let boxes=[],curClass=0,sel=-1,drag=null;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const $=id=>document.getElementById(id);

/* ---------- api + toast ---------- */
const getJSON=p=>fetch(p).then(r=>r.json());
const postJSON=(p,b)=>fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}).then(r=>r.json());
function toast(msg,type){const t=document.createElement('div');t.className='toast '+(type||'');
  t.innerHTML=`<span>${type==='ok'?'✅':type==='err'?'⛔':type==='warn'?'⚠️':'ℹ️'}</span><div>${msg}</div>`;
  $('toasts').appendChild(t);setTimeout(()=>{t.style.opacity='0';t.style.transition='.4s';setTimeout(()=>t.remove(),400)},type==='err'?6000:3200);}

/* ---------- init / setup ---------- */
async function init(){
  const c=await getJSON('/api/config');
  if(c.needs_setup){ DEFAULTS=c.defaults||DEFAULTS; openSetup(); return; }   // show the setup screen (pre-filled)
  $('setup').classList.remove('open');
  CLASSES=c.classes;IMAGES=c.images;HAS_AI=c.has_ai;CAN_TRAIN=c.can_train;
  /* the methodology picker lives in this section, so the section stays visible even without a
     training env — only the training controls themselves are hidden then. */
  $('trainctl').style.display=CAN_TRAIN?'':'none';
  if(!CAN_TRAIN){$('trainsec').querySelector('.hd span').textContent='Methodology';   // no GPU env → the card is just the picker
    $('trainsec').title='training unavailable — no training interpreter or algorithm library found';}
  renderAdaptive(c.adaptive);
  await loadMethods();   // always populate the method dropdown (the cycle can train from scratch)
  renderPalette();renderImageList();
  $('imcount').textContent=IMAGES.length;
  if(IMAGES.length) load(0); else $('counter').textContent='no images';
  refreshStatus();
  resumePolling();                                   // reattach to a job still running (e.g. page reload)
}
async function resumePolling(){                       // if a train/cycle is mid-flight, re-arm its poller + Stop button
  try{const ts=await getJSON('/api/train/status');
    if(ts.state==='running'){$('trainlog').style.display='block';setJobButtons(true,'train');
      if(trainPoll)clearInterval(trainPoll);trainPoll=setInterval(pollTrain,1500);pollTrain();}}catch(e){}
  try{const cs=await getJSON('/api/cycle/status');
    if(cs.state==='running'){$('trainlog').style.display='block';setJobButtons(true,'cycle');
      if(cyclePoll)clearInterval(cyclePoll);cyclePoll=setInterval(pollCycle,1500);pollCycle();}}catch(e){}
}
function openSetup(){
  const d=DEFAULTS||{};
  if(!$('suImages').value) $('suImages').value = (IMAGES.length ? '' : (d.images||''));
  if(!$('suClasses').value) $('suClasses').value = (CLASSES.join(', ') || d.classes || '');
  if(!$('suLabels').value) $('suLabels').value = (d.labels||'');
  $('suErr').textContent=''; $('setup').classList.add('open');
  loadDatasets();
  $('suImages').focus(); $('suImages').select();
}
/* dataset presets: pick one → auto-fill paths/classes + seed N/class from that dataset's GT */
let PRESETS=[], SEL_PRESET=-1, SEED_LABELS='', SEED_PCT=0;
async function loadDatasets(){
  let r; try{ r=await getJSON('/api/datasets'); }catch(e){ $('dsPresets').innerHTML='<span class="k" style="color:var(--mut2)">no presets</span>'; return; }
  PRESETS=r.datasets||[];
  if(!PRESETS.length){ $('dsPresets').innerHTML='<span class="k" style="color:var(--mut2)">No preset datasets available</span>'; return; }
  $('dsPresets').innerHTML=PRESETS.map((p,i)=>
    `<div class="dscard${i===SEL_PRESET?' sel':''}" onclick="pickDataset(${i})" title="${p.classes}">
      <b>${p.name}</b><div class="meta">train ${p.train_images}${p.val_images?' · val '+p.val_images:''} · ${p.num_classes} classes</div></div>`).join('');
}
function pickDataset(i){SEL_PRESET=i;const p=PRESETS[i];
  $('suImages').value=p.images; $('suClasses').value=p.classes; $('suLabels').value=p.labels;
  SEED_LABELS=p.seed_labels; SEED_PCT=p.seed_percent||5;
  loadDatasets();   // refresh selected highlight
  $('suErr').textContent=`✓ ${p.name} selected — seeds ${SEED_PCT}% per class from ground truth when you press Start`;}
let BROWSE_PATH='';
function openBrowser(){ $('browser').classList.add('open'); browseTo($('suImages').value.trim()); }
function closeBrowser(){ $('browser').classList.remove('open'); }
async function browseTo(path){
  let d; try{ d=await getJSON('/api/browse?path='+encodeURIComponent(path||'')); }catch(e){ return; }
  BROWSE_PATH=d.path; $('bpath').textContent=d.path;
  const L=$('brows'); L.innerHTML='';
  if(d.parent!==null){ const r=document.createElement('div'); r.className='brow up'; r.textContent='⬆  ..';
    r.onclick=()=>browseTo(d.parent); L.appendChild(r); }
  (d.dirs||[]).forEach(x=>{ const r=document.createElement('div'); r.className='brow'; r.textContent='📁  '+x.name;
    // x.path is the server-joined child path (Windows separators included); the join is the fallback
    r.onclick=()=>browseTo(x.path || (d.path.replace(/\/+$/,'')+'/'+x.name)); L.appendChild(r); });
  if(!(d.dirs||[]).length && d.parent===null) L.innerHTML='<div class="k" style="padding:10px">(no subfolders)</div>';
  $('bcount').textContent = d.images>0 ? `🖼 ${d.images} images in this folder` : 'no images directly in this folder';
  $('bselect').disabled = d.images<1;
}
function selectBrowseFolder(){ $('suImages').value=BROWSE_PATH; closeBrowser(); }
async function startSetup(){
  const images=$('suImages').value.trim();
  if(!images){ $('suErr').textContent='Enter the path to an images folder.'; return; }
  $('suErr').textContent='setting up…';
  const preset=(SEL_PRESET>=0&&PRESETS[SEL_PRESET]&&PRESETS[SEL_PRESET].images===images)?PRESETS[SEL_PRESET]:null;
  const body={images, classes:$('suClasses').value.trim(), labels:$('suLabels').value.trim()};
  if(preset){ body.seed_labels=SEED_LABELS; body.seed_percent=SEED_PCT; }   // preset → auto-seed from GT
  const r=await postJSON('/api/setup',body);
  if(r.error){ $('suErr').textContent=r.error; return; }
  $('setup').classList.remove('open');
  toast(`Loaded ${r.images} images · classes: ${(r.classes||[]).join(', ')}`,'ok');
  init();                                              // reload config → enter the labeling UI
}
function setAIEnabled(on){['autoBtn','autoAllBtn'].forEach(id=>$(id).disabled=!on);   // method stays usable (cycle picks it before training)
  $('autoBtn').title=on?'':(CAN_TRAIN?'Train a model first (🎓 above)':'no AI backend (manual mode)');}
/* Per-dataset adaptive default (containment on/off + count cap), derived from the few-label seed.
   Always applied and NOT shown in the UI — the only thing the page needs from it is the containment
   mirror, so the manual-slider preview suppresses (or keeps) nested boxes exactly like the server. */
function renderAdaptive(a){
  if(!a || a.nest_frac===null || a.nest_frac===undefined) return;   // no seed yet → keep the default
  ADAPT_CONT = a.containment!==false;
}
async function loadMethods(){const m=await getJSON('/api/methods');
  const groups={};(m.methods||[]).forEach(x=>{const g=x.group||'Methods';(groups[g]=groups[g]||[]).push(x);});
  $('method').innerHTML=Object.entries(groups).map(([g,items])=>
    `<optgroup label="${g}">`+items.map(x=>`<option value="${x.id}">${x.label}</option>`).join('')+'</optgroup>').join('');
  setAIEnabled(HAS_AI);onMethodChange();}   // enable Inference only with a model

/* ---------- classes ---------- */
const hasClass=(nm,c)=>(IMG_CLASSES[nm]||[]).includes(c)||(AI_CLASSES[nm]||[]).includes(c);  // saved OR AI-inferred
const matchFilter=nm=>FILTER_CLASS===null||hasClass(nm,FILTER_CLASS);
function classImgCount(c){return IMAGES.reduce((s,nm)=>s+(hasClass(nm,c)?1:0),0);}
function toggleClassFilter(i){FILTER_CLASS=(FILTER_CLASS===i)?null:i;curClass=i;renderPalette();renderImageList();}
function renderPalette(){$('palette').innerHTML='';
  CLASSES.forEach((n,i)=>{const d=document.createElement('div');
    d.className='chip'+(i===curClass?' active':'')+(i===FILTER_CLASS?' filtering':'');
    const cnt=classImgCount(i);
    d.title=`class ${i+1} · ${cnt} image(s) · click to ${i===FILTER_CLASS?'clear filter':'show only these'}`;
    d.innerHTML=`<span class="dot" style="background:${PALETTE[i%PALETTE.length]}"></span>${n}`
      +(cnt?` <span class="n">${cnt}</span>`:'');
    d.onclick=()=>toggleClassFilter(i);
    $('palette').appendChild(d);});
  if(CLASSES.length<9){const a=document.createElement('div');a.className='chip addcls';
    a.title='Add a class — picked up by the next Train / Run cycle';a.innerHTML='<b>＋</b> class';
    a.onclick=addClass;$('palette').appendChild(a);}}
async function addClass(){const name=(prompt('New class name:')||'').trim();
  if(!name)return;
  const r=await postJSON('/api/classes',{name});
  if(r.detail){toast(r.detail,'warn');return;}
  CLASSES=r.classes;curClass=CLASSES.length-1;renderPalette();
  toast(`Class <b>${name}</b> added (${CLASSES.length} total) — label with it, then Train to pick it up.`,'ok');}

/* ---------- image list ---------- */
function renderImageList(){const L=$('imglist');L.innerHTML='';
  if(FILTER_CLASS!==null){const f=document.createElement('div');f.className='imfilter';
    f.innerHTML=`<span class="dot" style="width:9px;height:9px;border-radius:50%;background:${PALETTE[FILTER_CLASS%PALETTE.length]}"></span>`
      +`🔍 ${CLASSES[FILTER_CLASS]||FILTER_CLASS} · ${classImgCount(FILTER_CLASS)} <span class="x" title="show all">✕</span>`;
    f.querySelector('.x').onclick=()=>{FILTER_CLASS=null;renderPalette();renderImageList();};
    L.appendChild(f);}
  let shown=0,curEl=null;
  IMAGES.forEach((nm,i)=>{if(!matchFilter(nm))return;shown++;
    const n=COUNTS[nm]||0;const own=OWNER[nm]||'none';const d=document.createElement('div');
    /* human but 0 boxes = you marked it reviewed/empty — a hollow dot, so "green with no boxes"
       is never mistaken for a seeded image that carries ground-truth boxes. */
    d.className='imrow'+(i===idx?' cur':'')+(own==='human'?(n>0?' done':' done nobox'):own==='auto'?' auto':'');
    d.innerHTML=`<span class="st" title="${own==='human'?(n>0?'Your labels — '+n+' box(es)':'Marked reviewed by you — no boxes'):own==='auto'?'AI-labeled (can be re-applied)':'unlabeled'}"></span>`
      +`<span class="nm">${nm}</span>`+(n>0?`<span class="cnt">${n}</span>`:'');
    d.onclick=()=>{if(i!==idx) save(true).then(()=>load(i));};L.appendChild(d);if(i===idx)curEl=d;});
  if(FILTER_CLASS!==null&&!shown){const e=document.createElement('div');e.className='empty';
    e.textContent='No images with this class yet.';L.appendChild(e);}
  if(curEl)curEl.scrollIntoView({block:'nearest'});}

/* ---------- progress ---------- */
async function refreshStatus(){try{const s=await getJSON('/api/status');COUNTS=s.counts||{};OWNER=s.owner||{};IMG_CLASSES=s.classes||{};AI_CLASSES=s.ai_classes||{};
  $('prog').textContent=`${s.labeled} / ${s.total}`;$('progbar').style.width=(s.total?100*s.labeled/s.total:0)+'%';
  const auto=Object.values(OWNER).filter(o=>o==='auto').length;
  $('imcount').innerHTML=`${s.total} · <span style="color:var(--ok)">Human (${s.human||0})</span> · <span style="color:var(--band-amber)">AI (${auto})</span>`;
  renderAdaptive(s.adaptive);   // the adaptive decision moves as the seed grows — re-read it, don't cache it from startup
  renderPalette();renderImageList();}catch(e){}}

/* ---------- image load + canvas ---------- */
async function load(i){if(i<0||i>=IMAGES.length)return;hideClassPicker();idx=i;sel=-1;humanDirty=false;
  const name=IMAGES[idx];const d=await getJSON('/api/labels/'+encodeURIComponent(name));
  imgW=d.width;imgH=d.height;boxes=d.boxes.map(b=>({...b,ai:false}));
  img=new Image();img.onload=()=>{fit();draw();};img.src='/api/file/'+encodeURIComponent(name);
  $('counter').textContent=(idx+1)+' / '+IMAGES.length;renderImageList();
  if($('method')&&$('method').value==='manual'){loadImgCands().then(onThr);}   // refresh slider preview
}
function fit(){const maxW=$('stage').clientWidth-48,maxH=$('stage').clientHeight-48;
  scale=Math.min(maxW/imgW,maxH/imgH,2);cv.width=Math.round(imgW*scale);cv.height=Math.round(imgH*scale);}
const nx=b=>[(b.cx-b.w/2)*cv.width,(b.cy-b.h/2)*cv.height,b.w*cv.width,b.h*cv.height];
const corners=(x,y,w,h)=>[[x,y],[x+w,y],[x,y+h],[x+w,y+h]];
function clampBox(b){let x1=b.cx-b.w/2,y1=b.cy-b.h/2,x2=b.cx+b.w/2,y2=b.cy+b.h/2;
  x1=Math.min(Math.max(x1,0),1);y1=Math.min(Math.max(y1,0),1);x2=Math.min(Math.max(x2,0),1);y2=Math.min(Math.max(y2,0),1);
  b.cx=(x1+x2)/2;b.cy=(y1+y2)/2;b.w=x2-x1;b.h=y2-y1;}
function iou(a,b){const ix=Math.max(0,Math.min(a.cx+a.w/2,b.cx+b.w/2)-Math.max(a.cx-a.w/2,b.cx-b.w/2));
  const iy=Math.max(0,Math.min(a.cy+a.h/2,b.cy+b.h/2)-Math.max(a.cy-a.h/2,b.cy-b.h/2));
  const I=ix*iy,U=a.w*a.h+b.w*b.h-I;return U>0?I/U:0;}
const areaN=b=>b.w*b.h;
function containFrac(inner,outer){const ix=Math.max(0,Math.min(inner.cx+inner.w/2,outer.cx+outer.w/2)-Math.max(inner.cx-inner.w/2,outer.cx-outer.w/2));
  const iy=Math.max(0,Math.min(inner.cy+inner.h/2,outer.cy+outer.h/2)-Math.max(inner.cy-inner.h/2,outer.cy-outer.h/2));
  const a=inner.w*inner.h;return a>0?(ix*iy)/a:0;}
/* mirror the server (_dedup runs on AI candidates only): drop an AI box ≥80% contained inside a
   strictly-larger *AI* box — a human box never suppresses an AI proposal here.
   ADAPT_CONT mirrors the server's adaptive containment decision: on a dataset whose seed shows
   genuine nesting the server keeps inner objects, so the slider preview must keep them too. */
const dropContained=list=>ADAPT_CONT
  ?list.filter((b,i)=>!(b.ai&&list.some((o,j)=>j!==i&&o.ai&&areaN(o)>areaN(b)&&containFrac(b,o)>=0.80)))
  :list;
function draw(){ctx.clearRect(0,0,cv.width,cv.height);if(img.complete)ctx.drawImage(img,0,0,cv.width,cv.height);
  boxes.forEach((b,i)=>{const[x,y,w,h]=nx(b);
    const col=b.ai?(BAND[b.band]||'#888'):HUMAN_COL;
    ctx.setLineDash(b.ai?[6,4]:[]);
    /* human / seed boxes are BLACK. A black line disappears on dark pixels, so paint a white halo
       under it first — this is what makes the pre-loaded ground-truth seed verifiable on any image. */
    if(!b.ai){ctx.lineWidth=(i===sel?4:3)+3;ctx.strokeStyle='rgba(255,255,255,.9)';
      ctx.strokeRect(x,y,w,h);}
    ctx.lineWidth=b.ai?(i===sel?3:2):(i===sel?4:3);ctx.strokeStyle=col;
    ctx.strokeRect(x,y,w,h);ctx.setLineDash([]);
    const tag=(CLASSES[b.cls]||b.cls)+(b.ai?` ${b.score??''}▲`:'');
    ctx.font='600 12px ui-sans-serif,system-ui';const tw=ctx.measureText(tag).width;
    ctx.fillStyle=col;ctx.globalAlpha=.9;ctx.fillRect(x,(y>16?y-16:y),tw+10,15);ctx.globalAlpha=1;
    ctx.fillStyle=b.ai?'#0b0d13':'#ffffff';ctx.fillText(tag,x+5,(y>16?y-4:y+12));
    if(i===sel){ctx.fillStyle='#fff';for(const[hx,hy]of corners(x,y,w,h))ctx.fillRect(hx-4,hy-4,8,8);}});
  renderBoxList();}
function renderBoxList(){const L=$('boxlist');L.innerHTML='';
  if(!boxes.length){L.innerHTML='<div class="empty">No boxes yet — draw one, or run Inference.</div>';}
  boxes.forEach((b,i)=>{const d=document.createElement('div');d.className='b'+(i===sel?' sel':'');
    const col=b.ai?(BAND[b.band]||'#888'):HUMAN_COL;
    d.innerHTML=`<span class="sw" style="background:${col}"></span>`
      +`<span class="nm clsedit" title="Click to change the class">${CLASSES[b.cls]||b.cls}</span>`
      +(b.ai?`<span class="badge ai">AI ${b.score??''}</span>`:'')+`<span class="x" title="delete">✕</span>`;
    d.onclick=e=>{if(e.target.classList.contains('x')){boxes.splice(i,1);sel=-1;hideClassPicker();mark();draw();return;}
      sel=i;draw();if(e.target.classList.contains('clsedit'))showClassPicker(boxes[i]);};
    L.appendChild(d);});
  $('boxcount').textContent=boxes.length+(boxes.some(b=>b.ai)?` · ${boxes.filter(b=>b.ai).length} AI`:'');}

/* ---------- mouse ---------- */
function mp(e){const r=cv.getBoundingClientRect();return[e.clientX-r.left,e.clientY-r.top];}
cv.addEventListener('mousedown',e=>{hideClassPicker();const[mx,my]=mp(e);
  if(sel>=0){const[x,y,w,h]=nx(boxes[sel]);const cs=corners(x,y,w,h);
    for(let k=0;k<4;k++)if(Math.abs(mx-cs[k][0])<6&&Math.abs(my-cs[k][1])<6){drag={mode:'resize',k,anchor:cs[3-k]};return;}}
  for(let i=boxes.length-1;i>=0;i--){const[x,y,w,h]=nx(boxes[i]);
    if(mx>=x&&mx<=x+w&&my>=y&&my<=y+h){sel=i;drag={mode:'move',ox:mx,oy:my,b:{...boxes[i]}};draw();return;}}
  const nb={cls:curClass,cx:mx/cv.width,cy:my/cv.height,w:0,h:0,ai:false};
  boxes.push(nb);sel=boxes.length-1;drag={mode:'draw',x0:mx,y0:my};});
cv.addEventListener('mousemove',e=>{if(!drag)return;const[mx,my]=mp(e);const b=boxes[sel];drag.moved=true;
  if(drag.mode==='draw'){const x=Math.min(drag.x0,mx),y=Math.min(drag.y0,my),w=Math.abs(mx-drag.x0),h=Math.abs(my-drag.y0);
    b.cx=(x+w/2)/cv.width;b.cy=(y+h/2)/cv.height;b.w=w/cv.width;b.h=h/cv.height;}
  else if(drag.mode==='move'){b.cx=drag.b.cx+(mx-drag.ox)/cv.width;b.cy=drag.b.cy+(my-drag.oy)/cv.height;b.ai=false;}
  else if(drag.mode==='resize'){const opp=drag.anchor;
    const x1=Math.min(mx,opp[0]),y1=Math.min(my,opp[1]),x2=Math.max(mx,opp[0]),y2=Math.max(my,opp[1]);
    b.cx=(x1+x2)/2/cv.width;b.cy=(y1+y2)/2/cv.height;b.w=(x2-x1)/cv.width;b.h=(y2-y1)/cv.height;b.ai=false;}
  draw();});
window.addEventListener('mouseup',()=>{if(drag){const b=boxes[sel];
  const plainClick=(drag.mode==='move'&&!drag.moved);   // clicked a box without moving → edit its class
  if(drag.mode==='draw'&&b&&(b.w<0.004||b.h<0.004)){boxes.splice(sel,1);sel=-1;}
  else if(b){clampBox(b);if(drag.mode==='draw'||drag.moved)mark();}  // real geometry change (not a plain select)
  drag=null;draw();
  if(plainClick&&sel>=0)showClassPicker(boxes[sel]);}});

/* ---------- click-a-box class picker (change a box's class from the declared classes) ---------- */
function showClassPicker(b){const cp=$('classPicker');if(!cp||!b||!CLASSES.length)return;
  cp.innerHTML='<div class="cphd">Select class</div>'+CLASSES.map((n,ci)=>
    `<div class="cpchip${ci===b.cls?' sel':''}" onclick="pickBoxClass(${ci})">`
    +`<span class="dot" style="background:${PALETTE[ci%PALETTE.length]}"></span>${n}<span class="k">${ci+1}</span></div>`).join('');
  const r=cv.getBoundingClientRect(),xywh=nx(b);
  cp.style.left=Math.round(r.left+xywh[0])+'px';cp.style.top=Math.round(r.top+xywh[1])+'px';cp.classList.add('open');
  requestAnimationFrame(()=>{const pr=cp.getBoundingClientRect();
    if(pr.right>innerWidth-8)cp.style.left=Math.max(8,innerWidth-8-pr.width)+'px';
    if(pr.bottom>innerHeight-8)cp.style.top=Math.max(8,innerHeight-8-pr.height)+'px';});}
function pickBoxClass(ci){if(sel>=0&&boxes[sel]){boxes[sel].cls=ci;boxes[sel].ai=false;mark();renderPalette();draw();
    toast(`Class → <b>${CLASSES[ci]||ci}</b>`,'ok');}hideClassPicker();}
function hideClassPicker(){const cp=$('classPicker');if(cp)cp.classList.remove('open');}
document.addEventListener('mousedown',e=>{const cp=$('classPicker');
  if(cp&&cp.classList.contains('open')&&!cp.contains(e.target))hideClassPicker();});

/* ---------- keyboard ---------- */
window.addEventListener('keydown',e=>{const t=e.target;
  if(t&&(t.tagName==='SELECT'||t.tagName==='INPUT'||t.tagName==='TEXTAREA'||t.isContentEditable))return;
  /* A modal owns the keyboard while it is open — the rule review now opens on EVERY training run,
     and Enter/A/S/Del behind it would edit, re-infer or promote the image under the dialog. */
  if(document.querySelector('#gate.open,#setup.open,#browser.open,#overlay.open')){
    if(e.key==='Escape'&&$('overlay').classList.contains('open'))toggleHelp();
    return;}
  if(e.key==='a'||e.key==='A'){e.preventDefault();automate();}
  else if(e.key==='s'||e.key==='S'){e.preventDefault();save();}
  else if(e.key==='n'||e.key==='N'||e.key==='ArrowRight'){e.preventDefault();next();}
  else if(e.key==='p'||e.key==='P'||e.key==='ArrowLeft'){e.preventDefault();prev();}
  else if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();delSel();}
  else if(e.key==='Enter'){e.preventDefault();confirmHuman();}   // ✓ Check → promote to Human
  else if(e.key==='Escape'){sel=-1;hideClassPicker();draw();}
  else if(e.key>='1'&&e.key<='9'){const c=+e.key-1;if(c<CLASSES.length){curClass=c;
    if(sel>=0){boxes[sel].cls=c;boxes[sel].ai=false;mark();renderBoxList();}renderPalette();draw();}}});

/* ---------- actions ---------- */
function delSel(){if(sel>=0){boxes.splice(sel,1);sel=-1;hideClassPicker();mark();draw();}}
function clearAll(){if(boxes.length&&!confirm('Remove all boxes on this image?'))return;if(boxes.length)mark();boxes=[];sel=-1;draw();}
function next(){move(1);}
function prev(){move(-1);}
function move(dir){let i=idx;                                   // when a class filter is on, skip non-matching
  do{i+=dir;}while(i>=0&&i<IMAGES.length&&FILTER_CLASS!==null&&!matchFilter(IMAGES[i]));
  if(i<0||i>=IMAGES.length)return;save(true).then(()=>load(i));}
async function save(silent){const clean=boxes.map(b=>({cls:b.cls,cx:b.cx,cy:b.cy,w:b.w,h:b.h}));
  const human=(!silent)||humanDirty;   // explicit Save or a real edit ⇒ human-owned (kept on re-apply)
  await postJSON('/api/labels/'+encodeURIComponent(IMAGES[idx]),{boxes:clean,human});
  humanDirty=false;
  if(!silent)toast(`Saved <b>${IMAGES[idx]}</b> · ${clean.length} boxes`,'ok');refreshStatus();}
/* ✓ Check: approve the current image's (AI) work → promote to Human-owned (green) + save. Editing an
   AI image already promotes it on save (humanDirty); this is the no-edit "reviewed & approved" action. */
async function confirmHuman(){if(!IMAGES.length)return;
  const clean=boxes.map(b=>({cls:b.cls,cx:b.cx,cy:b.cy,w:b.w,h:b.h}));
  await postJSON('/api/labels/'+encodeURIComponent(IMAGES[idx]),{boxes:clean,human:true});
  humanDirty=false;
  toast(`✓ <b>${IMAGES[idx]}</b> — marked as reviewed by you (green) · ${clean.length} boxes`,'ok');
  await refreshStatus();await refreshSummary();}

/* ---------- manual confidence threshold (slider) ---------- */
let SCORE='p_good',THR=0.35,IMG_CANDS=null,SUMMARY=null;
const scoreOf=c=>SCORE==='p_good'?c.p_good:c.det_conf;
function bandOf(s){return s>=0.65?'green':(s>=0.35?'amber':'red');}
function onMethodChange(){const v=$('method').value, manual=v==='manual';
  $('manualPanel').style.display=manual?'block':'none';
  $('autoBtn').style.display=manual?'none':'';   // slider drives the current image in manual mode
  if(manual)ensureManual();
  else{IMG_CANDS=null;boxes=boxes.filter(b=>!b.ai);sel=-1;draw();}}  // clear stale slider previews
async function ensureManual(){setScoreBtns();
  try{SUMMARY=await getJSON('/api/score_summary');}catch(e){SUMMARY=null;}
  await loadImgCands();onThr();}
/* The histogram and the "All ≈ N objects" estimate are a snapshot of the CURRENT model over the
   CURRENT auto-target set. Both move when the model is retrained and when an image is promoted into
   the human seed, so re-pull them at those points — leaving the previous model's score
   distribution on screen is the one place the UI can silently show stale inference. */
async function refreshSummary(){if(HAS_AI&&$('method').value==='manual')await ensureManual();}
async function loadImgCands(){if(!HAS_AI||$('method').value!=='manual'){IMG_CANDS=null;return;}
  try{IMG_CANDS=(await getJSON('/api/candidates/'+encodeURIComponent(IMAGES[idx]))).candidates||[];}catch(e){IMG_CANDS=[];}}
function setScore(s){SCORE=s;setScoreBtns();onThr();}
function setScoreBtns(){$('scPgood').classList.toggle('active',SCORE==='p_good');$('scConf').classList.toggle('active',SCORE==='det_conf');}
function histAtLeast(hist,thr,bins){if(!hist)return '–';let s=0;   // start = bin CONTAINING thr (server floors), clamp top edge
  for(let i=Math.min(bins-1,Math.max(0,Math.round(thr*bins)));i<hist.length;i++)s+=hist[i];return s;}
function onThr(){THR=parseFloat($('thrSlider').value);$('thrVal').textContent=THR.toFixed(2);
  const hist=SUMMARY?(SCORE==='p_good'?SUMMARY.p_good_hist:SUMMARY.det_conf_hist):null;
  const all=SUMMARY?histAtLeast(hist,THR,SUMMARY.bins):'–';
  const sc=SCORE==='p_good'?'P(good)':'conf';
  if(OWNER[IMAGES[idx]]==='human'){                                  // Auto-label ALL skips your labels
    boxes=boxes.filter(b=>!b.ai);draw();
    $('thrCount').innerHTML=`This image: <b>kept</b> (your label, not overwritten) · All ≈ <b>${all}</b> at ${sc} ≥ ${THR.toFixed(2)}`;
  }else if(IMG_CANDS){
    boxes=boxes.filter(b=>!b.ai);
    if(OWNER[IMAGES[idx]]==='auto'&&!humanDirty)boxes=[];
    IMG_CANDS.filter(c=>scoreOf(c)>=THR).forEach(c=>{
      if(!boxes.some(eb=>iou(c,eb)>0.55)){boxes.push({cls:c.cls||0,cx:c.cx,cy:c.cy,w:c.w,h:c.h,score:+scoreOf(c).toFixed(3),band:bandOf(scoreOf(c)),ai:true});}});
    boxes=dropContained(boxes);const n=boxes.filter(b=>b.ai).length;   // suppress nested boxes (matches server)
    draw();
    $('thrCount').innerHTML=`This image: <b>${n}</b> · All ≈ <b>${all}</b> objects  (${sc} ≥ ${THR.toFixed(2)})`;
  }
  drawHist();}
function drawHist(){const cv2=$('scoreHist');if(!cv2)return;const x2=cv2.getContext('2d');
  const W=cv2.width,H=cv2.height;x2.clearRect(0,0,W,H);
  const hist=SUMMARY?(SCORE==='p_good'?SUMMARY.p_good_hist:SUMMARY.det_conf_hist):null;
  if(!hist||!hist.length)return;
  const bins=hist.length,mx=Math.max(1,...hist),pad=3,bw=(W-pad*2)/bins,cs=getComputedStyle(document.body);
  const acc=(cs.getPropertyValue('--ok')||'#22c55e').trim(),rej=(cs.getPropertyValue('--line2')||'#556').trim(),cur=(cs.getPropertyValue('--acc')||'#6366f1').trim();
  for(let i=0;i<bins;i++){const x=pad+i*bw,hgt=(hist[i]/mx)*(H-6);
    x2.fillStyle=(i/bins)>=THR?acc:rej;x2.fillRect(x,H-hgt,Math.max(1,bw-0.5),hgt);}
  const cx=pad+THR*(W-pad*2);x2.strokeStyle=cur;x2.lineWidth=1.5;x2.beginPath();x2.moveTo(cx,0.5);x2.lineTo(cx,H);x2.stroke();}

async function automate(){if(!HAS_AI)return;
  if($('method').value==='manual'){if(!IMG_CANDS)await loadImgCands();onThr();
    toast('Adjust the slider to set the confidence threshold.');return;}
  const r=await postJSON('/api/automate/'+encodeURIComponent(IMAGES[idx]),{method:$('method').value});
  if(r.detail){toast('AI: '+r.detail,'warn');return;}
  boxes=boxes.filter(b=>!b.ai);sel=-1;                                 // drop previous AI proposals
  if(OWNER[IMAGES[idx]]==='auto'&&!humanDirty)boxes=[];                // pure AI image → replace, don't stack
  let added=0;
  (r.boxes||[]).forEach(nb=>{ if(!boxes.some(eb=>iou(nb,eb)>0.55)){boxes.push(nb);added++;} }); // skip duplicates
  if(r.fell_back)toast('No human seed yet — fell back to Otsu. Label a few images for count-guided.','warn');
  toast(`⚡ ${added} boxes · ${r.op}`);draw();}

async function automateAll(){if(!HAS_AI)return;
  const manual=$('method').value==='manual';
  const meth=manual?`Manual: ${SCORE==='p_good'?'P(good)':'conf'} ≥ ${THR.toFixed(2)}`
                   :$('method').options[$('method').selectedIndex].text;
  if(!confirm(`Run inference on every AI / unlabeled image and save?\n\n${meth}\n\nYour own labels are kept.`))return;
  toast('Running inference…');
  const body=manual?{method:'manual',thr:THR,score:SCORE}:{method:$('method').value};
  const r=await postJSON('/api/automate_all',body);
  if(r.detail){toast('AI: '+r.detail,'warn');return;}
  if(r.fell_back)toast('No human seed yet — fell back to Otsu. Label a few images for count-guided.','warn');
  toast(`✅ ${r.auto_labeled} images / ${r.total_boxes} boxes · ${r.op}`,'ok');
  refreshStatus();
  await refreshSummary();                                  // target set changed → re-pull, don't reuse
  load(idx);}

/* ---------- training ---------- */
// while a job runs, its button becomes a RED ⏹ Stop (click to abort); the other button is disabled.
function setJobButtons(running,which){const tb=$('trainBtn'),cb=$('cycleBtn');if(!tb)return;
  if(running){const run=(which==='cycle')?cb:tb,other=(which==='cycle')?tb:cb;
    run.classList.remove('train');run.classList.add('danger');run.disabled=false;run.innerHTML='⏹ Stop';run.onclick=stopTraining;
    if(other){other.disabled=true;}
  }else{
    tb.classList.add('train');tb.classList.remove('danger');tb.disabled=false;tb.innerHTML='🎓 Train the model';tb.onclick=trainModel;
    if(cb){cb.classList.add('train');cb.classList.remove('danger');cb.disabled=false;cb.innerHTML='🔁 Run cycle (train → auto-label → repeat)';cb.onclick=runCycle;}}}
async function stopTraining(){if(!confirm('Stop the running job?'))return;
  const r=await postJSON('/api/train/stop',{});
  if(r.detail){toast(r.detail,'warn');return;}
  toast('⏹ Stop requested — the job will halt after the current step.','warn');}
async function trainModel(){if(!CAN_TRAIN)return;
  const st=await getJSON('/api/status');
  const human=st.human||0, auto=Object.values(OWNER).filter(o=>o==='auto').length;
  const scope=$('trainScope').value;
  const nTrain = scope==='all' ? human+auto : human;
  if(nTrain<1){toast('Label a few images first (the few-label seed), then train.','warn');return;}
  const scopeTxt = scope==='all'
    ? `🟢 ${human} of your labels + 🟡 ${auto} AI-labeled`
    : `🟢 ${human} of your labels`;
  if(!confirm(`Train on ${scopeTxt}, on the GPU?\n\nTraining pauses once for the rule review.`))return;
  setJobButtons(true,'train');$('tiles').style.display='none';$('minirows').innerHTML='';
  $('trainlog').style.display='block';$('trainlog').textContent='starting…';
  const r=await postJSON('/api/train',{scope,det_mode:$('detMode').value});
  if(r.detail){toast('Cannot start: '+r.detail,'err');setJobButtons(false);return;}
  $('trainstat').innerHTML=`<span class="spin">⏳</span> training on ${nTrain} image(s) (${scope==='all'?'green + AI':'green'})…`;
  if(trainPoll)clearInterval(trainPoll);trainPoll=setInterval(pollTrain,1500);}

async function pollTrain(){let s;try{s=await getJSON('/api/train/status');}catch(e){return;}
  maybeGate(s);                                                 // open/close the rule-review window
  $('trainlog').textContent=(s.log||[]).join('\n');$('trainlog').scrollTop=1e9;
  if(s.report)renderReport(s.report);
  if(s.state==='running'){$('trainstat').innerHTML=`<span class="spin">⏳</span> training… ${s.elapsed}s`;return;}
  clearInterval(trainPoll);trainPoll=null;setJobButtons(false);
  if(s.state==='done'){HAS_AI=true;await loadMethods();
    $('trainstat').innerHTML=`✅ done in ${s.report?.seconds||'?'}s`;
    toast('🎓 Training done — the trained model is now active.','ok');
    load(idx);refreshStatus();refreshSummary();}
  else if(s.state==='stopped'){$('trainstat').innerHTML='⏹ Training was stopped.';toast('⏹ Training stopped.','warn');refreshStatus();}
  else{$('trainstat').innerHTML='❌ training failed — see log.';toast('Training failed — see the log panel.','err');}}

/* ---------- self-training cycle ---------- */
let cyclePoll=null;
async function runCycle(){if(!CAN_TRAIN)return;
  const st=await getJSON('/api/status');
  if(st.labeled<1){toast('Label a few images first, then run the cycle.','warn');return;}
  const iters=Math.max(1,Math.min(10,parseInt($('cycIters').value)||3));
  const include_ai=$('cycIncl').checked, manual=$('method').value==='manual', method=$('method').value;
  const opt=$('method').options[$('method').selectedIndex];
  const lbl=manual?`manual ${SCORE==='p_good'?'P(good)':'conf'} ≥ ${THR.toFixed(2)}`:(opt?opt.text:method);
  if(!confirm(`Run ${iters} self-training rounds on the GPU?\n\nEach round: 🎓 train → ⚡⚡ inference on all remaining (${lbl}).\n`
    +(include_ai?`Rounds 2+ fold the accepted AI labels back into training.`
                :`Every round trains on your labels only.`)))return;
  setJobButtons(true,'cycle');$('cycRows').innerHTML='';$('tiles').style.display='none';$('minirows').innerHTML='';
  $('trainlog').style.display='block';$('trainlog').textContent='starting cycle…';
  const body={iterations:iters,method,include_ai,det_mode:$('detMode').value};   // fold-gate: server default
  if(manual){body.thr=THR;body.score=SCORE;}
  const r=await postJSON('/api/cycle',body);
  if(r.detail){toast('Cannot start: '+r.detail,'err');setJobButtons(false);return;}
  $('trainstat').innerHTML='<span class="spin">🔁</span> self-training cycle…';
  if(cyclePoll)clearInterval(cyclePoll);cyclePoll=setInterval(pollCycle,1500);}
async function pollCycle(){let s;try{s=await getJSON('/api/cycle/status');}catch(e){return;}
  maybeGate(s);                                                 // round-1 rule-review window
  $('trainlog').textContent=(s.log||[]).join('\n');$('trainlog').scrollTop=1e9;
  const pct=v=>(v==null?'–':(100*v).toFixed(1)+'%');
  $('cycRows').innerHTML=(s.iters||[]).map(r=>
    `<div><span>round ${r.round} <span style="color:var(--mut2)">(${r.scope})</span></span><span class="v">mAP ${pct(r.map50)} · ${r.auto_labeled} imgs / ${r.total_boxes} box</span></div>`).join('');
  if(s.state==='running'){$('trainstat').innerHTML=`<span class="spin">🔁</span> round ${s.current}/${s.total} … ${s.elapsed}s`;return;}
  clearInterval(cyclePoll);cyclePoll=null;setJobButtons(false);
  if(s.state==='done'){HAS_AI=true;await loadMethods();
    $('trainstat').innerHTML=`✅ cycle complete (${s.total} rounds)`;
    toast('🔁 Cycle complete — model and labels updated.','ok');
    refreshStatus();load(idx);refreshSummary();}
  else if(s.state==='stopped'){HAS_AI=true;await loadMethods();$('trainstat').innerHTML='⏹ The cycle was stopped.';
    toast('⏹ Cycle stopped.','warn');refreshStatus();load(idx);refreshSummary();}
  else{$('trainstat').innerHTML='❌ cycle failed — see log.';toast('Cycle failed — see the log panel.','err');}}

function renderReport(rep){const pct=v=>(v==null?'–':(100*v).toFixed(1)+'%');
  const d=rep.detector||{},v=rep.validator||{},p=rep.prediction||{},sp=rep.species;
  $('tiles').style.display='grid';
  $('tiles').innerHTML=`
    <div class="tile"><b class="g">${pct(d.map50)}</b><span>det mAP@50</span></div>
    <div class="tile"><b class="i">${sp?pct(sp.best_val_acc):pct(v.best_val_acc)}</b><span>${sp?'classifier acc':'validator acc'}</span></div>
    <div class="tile"><b>${p.candidates??'–'}</b><span>candidates</span></div>`;
  $('minirows').innerHTML=`
    <div><span>detection mode</span><span class="v">${rep.det_mode||'multi'}</span></div>
    <div><span>detector P / R</span><span class="v">${pct(d.precision)} / ${pct(d.recall)}</span></div>
    ${sp?`<div><span>🧬 classifier acc · crops</span><span class="v">${pct(sp.best_val_acc)} · ${sp.n_crops}</span></div>`:''}
    <div><span>validator acc</span><span class="v">${pct(v.best_val_acc)}</span></div>
    <div><span>train / val imgs</span><span class="v">${d.n_train??'–'} / ${d.n_val??'–'}</span></div>
    <div><span>mean P(good) · conf</span><span class="v">${p.mean_p_good??'–'} · ${p.collection_conf??'–'}</span></div>
    ${rep.error?`<div><span>error</span><span class="v" style="color:var(--warn)">${rep.error}</span></div>`:''}`;}

/* ---------- rule-review gate (crop confirmation before the filter trains) ---------- */
let GATE_ON=false,GATE_RULE='baseline';
function setGateRule(r){GATE_RULE=r;$('grBase').classList.toggle('active',r==='baseline');$('grRef').classList.toggle('active',r==='refined');
  $('grIouRow').style.display=r==='baseline'?'':'none';$('grRejRow').style.display=r==='refined'?'':'none';}
function onGateSlide(){['Good','Empty','Dev','Jit','Iou','Rej','Shift'].forEach(k=>{const el=$('gr'+k);if(el)$('gr'+k+'V').textContent=(+el.value).toFixed(2);});}
function fillGateControls(c){c=c||{};
  $('grGood').value=c.good_crop_ratio??0.4;$('grEmpty').value=c.empty_crop_ratio??0.3;$('grDev').value=c.deviation_ratio??0.3;
  $('grJit').value=c.good_crop_jitter??0.05;$('grIou').value=c.deviation_max_iou??0.10;$('grRej').value=c.negative_iou_reject??0.75;
  $('grShift').value=c.deviation_shift??0.8;
  $('grNdev').value=c.deviation_per_image??10;$('grNdevV').textContent=$('grNdev').value;
  setGateRule(c.negative_rule==='refined'?'refined':'baseline');onGateSlide();}
function gateConfig(){return{negative_rule:GATE_RULE,
  good_crop_ratio:+$('grGood').value,empty_crop_ratio:+$('grEmpty').value,deviation_ratio:+$('grDev').value,
  good_crop_jitter:+$('grJit').value,deviation_max_iou:+$('grIou').value,negative_iou_reject:+$('grRej').value,
  deviation_shift:+$('grShift').value,deviation_per_image:+$('grNdev').value};}
let GATE_V=0;   // cache-buster: regenerated crops reuse filenames, so bust the browser cache per render
function renderGateCrops(m){if(!m)return;GATE_V++;const c=m.counts||{},cf=m.config||{};
  $('gateCounts').innerHTML=`good <b>${c.good||0}</b> · empty <b>${c.empty||0}</b> · deviated <b>${c.deviated||0}</b><br><b>${c.total||0}</b> crops total · rule <b>${cf.negative_rule||''}</b>`;
  const s=m.samples||{};
  /* each group header carries the crop-size control at its RIGHT END — the headers are sticky, so the
     control is always in reach no matter which group you have scrolled to. All copies stay in sync. */
  const zoomCtl=()=>`<span class="gz" title="Size of the crop samples">`
    +`<button class="zb" onclick="stepCropZoom(-16)">－</button>`
    +`<input type="range" class="cropz" min="48" max="240" step="4" value="${CROP_PX}" oninput="setCropZoom(this.value)">`
    +`<button class="zb" onclick="stepCropZoom(16)">＋</button>`
    +`<span class="zv cropv">${CROP_PX}px</span></span>`;
  const col=(title,type,files,cls)=>{const imgs=(files||[]).map(f=>`<img loading="lazy" src="/api/train/crop_img?type=${type}&f=${encodeURIComponent(f)}&v=${GATE_V}">`).join('');
    return `<div class="gcol ${cls}"><div class="ghd">${title} <span style="color:var(--mut2);font-weight:400">${(files||[]).length}</span>${zoomCtl()}</div><div class="grid">${imgs||'<span class="empty">none</span>'}</div></div>`;};
  $('gateGrid').innerHTML=col('🟢 good — your labels','good',s.good,'good')
    +col('🔴 empty — background (negative)','empty',s.empty,'noise')
    +col('🔴 deviated — displaced (negative)','deviated',s.deviated,'noise');}

/* deviation overlay: draw GT (green) + deviated (colored) boxes on ONE original image */
let DEV_IMG=null, DEV_DATA=null; const devImgEl=new Image();
function humanImages(){return IMAGES.filter(nm=>OWNER[nm]==='human'&&(COUNTS[nm]||0)>0);}
function loadDevImg(){if(!DEV_IMG)return;devImgEl.onload=drawDevOverlay;devImgEl.src='/api/file/'+encodeURIComponent(DEV_IMG);}
function devNextImage(){const hs=humanImages();if(!hs.length)return;DEV_IMG=hs[(hs.indexOf(DEV_IMG)+1)%hs.length];loadDevImg();updateDevOverlay();}
async function updateDevOverlay(){
  if(!DEV_IMG){const hs=humanImages();if(!hs.length){$('devNote').textContent='Needs at least one green (human) seed image.';return;}DEV_IMG=hs[0];loadDevImg();}
  try{const d=await postJSON('/api/train/deviation_preview',{image:DEV_IMG,config:gateConfig()});
    if(d.detail){$('devNote').textContent=d.detail;return;}DEV_DATA=d;drawDevOverlay();}catch(e){}}
/* Zoom / pan / split state for the review window. The canvas is rendered at a FIXED high internal
   resolution and scaled with CSS, so zooming never re-runs the (costly) preview request and the
   boxes stay crisp on the way up. */
const DEV_RES=1400;                       // internal canvas long side — headroom for 6× zoom
let DEV_ZOOM=1;                           // 1 = fit the viewport width
function devFitWidth(){const v=$('devView');const cv=$('devCanvas');
  if(!v||!cv||!cv.width)return 0;
  return Math.max(60,v.clientWidth-2);}
function applyDevZoom(){const cv=$('devCanvas');if(!cv||!cv.width)return;
  const w=devFitWidth()*DEV_ZOOM;
  if(w>0){cv.style.width=Math.round(w)+'px';                       // explicit height: canvas + height:auto
    cv.style.height=Math.round(w*cv.height/cv.width)+'px';}        // is not reliable across browsers
  const z=$('devZoom'),lbl=$('devZoomV');
  if(z)z.value=DEV_ZOOM;if(lbl)lbl.textContent=DEV_ZOOM.toFixed(1)+'×';
  try{localStorage.setItem('pglabel-devzoom',DEV_ZOOM);}catch(e){}}
function setDevZoom(v){DEV_ZOOM=Math.min(6,Math.max(0.5,+v||1));applyDevZoom();}
function stepDevZoom(d){setDevZoom(DEV_ZOOM+d);}
function fitDev(){setDevZoom(1);const v=$('devView');if(v){v.scrollLeft=0;v.scrollTop=0;}}
let CROP_PX=76;                           // sample thumbnail size, shared by all three group headers
function setCropZoom(px){CROP_PX=Math.min(240,Math.max(48,Math.round(+px||76)));
  const r=$('gRight');if(r)r.style.setProperty('--cropw',CROP_PX+'px');
  document.querySelectorAll('.cropz').forEach(el=>{el.value=CROP_PX;});      // keep every copy in step
  document.querySelectorAll('.cropv').forEach(el=>{el.textContent=CROP_PX+'px';});
  try{localStorage.setItem('pglabel-cropw',CROP_PX);}catch(e){}}
function stepCropZoom(d){setCropZoom(CROP_PX+d);}
function setDevHeight(px){const r=$('gRight');if(!r)return;
  const n=Math.round(Math.min(window.innerHeight*0.62,Math.max(120,px)));
  r.style.setProperty('--devh',n+'px');applyDevZoom();
  try{localStorage.setItem('pglabel-devh',n);}catch(e){}}
function restoreGateView(){
  try{const z=parseFloat(localStorage.getItem('pglabel-devzoom'));if(z)DEV_ZOOM=Math.min(6,Math.max(0.5,z));}catch(e){}
  try{const c=parseInt(localStorage.getItem('pglabel-cropw'));setCropZoom(c||76);}catch(e){setCropZoom(76);}
  try{const h=parseInt(localStorage.getItem('pglabel-devh'));if(h)setDevHeight(h);}catch(e){}
  applyDevZoom();}
/* wheel = zoom about the cursor; drag = pan (only meaningful once zoomed in) */
(function wireDevView(){const v=$('devView');if(!v)return;
  v.addEventListener('wheel',e=>{e.preventDefault();
    const cv=$('devCanvas'),vr=v.getBoundingClientRect(),cr=cv.getBoundingClientRect();
    const px=(e.clientX-cr.left)/Math.max(1,cr.width),      // the image point under the cursor, 0…1
          py=(e.clientY-cr.top)/Math.max(1,cr.height);
    const vx=e.clientX-vr.left,vy=e.clientY-vr.top;         // where that point sits in the viewport
    setDevZoom(DEV_ZOOM*(e.deltaY<0?1.12:1/1.12));
    v.scrollLeft=px*cv.clientWidth-vx;                      // …and keep it there after the zoom
    v.scrollTop=py*cv.clientHeight-vy;},{passive:false});
  let pan=null;
  v.addEventListener('mousedown',e=>{pan={x:e.clientX,y:e.clientY,l:v.scrollLeft,t:v.scrollTop};
    v.classList.add('grabbing');e.preventDefault();});
  window.addEventListener('mousemove',e=>{if(!pan)return;
    v.scrollLeft=pan.l-(e.clientX-pan.x);v.scrollTop=pan.t-(e.clientY-pan.y);});
  window.addEventListener('mouseup',()=>{pan=null;v.classList.remove('grabbing');});
  v.addEventListener('dblclick',()=>fitDev());})();
/* drag the bar between the preview and the sample grid */
(function wireSplit(){const s=$('gSplit');if(!s)return;let dr=null;
  s.addEventListener('mousedown',e=>{const r=$('gRight');
    dr={y:e.clientY,h:parseInt(getComputedStyle(r).getPropertyValue('--devh'))||240};
    s.classList.add('active');document.body.style.userSelect='none';e.preventDefault();});
  window.addEventListener('mousemove',e=>{if(!dr)return;setDevHeight(dr.h+(e.clientY-dr.y));});
  window.addEventListener('mouseup',()=>{if(!dr)return;dr=null;s.classList.remove('active');
    document.body.style.userSelect='';});})();
window.addEventListener('resize',()=>{if(GATE_ON)applyDevZoom();});

function drawDevOverlay(){const cv=$('devCanvas');if(!cv||!DEV_DATA)return;const x=cv.getContext('2d');
  const sc=DEV_RES/Math.max(DEV_DATA.width,DEV_DATA.height);                 // fixed internal resolution (crisp)
  const W=Math.round(DEV_DATA.width*sc),H=Math.round(DEV_DATA.height*sc);
  const k=W/640;                                                             // keep stroke weights as designed
  cv.width=W;cv.height=H;x.clearRect(0,0,W,H);
  if(devImgEl.complete&&devImgEl.naturalWidth)x.drawImage(devImgEl,0,0,W,H);
  const bx=b=>[(b.cx-b.w/2)*W,(b.cy-b.h/2)*H,b.w*W,b.h*H];
  x.setLineDash([]);x.lineWidth=8*k;x.strokeStyle='#22c55e';                 // GT = THICK solid green
  (DEV_DATA.gt||[]).forEach(b=>{const[X,Y,Wd,Hd]=bx(b);x.strokeRect(X,Y,Wd,Hd);});
  x.setLineDash([9*k,5*k]);x.lineWidth=3*k;x.strokeStyle='#ef4444';          // deviated = RED dashed (all)
  (DEV_DATA.deviated||[]).forEach(b=>{const[X,Y,Wd,Hd]=bx(b);x.strokeRect(X,Y,Wd,Hd);});
  x.setLineDash([]);applyDevZoom();
  $('devNote').innerHTML=`GT <b style="color:var(--ok)">${(DEV_DATA.gt||[]).length}</b> · deviated <b style="color:var(--danger)">${(DEV_DATA.deviated||[]).length}</b> · shift=<b>${DEV_DATA.shift}</b> · <span style="color:var(--mut2)">${DEV_IMG}</span> · <span style="color:var(--mut2)">scroll to zoom · drag to pan · double-click to fit</span>`;}

function openGate(g){GATE_ON=true;fillGateControls(g.noise_config||(g.crops||{}).config);
  $('gate').classList.add('open');   // open FIRST: the sizes below need a laid-out modal (clientWidth ≠ 0)
  restoreGateView();                 // zoom · crop size · split height, as you last left them
  renderGateCrops(g.crops);          // …so the group headers render at the restored crop size
  DEV_IMG=null;$('gateMsg').textContent='';gateBusy(false,'');
  updateDevOverlay();}
function closeGate(){GATE_ON=false;$('gate').classList.remove('open');}
function maybeGate(s){if(s&&s.gate&&s.gate.active){if(!GATE_ON)openGate(s.gate);}else if(GATE_ON)closeGate();}
function gateBusy(on,msg){['gateRegen','gateConfirm'].forEach(id=>$(id).disabled=on);$('gateMsg').textContent=msg||'';}
async function regenGate(){gateBusy(true,'⏳ regenerating crops…');
  const r=await postJSON('/api/train/regen',{config:gateConfig()});gateBusy(false,'');
  if(r.detail){toast('Regeneration failed: '+r.detail,'err');return;}
  renderGateCrops(r.crops);toast('Preview updated.','ok');}
async function confirmGate(){gateBusy(true,'⏳ generating the approved crops…');
  const r=await postJSON('/api/train/confirm',{config:gateConfig()});
  if(r.detail){gateBusy(false,'');toast('Confirm failed: '+r.detail,'err');return;}
  closeGate();toast('✅ Confirmed — training the filter on the approved crops.','ok');}
async function cancelGate(){if(!confirm('Cancel training?'))return;
  await postJSON('/api/train/cancel',{});closeGate();toast('Training canceled.','warn');}

/* ---------- export + misc ---------- */
async function exportCoco(){closeMenu();const r=await getJSON('/api/export?fmt=coco');
  toast(`Exported COCO: ${r.images} imgs / ${r.annotations} anns →<br><small>${r.path}</small>`,'ok');}
async function exportYolo(){closeMenu();const r=await getJSON('/api/export?fmt=yolo');
  toast(`YOLO labels saved in-place →<br><small>${r.labels_dir}</small>`,'ok');}
function toggleMenu(e){e.stopPropagation();$('exmenu').classList.toggle('open');}
function closeMenu(){$('exmenu').classList.remove('open');}
document.addEventListener('click',closeMenu);
function toggleHelp(){$('overlay').classList.toggle('open');}
function toggleTheme(){const light=document.body.classList.toggle('light');
  $('themeBtn').textContent=light?'🌞':'🌗';
  try{localStorage.setItem('pglabel-theme',light?'light':'dark');}catch(e){}}
try{if(localStorage.getItem('pglabel-theme')==='light'){document.body.classList.add('light');$('themeBtn').textContent='🌞';}}catch(e){}
window.addEventListener('resize',()=>{if(img.complete){fit();draw();}});
/* ---------- resizable side panels (drag the bars between panels) ---------- */
let _rz=null;
function startResize(e,side){e.preventDefault();
  const varName=side==='L'?'--leftw':'--rightw';
  const cur=parseFloat(getComputedStyle(document.body).getPropertyValue(varName))||(side==='L'?222:320);
  _rz={side,varName,startX:e.clientX,startW:cur};
  document.body.style.cursor='col-resize';document.body.style.userSelect='none';
  e.currentTarget.classList.add('active');}
window.addEventListener('mousemove',e=>{if(!_rz)return;
  const dx=e.clientX-_rz.startX;
  let w=_rz.side==='L'?_rz.startW+dx:_rz.startW-dx;   // right panel grows when dragging left
  w=Math.max(150,Math.min(620,w));
  document.body.style.setProperty(_rz.varName,w+'px');
  if(img&&img.complete){fit();draw();}});
window.addEventListener('mouseup',()=>{if(!_rz)return;
  document.body.style.cursor='';document.body.style.userSelect='';
  document.querySelectorAll('.resizer.active').forEach(r=>r.classList.remove('active'));
  try{localStorage.setItem('pglabel-'+(_rz.side==='L'?'leftw':'rightw'),
      getComputedStyle(document.body).getPropertyValue(_rz.varName).trim());}catch(e){}
  _rz=null;if(img&&img.complete){fit();draw();}});
try{['leftw','rightw'].forEach(k=>{const v=localStorage.getItem('pglabel-'+k);
  if(v)document.body.style.setProperty('--'+k,v);});}catch(e){}
['suImages','suClasses','suLabels'].forEach(id=>$(id).addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();startSetup();}}));
init();
