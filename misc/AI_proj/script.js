const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
const TOTAL=8;
const state={photo:null,flowers:[{n:'Rose (Pink)',c:'#d98a98',q:7},{n:'Rose (White)',c:'#efeadf',q:6}],
  style:0,container:0,history:['home']};
const STYLES=[
 {n:'Soft & Romantic',d:'Airy, flowing, with a gentle garden feel.',c:'#e48a9e',name:'Garden Whisper',txt:'A soft, romantic arrangement designed around your blooms.'},
 {n:'Wild Garden',d:'Organic, free-flowing, and beautifully uneven.',c:'#c9a35a',name:'Meadow Tangle',txt:'A loose, natural gathering that looks freshly picked.'},
 {n:'Modern Minimal',d:'Sculptural lines with intentional negative space.',c:'#7a9a7e',name:'Quiet Line',txt:'A clean, sculptural design that lets each bloom stand out.'}];
const CONTS=[{n:'Bouquet',d:'Hand-tied, no vessel'},{n:'Vase',d:'Tall, narrow shape'},{n:'Vase',d:'Short, wide shape'},{n:'Mug',d:'Small, round shape'}];

function toast(m){const t=$('#toast');t.textContent=m;t.classList.add('show');clearTimeout(t._t);t._t=setTimeout(()=>t.classList.remove('show'),2200)}

function go(id,back){
  const cur=$('.screen:not([hidden])'); if(cur) cur.hidden=true;
  const el=$('#'+id); el.hidden=false; el.scrollIntoView?.({block:'start'}); window.scrollTo(0,0);
  if(!back && state.history[state.history.length-1]!==id) state.history.push(id);
  const step=+el.dataset.step, chrome=step>0;
  $('#top').hidden=!chrome; $('#progress').hidden=!chrome;
  $('#stepLabel').textContent=`STEP ${step} OF ${TOTAL}`;
  const pct=Math.round(step/TOTAL*100); $('#stepPct').textContent=pct+'%'; $('#bar').style.width=pct+'%';
  if(id==='found') renderInv(); if(id==='review') renderReview();
  if(id==='crop') $('#cropimg').style.backgroundImage=photoCSS();
}
$$('[data-go]').forEach(b=>b.addEventListener('click',()=>go(b.dataset.go)));
$('#back').onclick=()=>{ if(state.history.length>1){state.history.pop();go(state.history[state.history.length-1],true)} };
$('#restart').onclick=()=>{state.history=['home'];state.photo=null;$('#preview').hidden=true;$('#dropEmpty').hidden=false;go('home',true)};

/* upload: after choosing or taking a photo, go straight to crop */
function setPhoto(url){state.photo=url;const p=$('#preview');p.src=url;p.hidden=false;$('#dropEmpty').hidden=true;go('crop')}
function onPick(e){const f=e.target.files[0];e.target.value='';if(!f)return;
  if(f.size>12*1048576)return toast('Photo is over 12MB — choose a smaller one');setPhoto(URL.createObjectURL(f))}
$('#file').onchange=onPick;$('#cam').onchange=onPick;
$('#takePhoto').onclick=()=>$('#cam').click();
$('#sample').onclick=()=>{state.photo=sampleSVG();go('found')};
/* load results/bouquet_input/bouquet.json made by flower_counter_demo_4ชนิด.ipynb */
$('#loadAI').onclick=()=>$('#aiFile').click();
$('#aiFile').onchange=async e=>{const f=e.target.files[0];e.target.value='';if(!f)return;
  try{const d=JSON.parse(await f.text());
    const fl=(d.flowers||[]).filter(x=>x&&typeof x.n==='string'&&/^#[0-9a-f]{6}$/i.test(x.c)&&x.q>=1)
      .map(x=>({n:x.n.replace(/[<>&"]/g,''),c:x.c,q:Math.round(x.q)}));
    if(!fl.length)return toast('No flowers in this file');
    state.flowers=fl;const si=STYLES.findIndex(s=>s.n===d.style);if(si>=0)state.style=si;
    const ci=['bouquet','narrow vase','wide vase','mug'].findIndex(k=>String(d.container||'').includes(k));if(ci>=0)state.container=ci;
    state.photo=sampleSVG();go('found');toast(`Loaded ${fl.length} flower types from AI`)}
  catch{toast('Could not read this file')}};
function sampleSVG(){const s=`<svg xmlns='http://www.w3.org/2000/svg' width='600' height='600'><rect width='600' height='600' fill='#cfd8d3'/><g stroke='#6f8f72' stroke-width='6'><path d='M300 600V330M250 600L270 330M350 600L330 330'/></g><g><circle cx='300' cy='290' r='55' fill='#e48a9e'/><circle cx='235' cy='320' r='48' fill='#f3c9cf'/><circle cx='365' cy='320' r='48' fill='#d56b86'/><circle cx='300' cy='235' r='44' fill='#f7e2e3'/><circle cx='260' cy='255' r='38' fill='#fff'/></g></svg>`;return 'data:image/svg+xml;utf8,'+encodeURIComponent(s)}
function photoCSS(){return state.photo?`url("${state.photo}")`:'none'}

/* crop: drag + zoom */
(()=>{const box=$('#cropbox'),img=$('#cropimg');let x=0,y=0,z=1,drag=null;
 const apply=()=>img.style.transform=`translate(${x}px,${y}px) scale(${z})`;
 $('#zoom').oninput=e=>{z=+e.target.value;apply()};
 box.onpointerdown=e=>{drag={x:e.clientX-x,y:e.clientY-y};box.setPointerCapture(e.pointerId);box.style.cursor='grabbing'};
 box.onpointermove=e=>{if(!drag)return;x=e.clientX-drag.x;y=e.clientY-drag.y;apply()};
 box.onpointerup=()=>{drag=null;box.style.cursor='grab'};
})();
$('#analyze').onclick=()=>{load('Identifying your flowers…',1400,()=>go('found'))};
function load(msg,ms,done){$('#loading p').textContent=msg;$('#loading').hidden=false;setTimeout(()=>{$('#loading').hidden=true;done()},ms)}

/* flower inventory */
function renderInv(){
  const ul=$('#inv');ul.innerHTML='';
  state.flowers.forEach((f,i)=>{const li=document.createElement('li');
    li.innerHTML=`<span class="sw" style="background:${f.c};border:1px solid #e4dfd2"></span><span class="name">${f.n}</span>
    <button aria-label="Fewer">−</button><span>${f.q}</span><button aria-label="More">+</button><button aria-label="Remove ${f.n}">🗑</button>`;
    const [m,p,d]=$$('button',li);
    m.onclick=()=>{f.q=Math.max(1,f.q-1);renderInv()}; p.onclick=()=>{f.q++;renderInv()};
    d.onclick=()=>{state.flowers.splice(i,1);renderInv()}; ul.append(li)});
  const stems=state.flowers.reduce((a,f)=>a+f.q,0);
  $('#varCount').textContent=`${state.flowers.length} variet${state.flowers.length===1?'y':'ies'}`;
  $('#stemCount').textContent=`${stems} stems detected`;
}
const sheet=$('#sheet');
function openSheet(){$('#sheetChoose').hidden=false;$('#sheetForm').hidden=true;$('#flowerName').value='';sheet.hidden=false;$('#optPhoto').focus()}
function closeSheet(){sheet.hidden=true}
$('#addFlower').onclick=openSheet;
$('#sheetCancel').onclick=closeSheet;
sheet.onclick=e=>{if(e.target===sheet)closeSheet()};
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!sheet.hidden)closeSheet()});
$('#optPhoto').onclick=()=>{closeSheet();go('upload')};
$('#optType').onclick=()=>{$('#sheetChoose').hidden=true;$('#sheetForm').hidden=false;$('#flowerName').focus()};
$('#sheetForm').onsubmit=e=>{e.preventDefault();const n=$('#flowerName').value.trim();if(!n)return $('#flowerName').focus();
  const cols=['#f0b44c','#b79ad6','#e56b5d','#8fb7d6'];state.flowers.push({n,c:cols[state.flowers.length%4],q:1});
  closeSheet();renderInv();toast(n+' added')};

/* style + container */
function optList(root,items,key,thumb){
  root.innerHTML='';items.forEach((it,i)=>{const b=document.createElement('button');b.className='opt'+(state[key]===i?' on':'');
    b.innerHTML=`<div class="thumb art" style="${thumb(it)}"></div><div><b>${it.n}</b><small>${it.d}</small></div><span class="radio">✓</span>`;
    b.onclick=()=>{state[key]=i;optList(root,items,key,thumb)};root.append(b)})}
optList($('#styleOpts'),STYLES,'style',s=>`background:radial-gradient(circle at 40% 40%,${s.c} 0 30%,transparent 31%),#e8ebe0`);
optList($('#contOpts'),CONTS,'container',()=>'background:radial-gradient(ellipse at 50% 60%,#fff 0 40%,transparent 41%),#efe9dd');

/* review + result */
function renderReview(){
  const s=state.flowers.reduce((a,f)=>a+f.q,0);
  $('#rvFlowers').textContent=`${s} stems · ${state.flowers.length} varieties`;
  $('#rvStyle').textContent=STYLES[state.style].n;$('#rvCont').textContent=CONTS[state.container].n}
$('#generate').onclick=()=>load('Arranging your blooms…',2200,()=>{
  const s=STYLES[state.style];$('#resName').textContent=s.name;$('#resDesc').textContent=s.txt;go('result')});
$('#save').onclick=()=>toast('Image saved to your device');
$('#share').onclick=async()=>{try{await navigator.share({title:'My Bloom Creft arrangement',url:location.href})}catch{toast('Link copied')}};

go('home',true);
