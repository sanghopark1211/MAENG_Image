# -*- coding: utf-8 -*-
"""
============================================================
 초고속 카메라 이미지 비교 뷰어 — image_compare.py
------------------------------------------------------------
 [목적] 조건 폴더(예: 15_100_30)마다 들어 있는 초고속 카메라
        Phantom .cine 영상(또는 tif/png 시퀀스)을, PCC 를 일일이 열지
        않고 브라우저에서 두 조건을 나란히 놓고 재생·비교한다.

 [사용법]
    python image_compare.py                # 폴더 선택창이 뜸
    python image_compare.py "D:\\데이터 경로"  # 상위 폴더 직접 지정
    → 브라우저가 자동으로 열림 (http://localhost:8765)

 [기능]
    · 좌/우 패널에 각각 조건 선택 (검색 가능한 목록)
    · 재생 / 일시정지 / 프레임 단위 이동 / 슬라이더 / 재생 속도
    · 동기 재생 (두 시퀀스를 같은 프레임 번호로 잠금) / 개별 재생
    · 프레임 번호·파일명 표시, 루프 재생
    · TIF 도 표시 가능 (서버에서 JPEG 로 실시간 변환, 캐시)

 [폴더 구조 가정]
    <상위폴더> 아래(하위 폴더 포함)의 모든 .cine 파일 = 시퀀스 1개.
    tif/png 시퀀스 폴더(MIN_FRAMES 장 이상)도 함께 인식한다.

 · 의존: Python 3.8+, Pillow, pycine  (pip install pillow pycine)
============================================================
"""
import io
import os
import re
import sys
import json
import threading
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, quote, unquote

from PIL import Image

# ===== 설정 =====================================================
PORT        = 8765
MIN_FRAMES  = 5          # (이미지 폴더 인식용) 이 장수 이상이면 시퀀스로 인식
IMG_EXTS    = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
CINE_STRIDE = 1          # cine 프레임 간격 (2 로 두면 짝수 프레임만 → 목록 절반)
IMG_FORMAT   = 'png'     # ★ 'png' = 무손실(화질 원본 그대로, 기본)
                         #   'jpeg' = 손실 압축(느린 디스크/원격일 때만)
JPEG_QUALITY = 95        # IMG_FORMAT='jpeg' 일 때 품질
MAX_EDGE    = 1024       # 전송 이미지 최대 변 [px] (0 = 원본 크기 그대로)
CACHE_FRAMES = 512       # 변환 결과 LRU 캐시 장수
DEFAULT_ROOT = r"F:\실시간 재료거동\박상호\1) 압연재 조건별 비교 데이터\초고속 카메라 이미지 데이터"
# ================================================================


def natural_key(s):
    """Img000010 < Img000100 처럼 숫자를 숫자답게 정렬"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def cine_frame_count(path):
    """cine 헤더에서 저장된 프레임 수를 읽는다 (pycine)"""
    from pycine.file import read_header
    h = read_header(path)
    return int(h['cinefileheader'].ImageCount)


def scan_sequences(root):
    """root 아래의 .cine 파일 + 이미지 폴더를 전부 시퀀스로 수집한다.
       반환: {표시이름: {'type':'cine'|'imgs', ...}}"""
    seqs = {}
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)

        # (a) Phantom cine — 파일 하나가 시퀀스 하나
        for f in filenames:
            if os.path.splitext(f)[1].lower() == '.cine':
                path = os.path.join(dirpath, f)
                try:
                    n = cine_frame_count(path)
                except Exception as e:
                    print(f'  [경고] cine 헤더 읽기 실패: {path} ({e})')
                    continue
                name = os.path.normpath(os.path.join(rel, f)) if rel != '.' else f
                seqs[name] = {'type': 'cine', 'path': path, 'n': n}

        # (b) 이미지 폴더 시퀀스 (tif/png ...)
        imgs = [f for f in filenames
                if os.path.splitext(f)[1].lower() in IMG_EXTS
                and not f.lower().startswith('mask')]
        if len(imgs) >= MIN_FRAMES:
            imgs.sort(key=natural_key)
            name = rel if rel != '.' else '(root)'
            seqs[name] = {'type': 'imgs', 'dir': dirpath, 'files': imgs,
                          'n': len(imgs)}
    return seqs


class State:
    root = None
    seqs = {}


S = State()


def _cine_frame_to_pil(path, idx):
    """cine 의 idx(0-기준) 프레임을 8-bit PIL 이미지로"""
    import numpy as np
    from pycine.raw import read_frames
    gen, setup, bpp = read_frames(path, start_frame=idx, count=1)
    fr = next(gen)                                   # uint8 또는 uint16 (bpp 비트 유효)
    if fr.dtype != np.uint8:
        shift = max(0, int(bpp) - 8)
        fr = (fr >> shift).astype(np.uint8)          # 상위 8비트 → 밝기 고정(깜빡임 없음)
    return Image.fromarray(fr)      # 2-D→L, 3-D→RGB 자동 (mode 인자는 deprecated)


@lru_cache(maxsize=CACHE_FRAMES)
def load_frame_jpeg(seq_name, idx):
    """시퀀스의 idx 번째 프레임을 JPEG 바이트로 변환 (LRU 캐시)"""
    info = S.seqs.get(seq_name)
    if not info or not (0 <= idx < info['n']):
        return None
    try:
        if info['type'] == 'cine':
            im = _cine_frame_to_pil(info['path'], idx * CINE_STRIDE)
        else:
            im = Image.open(os.path.join(info['dir'], info['files'][idx]))
            if im.mode not in ('RGB', 'L'):
                im = im.convert('RGB')
        if MAX_EDGE and max(im.size) > MAX_EDGE:
            im.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
        buf = io.BytesIO()
        if IMG_FORMAT == 'png':
            # 무손실 — 스펙클/입자 텍스처가 뭉개지지 않는다 (grayscale 은 L 그대로)
            im.save(buf, 'PNG', compress_level=1)
        else:
            im.convert('RGB').save(buf, 'JPEG', quality=JPEG_QUALITY,
                                   subsampling=0)
        return buf.getvalue()
    except Exception as e:
        print(f'  [경고] 프레임 변환 실패: {seq_name}[{idx}] ({e})')
        return None


PAGE = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>MAENG 이미지 비교 뷰어</title>
<style>
  :root { --bg:#14161a; --panel:#1e2128; --line:#2e333d; --fg:#e8eaf0;
          --sub:#9aa3b2; --acc:#4da3ff; --acc2:#ff7a4d; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--fg);
         font-family:'Malgun Gothic','Segoe UI',sans-serif; font-size:14px; }
  header { padding:10px 16px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  header h1 { font-size:16px; font-weight:700; }
  header .root { color:var(--sub); font-size:12px; }
  main { display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:10px; }
  .pane { background:var(--panel); border:1px solid var(--line);
          border-radius:8px; padding:10px; display:flex;
          flex-direction:column; gap:8px; }
  .pane select, .pane input[type=text] {
      width:100%; background:#12151a; color:var(--fg);
      border:1px solid var(--line); border-radius:5px; padding:6px 8px; }
  .imgbox { position:relative; background:#000; border-radius:6px;
            overflow:hidden; aspect-ratio:1/1; }
  .imgbox img { width:100%; height:100%; object-fit:contain; display:block;
                image-rendering:auto; }
  body.pix .imgbox img { image-rendering:pixelated; }  /* 픽셀 보기 모드 */
  .tag { position:absolute; left:8px; top:8px; background:rgba(0,0,0,.55);
         padding:2px 8px; border-radius:4px; font-size:12px; }
  .tagL { border-left:3px solid var(--acc); }
  .tagR { border-left:3px solid var(--acc2); }
  .row { display:flex; align-items:center; gap:8px; }
  .row input[type=range] { flex:1; }
  .fno { min-width:110px; text-align:right; color:var(--sub);
         font-variant-numeric:tabular-nums; font-size:12px; }
  footer { padding:8px 16px 14px; display:flex; align-items:center;
           gap:10px; flex-wrap:wrap; border-top:1px solid var(--line); }
  button { background:#262b34; color:var(--fg); border:1px solid var(--line);
           border-radius:6px; padding:7px 14px; cursor:pointer; font-size:14px; }
  button:hover { background:#303743; }
  button.on { background:var(--acc); border-color:var(--acc); color:#fff; }
  button.big { font-weight:700; padding:8px 18px; }
  .spd { width:70px; }
  label.chk { display:flex; gap:5px; align-items:center; color:var(--sub);
              cursor:pointer; user-select:none; }
  kbd { background:#262b34; border:1px solid var(--line); border-radius:4px;
        padding:0 5px; font-size:11px; color:var(--sub); }
  .help { color:var(--sub); font-size:12px; margin-left:auto; }
</style></head><body>
<header>
  <h1>MAENG 이미지 비교 뷰어</h1>
  <span class="root" id="rootLabel"></span>
</header>

<main>
  <div class="pane">
    <input type="text" id="filterL" placeholder="조건 검색 (예: 15_100)">
    <select id="selL" size="1"></select>
    <div class="imgbox"><img id="imgL"><div class="tag tagL" id="tagL">-</div></div>
    <div class="row">
      <button id="playL">▶</button>
      <button id="prevL">-1</button>
      <button id="nextL">+1</button>
      <input type="range" id="sldL" min="0" max="0" value="0">
      <span class="fno" id="fnoL">- / -</span>
    </div>
  </div>

  <div class="pane">
    <input type="text" id="filterR" placeholder="조건 검색 (예: 165_100)">
    <select id="selR" size="1"></select>
    <div class="imgbox"><img id="imgR"><div class="tag tagR" id="tagR">-</div></div>
    <div class="row">
      <button id="playR">▶</button>
      <button id="prevR">-1</button>
      <button id="nextR">+1</button>
      <input type="range" id="sldR" min="0" max="0" value="0">
      <span class="fno" id="fnoR">- / -</span>
    </div>
  </div>
</main>

<footer>
  <button id="btnPlayAll" class="big">▶▶ 동시 재생</button>
  <button id="btnStopAll" class="big">⏸ 동시 정지</button>
  <button id="btnPrev">◀ 둘 다 1프레임</button>
  <button id="btnNext">둘 다 1프레임 ▶</button>
  <button id="btnHome">⟲ 둘 다 처음</button>
  <span>속도</span>
  <select id="spd" class="spd">
    <option value="60">60 fps</option><option value="30" selected>30 fps</option>
    <option value="15">15 fps</option><option value="5">5 fps</option>
    <option value="2">2 fps</option><option value="1">1 fps</option>
  </select>
  <label class="chk"><input type="checkbox" id="chkLoop" checked> 루프</label>
  <label class="chk"><input type="checkbox" id="chkPix"> 픽셀 선명 보기</label>
  <span class="help">
    <kbd>Space</kbd> 동시 재생/정지 <kbd>←</kbd><kbd>→</kbd> 둘 다 1프레임
    <kbd>Home</kbd> 처음
  </span>
</footer>

<script>
let SEQS = {};            // {name: nFrames}
let names = [];
let st = { L:{name:null,n:0,i:0,playing:false},
           R:{name:null,n:0,i:0,playing:false} };
let timer = null;

const $ = id => document.getElementById(id);
const enc = s => encodeURIComponent(s);

async function init(){
  const r = await fetch('/api/list');  const d = await r.json();
  $('rootLabel').textContent = d.root + '   (시퀀스 ' + Object.keys(d.seqs).length + '개)';
  SEQS = d.seqs;  names = Object.keys(SEQS).sort();
  fillSel('L');  fillSel('R');
  if(names.length){ setSeq('L', names[0]); setSeq('R', names[Math.min(1,names.length-1)]); }
}
function fillSel(side, filter){
  const sel = $('sel'+side);  sel.innerHTML='';
  const f = (filter||'').toLowerCase();
  names.filter(n=>n.toLowerCase().includes(f)).forEach(n=>{
    const o=document.createElement('option'); o.value=n;
    o.textContent = n + '  ('+SEQS[n]+'장)';  sel.appendChild(o);
  });
}
function setSeq(side, name){
  if(!name) return;
  st[side] = {name:name, n:SEQS[name], i:0, playing:false};
  updatePlayBtns();
  $('sel'+side).value = name;
  const sld = $('sld'+side);  sld.max = SEQS[name]-1;  sld.value = 0;
  show(side);
}
function show(side){
  const s = st[side];  if(!s.name) return;
  $('img'+side).src = '/img?seq='+enc(s.name)+'&i='+s.i;
  $('tag'+side).textContent = s.name;
  $('fno'+side).textContent = (s.i+1)+' / '+s.n;
  $('sld'+side).value = s.i;
}
function stepSide(side, d){
  const s = st[side];  if(!s.name) return;
  const loop = $('chkLoop').checked;
  let i = s.i + d;
  if(loop){ i = ((i % s.n) + s.n) % s.n; }
  else{
    i = Math.max(0, Math.min(s.n-1, i));
    if(i === s.i && d > 0){ s.playing = false; updatePlayBtns(); }  // 끝 도달 → 정지
  }
  s.i = i;  show(side);
}
function tick(){
  ['L','R'].forEach(side=>{ if(st[side].playing) stepSide(side, 1); });
  if(!st.L.playing && !st.R.playing) stopTimer();
}
function startTimer(){
  stopTimer();
  const fps = parseFloat($('spd').value);
  timer = setInterval(tick, 1000/fps);
}
function stopTimer(){ clearInterval(timer); timer = null; }
function updatePlayBtns(){
  $('playL').textContent = st.L.playing ? '⏸' : '▶';
  $('playR').textContent = st.R.playing ? '⏸' : '▶';
  $('playL').classList.toggle('on', st.L.playing);
  $('playR').classList.toggle('on', st.R.playing);
  $('btnPlayAll').classList.toggle('on', st.L.playing && st.R.playing);
  const any = st.L.playing || st.R.playing;
  if(any && !timer) startTimer();
  if(!any) stopTimer();
}
function setPlay(side, on){ st[side].playing = on; updatePlayBtns(); }

// --- 개별 컨트롤 ---
['L','R'].forEach(side=>{
  $('play'+side).onclick = ()=>setPlay(side, !st[side].playing);
  $('prev'+side).onclick = ()=>{ setPlay(side,false); stepSide(side,-1); };
  $('next'+side).onclick = ()=>{ setPlay(side,false); stepSide(side, 1); };
  $('sel'+side).onchange = e=>{ setSeq(side, e.target.value); };
  $('filter'+side).oninput = e=>fillSel(side, e.target.value);
  // ★ 슬라이더는 자기 패널만 움직인다 (연동 제거)
  $('sld'+side).oninput = e=>{
    setPlay(side, false);
    st[side].i = parseInt(e.target.value);
    show(side);
  };
});

// --- 동시 컨트롤 ---
$('btnPlayAll').onclick = ()=>{ st.L.playing=true; st.R.playing=true; updatePlayBtns(); };
$('btnStopAll').onclick = ()=>{ st.L.playing=false; st.R.playing=false; updatePlayBtns(); };
$('btnNext').onclick = ()=>{ stepSide('L',1); stepSide('R',1); };
$('btnPrev').onclick = ()=>{ stepSide('L',-1); stepSide('R',-1); };
$('btnHome').onclick = ()=>{
  st.L.playing=false; st.R.playing=false; updatePlayBtns();
  st.L.i=0; st.R.i=0; show('L'); show('R');
};
$('spd').onchange = ()=>{ if(timer) startTimer(); };
$('chkPix').onchange = e=>document.body.classList.toggle('pix', e.target.checked);

document.addEventListener('keydown', e=>{
  if(e.target.tagName==='INPUT' && e.target.type==='text') return;
  if(e.code==='Space'){
    e.preventDefault();
    const on = !(st.L.playing && st.R.playing);
    st.L.playing = on;  st.R.playing = on;  updatePlayBtns();
  }
  if(e.code==='ArrowRight'){ stepSide('L',1); stepSide('R',1); }
  if(e.code==='ArrowLeft'){ stepSide('L',-1); stepSide('R',-1); }
  if(e.code==='Home'){ $('btnHome').onclick(); }
});
init();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):   # 콘솔 스팸 방지
        pass

    def _send(self, code, ctype, body, cache=False):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        if cache:
            self.send_header('Cache-Control', 'max-age=3600')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path == '/':
            self._send(200, 'text/html; charset=utf-8', PAGE.encode('utf-8'))

        elif u.path == '/api/list':
            data = {'root': S.root,
                    'seqs': {k: (v['n'] // CINE_STRIDE if v['type'] == 'cine'
                                 else v['n'])
                             for k, v in S.seqs.items()}}
            self._send(200, 'application/json; charset=utf-8',
                       json.dumps(data, ensure_ascii=False).encode('utf-8'))

        elif u.path == '/img':
            seq = unquote(q.get('seq', [''])[0])
            try:
                idx = int(q.get('i', ['0'])[0])
            except ValueError:
                idx = 0
            jpg = load_frame_jpeg(seq, idx)
            if jpg is None:
                self._send(404, 'text/plain', b'not found')
            else:
                ctype = 'image/png' if IMG_FORMAT == 'png' else 'image/jpeg'
                self._send(200, ctype, jpg, cache=True)
        else:
            self._send(404, 'text/plain', b'not found')


def pick_root():
    """인자 없으면 폴더 선택창 (tkinter), 그것도 안 되면 콘솔 입력"""
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        return sys.argv[1]
    try:
        import tkinter as tk
        from tkinter import filedialog
        r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)
        init = DEFAULT_ROOT if os.path.isdir(DEFAULT_ROOT) else os.getcwd()
        path = filedialog.askdirectory(title='이미지 조건 폴더들이 들어있는 상위 폴더 선택',
                                       initialdir=init)
        r.destroy()
        if path:
            return path
    except Exception:
        pass
    path = input('상위 폴더 경로 입력: ').strip('" ')
    return path


def main():
    root = pick_root()
    if not root or not os.path.isdir(root):
        print('폴더가 유효하지 않습니다.'); sys.exit(1)
    S.root = root
    print(f'스캔 중: {root}')
    S.seqs = scan_sequences(root)
    if not S.seqs:
        print(f'이미지 시퀀스({MIN_FRAMES}장 이상)를 찾지 못했습니다.'); sys.exit(1)
    print(f'시퀀스 {len(S.seqs)}개 발견:')
    for k in sorted(S.seqs):
        v = S.seqs[k]
        print(f"  [{v['type']}] {k}  ({v['n']}프레임)")

    url = f'http://localhost:{PORT}'
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    print(f'\n서버 시작: {url}   (종료 = Ctrl+C)')
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\n종료합니다.')


if __name__ == '__main__':
    main()
