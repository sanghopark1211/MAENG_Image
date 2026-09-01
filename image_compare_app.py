# -*- coding: utf-8 -*-
"""
============================================================
 초고속 카메라 이미지 비교 앱 — image_compare_app.py
------------------------------------------------------------
 [목적] Phantom .cine 영상 두 개를 데스크톱 창에서 나란히 재생·비교.
        (웹 브라우저판 image_compare.py 의 데스크톱 앱 버전)

 [v3 — 앱 전환의 이유]
   · 브라우저판은 프레임마다 HTTP 요청 → 재생 시 연결 끊김/오류 발생
   · 앱은 파일을 '연속 제너레이터'로 읽어 프레임당 ~4 ms (46 ms → 4 ms)
   · 컬러 센서(CFA=Bayer) 지원: 흑백으로 보이던 것은 raw Bayer 를
     그대로 표시했기 때문 → 화이트밸런스 + 디모자이크 + 감마 LUT 로
     PCC 와 같은 컬러 재현 (프레임당 ~5 ms, LUT 기반)
   · 무손실 표시 (JPEG 압축 없음) + 픽셀 선명 확대 옵션

 [기능]
   · 시작 시 상위 폴더 선택 → 하위 .cine 전부 목록화 (+ 파일 추가 버튼)
   · 패널별 재생/정지/1프레임/슬라이더 (완전 독립)
   · 동시 재생/정지/1프레임/처음 버튼
   · 속도(fps), 루프, 픽셀 선명 확대, 컬러/그레이 전환

 · 의존: Python 3.8+, Pillow, numpy, opencv-python, pycine
============================================================
"""
import os
import re
import sys
import time
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import cv2
from PIL import Image, ImageTk

# ===== 설정 =====================================================
DEFAULT_ROOT = r"F:\실시간 재료거동\박상호\1) 압연재 조건별 비교 데이터\초고속 카메라 이미지 데이터"
PANEL_SIZE   = 640        # 각 패널 표시 크기 [px]
FPS_LIST     = [60, 30, 15, 5, 2, 1]
DEFAULT_FPS  = 30
IMG_EXTS     = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
MIN_FRAMES   = 5
# ================================================================


# ---------------- cine 판독 (연속 제너레이터 + 컬러 파이프라인) ----------------
class CineReader:
    """cine 파일 1개. 순차 재생은 열린 제너레이터로(빠름), 점프는 재오픈."""

    def __init__(self, path):
        from pycine.file import read_header
        self.path = path
        h = read_header(path)
        self.n = int(h['cinefileheader'].ImageCount)
        self.setup = h['setup']
        self.cfa = int(self.setup.CFA)
        self.bpp = None          # 첫 read 에서 확정
        self._gen = None
        self._pos = -1           # 제너레이터가 다음에 내놓을 프레임 번호
        self._lut = None         # 감마 LUT (bpp 확정 후 생성)
        self._wb = None

    def _open_at(self, idx):
        from pycine.raw import read_frames
        gen, setup, bpp = read_frames(self.path, start_frame=idx)
        self._gen, self._pos, self.bpp = gen, idx, int(bpp)

    def read(self, idx):
        """idx(0-기준) 프레임의 raw 배열. 순차면 이어읽기, 아니면 재오픈."""
        idx = max(0, min(self.n - 1, idx))
        if self._gen is None or idx != self._pos:
            self._open_at(idx)
        try:
            fr = next(self._gen)
            self._pos = idx + 1
            return fr
        except StopIteration:
            self._gen = None
            return None

    # ---- 표시 변환 -------------------------------------------------
    def _build_luts(self):
        maxv = (1 << self.bpp) - 1
        x = np.linspace(0.0, 1.0, maxv + 1)
        # PCC 기본과 같은 sRGB 근사 감마 (setup.fGamma ≈ 2.2)
        gamma = float(getattr(self.setup, 'fGamma', 2.2)) or 2.2
        self._lut = np.clip(255.0 * np.power(x, 1.0 / gamma), 0, 255).astype(np.uint8)
        try:
            wb = self.setup.WBGain[0]
            self._wb = (float(wb.R), float(wb.B))
        except Exception:
            self._wb = (1.0, 1.0)
        # ★ 속도 최적화: WB 게인을 감마 LUT 에 합친 채널별 LUT
        #   (float 곱셈 없이 uint16 인덱싱만으로 WB+감마를 한 번에)
        rG, bG = self._wb
        self._lutR = np.clip(255.0 * np.power(np.clip(x * rG, 0, 1), 1.0 / gamma),
                             0, 255).astype(np.uint8)
        self._lutB = np.clip(255.0 * np.power(np.clip(x * bG, 0, 1), 1.0 / gamma),
                             0, 255).astype(np.uint8)

    def to_rgb8(self, fr, color=True):
        """raw → 8-bit 표시 이미지 (컬러면 WB+디모자이크+감마, 모노면 감마만)"""
        if self.bpp is None:
            self.bpp = 12
        if self._lut is None:
            self._build_luts()
        maxv = (1 << self.bpp) - 1

        if self.cfa == 0 or not color:
            # 모노 (또는 그레이 강제): Bayer 라도 감마만 입혀 회색으로
            g = self._lut[np.clip(fr, 0, maxv)]
            return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

        # ---- 컬러: 디모자이크 → 채널별 (WB+감마) LUT  [정수 연산만] ----
        #   WB 를 디모자이크 뒤 채널 단위로 적용 (pycine 은 raw 단계 적용이나
        #   결과 차이는 시각적으로 무시 가능, 속도는 3배 빠름)
        raw16 = np.minimum(fr, maxv).astype(np.uint16, copy=False)
        rgb = cv2.cvtColor(raw16, cv2.COLOR_BAYER_GB2RGB)   # pycine 과 동일 코드
        out = np.empty(rgb.shape, np.uint8)
        out[..., 0] = self._lutR[rgb[..., 0]]
        out[..., 1] = self._lut[rgb[..., 1]]
        out[..., 2] = self._lutB[rgb[..., 2]]
        return out


class ImgSeqReader:
    """tif/png 시퀀스 폴더 (보조 지원)"""

    def __init__(self, d, files):
        self.dir, self.files = d, files
        self.n = len(files)
        self.cfa = 0

    def read(self, idx):
        idx = max(0, min(self.n - 1, idx))
        im = Image.open(os.path.join(self.dir, self.files[idx]))
        if im.mode not in ('RGB', 'L'):
            im = im.convert('RGB')
        return np.array(im)

    def to_rgb8(self, fr, color=True):
        if fr.ndim == 2:
            return cv2.cvtColor(fr, cv2.COLOR_GRAY2RGB)
        return fr


# ---------------- 시퀀스 목록 ----------------
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def scan_root(root):
    """상위 폴더 아래의 .cine + 이미지 폴더를 전부 수집"""
    seqs = {}
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        for f in filenames:
            if os.path.splitext(f)[1].lower() == '.cine':
                name = os.path.normpath(os.path.join(rel, f)) if rel != '.' else f
                seqs[name.replace('\\', '/')] = ('cine', os.path.join(dirpath, f))
        imgs = [f for f in filenames
                if os.path.splitext(f)[1].lower() in IMG_EXTS
                and not f.lower().startswith('mask')]
        if len(imgs) >= MIN_FRAMES:
            imgs.sort(key=natural_key)
            name = (rel if rel != '.' else '(root)').replace('\\', '/')
            seqs[name] = ('imgs', dirpath, imgs)
    return seqs


# ---------------- 패널 (영상 1개) ----------------
class Panel:
    def __init__(self, app, parent, side, accent):
        self.app, self.side = app, side
        self.reader = None
        self.i = 0
        self.playing = False
        self.want_seek = None            # 슬라이더로 요청된 프레임
        self._sld_guard = False

        f = ttk.Frame(parent, padding=6)
        self.frame = f

        self.cmb = ttk.Combobox(f, state='readonly', width=58)
        self.cmb.pack(fill='x')
        self.cmb.bind('<<ComboboxSelected>>', self.on_select)

        self.canvas = tk.Canvas(f, width=PANEL_SIZE, height=PANEL_SIZE,
                                bg='black', highlightthickness=1,
                                highlightbackground=accent)
        self.canvas.pack(pady=4)
        self._imgid = self.canvas.create_image(PANEL_SIZE // 2, PANEL_SIZE // 2)
        self._photo = None

        row = ttk.Frame(f)
        row.pack(fill='x')
        self.btn_play = ttk.Button(row, text='▶', width=3,
                                   command=self.toggle_play)
        self.btn_play.pack(side='left')
        ttk.Button(row, text='-1', width=3,
                   command=lambda: self.step(-1)).pack(side='left', padx=2)
        ttk.Button(row, text='+1', width=3,
                   command=lambda: self.step(+1)).pack(side='left')
        self.sld = ttk.Scale(row, from_=0, to=0, command=self.on_slider)
        self.sld.pack(side='left', fill='x', expand=True, padx=6)
        self.lbl = ttk.Label(row, text='- / -', width=14, anchor='e')
        self.lbl.pack(side='right')

    # ---- 조작 ----
    def on_select(self, _=None):
        name = self.cmb.get()
        self.playing = False
        self.app.load_sequence(self, name)

    def toggle_play(self):
        if not self.reader:
            return
        self.playing = not self.playing
        self.app.update_buttons()

    def step(self, d):
        if not self.reader:
            return
        self.playing = False
        self.app.update_buttons()
        self.request(self.i + d)

    def on_slider(self, v):
        if self._sld_guard or not self.reader:
            return
        self.playing = False
        self.app.update_buttons()
        self.want_seek = int(float(v))

    def request(self, idx, loop=False):
        n = self.reader.n
        if loop:
            idx = idx % n
        idx = max(0, min(n - 1, idx))
        self.want_seek = idx

    # ---- 표시 (메인 스레드) ----
    def show(self, idx, photo):
        self.i = idx
        self._photo = photo
        self.canvas.itemconfig(self._imgid, image=photo)
        self.lbl.config(text=f'{idx + 1} / {self.reader.n}')
        self._sld_guard = True
        self.sld.set(idx)
        self._sld_guard = False


# ---------------- 앱 본체 ----------------
class App:
    def __init__(self, root_dir):
        self.tk = tk.Tk()
        self.tk.title('MAENG 이미지 비교 앱')
        self.tk.configure(bg='#1e2128')
        try:
            ttk.Style().theme_use('clam')
        except Exception:
            pass

        self.seqs = {}
        self.q = queue.Queue(maxsize=8)      # (panel, idx, PIL) 표시 큐
        self.stop_flag = False

        # ---- 상단 툴바 ----
        top = ttk.Frame(self.tk, padding=(8, 6))
        top.pack(fill='x')
        ttk.Button(top, text='폴더 다시 선택', command=self.pick_root).pack(side='left')
        ttk.Button(top, text='+ 파일 추가', command=self.add_files).pack(side='left', padx=4)
        self.lbl_root = ttk.Label(top, text='')
        self.lbl_root.pack(side='left', padx=8)

        # ---- 패널 2개 ----
        mid = ttk.Frame(self.tk)
        mid.pack()
        self.L = Panel(self, mid, 'L', '#4da3ff')
        self.R = Panel(self, mid, 'R', '#ff7a4d')
        self.L.frame.grid(row=0, column=0, padx=4)
        self.R.frame.grid(row=0, column=1, padx=4)

        # ---- 하단 공통 컨트롤 ----
        bot = ttk.Frame(self.tk, padding=(8, 4, 8, 8))
        bot.pack(fill='x')
        self.btn_all = ttk.Button(bot, text='▶▶ 동시 재생', command=self.play_all)
        self.btn_all.pack(side='left')
        ttk.Button(bot, text='⏸ 동시 정지', command=self.stop_all).pack(side='left', padx=3)
        ttk.Button(bot, text='◀ 둘 다 1프레임',
                   command=lambda: self.step_all(-1)).pack(side='left', padx=3)
        ttk.Button(bot, text='둘 다 1프레임 ▶',
                   command=lambda: self.step_all(+1)).pack(side='left')
        ttk.Button(bot, text='⟲ 둘 다 처음', command=self.home_all).pack(side='left', padx=3)

        ttk.Label(bot, text='  속도').pack(side='left')
        self.cmb_fps = ttk.Combobox(bot, state='readonly', width=4,
                                    values=[str(f) for f in FPS_LIST])
        self.cmb_fps.set(str(DEFAULT_FPS))
        self.cmb_fps.pack(side='left', padx=2)
        ttk.Label(bot, text='fps').pack(side='left')

        self.var_loop = tk.BooleanVar(value=True)
        ttk.Checkbutton(bot, text='루프', variable=self.var_loop).pack(side='left', padx=8)
        self.var_pix = tk.BooleanVar(value=False)
        ttk.Checkbutton(bot, text='픽셀 선명', variable=self.var_pix).pack(side='left')
        self.var_color = tk.BooleanVar(value=True)
        ttk.Checkbutton(bot, text='컬러', variable=self.var_color).pack(side='left', padx=8)

        ttk.Label(bot, text='Space=동시재생  ←→=1프레임  Home=처음'
                  ).pack(side='right')

        # ---- 단축키 ----
        self.tk.bind('<space>', lambda e: self.toggle_all())
        self.tk.bind('<Right>', lambda e: self.step_all(+1))
        self.tk.bind('<Left>', lambda e: self.step_all(-1))
        self.tk.bind('<Home>', lambda e: self.home_all())
        self.tk.protocol('WM_DELETE_WINDOW', self.on_close)

        # ---- 데이터 + 워커 ----
        self.set_root(root_dir)
        self.worker = threading.Thread(target=self.decode_loop, daemon=True)
        self.worker.start()
        self.tk.after(15, self.poll_queue)

    # ---- 시퀀스 관리 ----
    def set_root(self, root_dir):
        self.root_dir = root_dir
        self.seqs = scan_root(root_dir) if root_dir else {}
        self.lbl_root.config(text=f'{root_dir}   (시퀀스 {len(self.seqs)}개)')
        names = sorted(self.seqs)
        for p in (self.L, self.R):
            p.cmb.config(values=names)
        if names:
            self.L.cmb.set(names[0])
            self.L.on_select()
            self.R.cmb.set(names[1] if len(names) > 1 else names[0])
            self.R.on_select()

    def pick_root(self):
        d = filedialog.askdirectory(title='조건 폴더들이 든 상위 폴더 선택',
                                    initialdir=self.root_dir or DEFAULT_ROOT)
        if d:
            self.set_root(d)

    def add_files(self):
        fs = filedialog.askopenfilenames(
            title='추가할 영상 파일 선택 (Ctrl 다중 선택)',
            initialdir=self.root_dir or DEFAULT_ROOT,
            filetypes=[('Phantom cine', '*.cine'), ('모든 파일', '*.*')])
        added = 0
        for p in fs:
            if os.path.splitext(p)[1].lower() != '.cine':
                continue
            name = '/'.join(os.path.normpath(p).split(os.sep)[-2:])
            if name in self.seqs:
                name = p.replace('\\', '/')
            self.seqs[name] = ('cine', p)
            added += 1
        if added:
            names = sorted(self.seqs)
            for p in (self.L, self.R):
                cur = p.cmb.get()
                p.cmb.config(values=names)
                if cur:
                    p.cmb.set(cur)
            self.lbl_root.config(
                text=f'{self.root_dir}   (시퀀스 {len(self.seqs)}개)')

    def load_sequence(self, panel, name):
        info = self.seqs.get(name)
        if not info:
            return
        try:
            if info[0] == 'cine':
                panel.reader = CineReader(info[1])
            else:
                panel.reader = ImgSeqReader(info[1], info[2])
        except Exception as e:
            print(f'[경고] 열기 실패: {name} ({e})')
            return
        panel.i = 0
        panel.sld.config(to=panel.reader.n - 1)
        panel.want_seek = 0

    # ---- 동시 컨트롤 ----
    def play_all(self):
        for p in (self.L, self.R):
            if p.reader:
                p.playing = True
        self.update_buttons()

    def stop_all(self):
        self.L.playing = self.R.playing = False
        self.update_buttons()

    def toggle_all(self):
        on = not (self.L.playing and self.R.playing)
        for p in (self.L, self.R):
            if p.reader:
                p.playing = on
        self.update_buttons()

    def step_all(self, d):
        self.stop_all()
        for p in (self.L, self.R):
            if p.reader:
                p.request(p.i + d)

    def home_all(self):
        self.stop_all()
        for p in (self.L, self.R):
            if p.reader:
                p.request(0)

    def update_buttons(self):
        self.L.btn_play.config(text='⏸' if self.L.playing else '▶')
        self.R.btn_play.config(text='⏸' if self.R.playing else '▶')

    # ---- 디코드 워커 (백그라운드 스레드) ----
    def decode_loop(self):
        next_t = time.time()
        while not self.stop_flag:
            fps = float(self.cmb_fps.get() or DEFAULT_FPS)
            period = 1.0 / fps
            now = time.time()
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                # 재생 중이 아니어도 슬라이더 점프 요청은 즉시 처리
                self._serve_seeks()
                continue
            next_t = max(next_t + period, now - period)

            loop = self.var_loop.get()
            for p in (self.L, self.R):
                if p.want_seek is not None:
                    self._decode_show(p, p.want_seek)
                    p.want_seek = None
                elif p.playing and p.reader:
                    nxt = p.i + 1
                    if nxt >= p.reader.n:
                        if loop:
                            nxt = 0
                        else:
                            p.playing = False
                            continue
                    self._decode_show(p, nxt)

    def _serve_seeks(self):
        for p in (self.L, self.R):
            if p.want_seek is not None:
                idx = p.want_seek
                p.want_seek = None
                self._decode_show(p, idx)

    def _decode_show(self, panel, idx):
        r = panel.reader
        if not r:
            return
        fr = r.read(idx)
        if fr is None:
            return
        rgb = r.to_rgb8(fr, color=self.var_color.get())
        interp = cv2.INTER_NEAREST if self.var_pix.get() else cv2.INTER_CUBIC
        h, w = rgb.shape[:2]
        scale = min(PANEL_SIZE / w, PANEL_SIZE / h)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                         interpolation=interp)
        try:
            self.q.put((panel, idx, Image.fromarray(rgb)), timeout=0.2)
        except queue.Full:
            pass

    # ---- 표시 폴러 (메인 스레드) ----
    def poll_queue(self):
        try:
            while True:
                panel, idx, pil = self.q.get_nowait()
                photo = ImageTk.PhotoImage(pil)
                panel.show(idx, photo)
        except queue.Empty:
            pass
        if not self.stop_flag:
            self.tk.after(15, self.poll_queue)

    def on_close(self):
        self.stop_flag = True
        self.tk.after(50, self.tk.destroy)

    def run(self):
        self.tk.mainloop()


def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else None
    if not root_dir:
        r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)
        init = DEFAULT_ROOT if os.path.isdir(DEFAULT_ROOT) else os.getcwd()
        root_dir = filedialog.askdirectory(
            title='조건 폴더들이 든 상위 폴더 선택', initialdir=init)
        r.destroy()
    if not root_dir:
        print('폴더를 선택하지 않았습니다.')
        return
    App(root_dir).run()


if __name__ == '__main__':
    main()
