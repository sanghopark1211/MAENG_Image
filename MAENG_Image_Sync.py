# -*- coding: utf-8 -*-
"""
============================================================
 MAENG_Image_Sync — 초고속 카메라 조건 비교 앱
------------------------------------------------------------
 Phantom .cine 영상 두 개를 나란히 재생·비교한다.

 [선택 방식]  파일명을 뒤지지 않고 '조건 버튼'으로 고른다
    각도(0~180°) → 절삭속도(m/min) → 실험 번호(반복)
    폴더 규칙   : <상위폴더>/각도_깊이_속도/*.cine
    실험 번호   : 파일명 끝 _N.cine 의 N
    (규칙에 안 맞는 파일은 아래 '직접 선택' 목록에서 선택)

 [재생]  패널별 재생/정지/1프레임/슬라이더 완전 독립
         + 하단의 동시 재생/정지/1프레임/처음
         컬러 센서(Bayer) 자동 복원 (WB+게인+EA 디모자이크+감마 LUT)
         밝기 게인(자동) · 노이즈 감소 옵션, 흑백도 디모자이크 후 변환

 · 의존: Python 3.8+, Pillow, numpy, opencv-python, pycine
============================================================
"""
import os
import re
import sys
import math
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog

import numpy as np
import cv2
from PIL import Image, ImageTk

# ===== 설정 =====================================================
APP_NAME     = 'MAENG_Image_Sync'
# 데이터 루트: 외장 드라이브 문자가 PC마다 달라(F:, I: …) 실행 시 드라이브를 훑어 찾는다
DEFAULT_ROOT_REL = r"실시간 재료거동\박상호\1) 압연재 조건별 비교 데이터\초고속 카메라 이미지 데이터"

def find_default_root():
    for d in 'CDEFGHIJKLMNOPQRSTUVWXYZ':
        p = f'{d}:' + os.sep + DEFAULT_ROOT_REL
        if os.path.isdir(p):
            return p
    return ''

DEFAULT_ROOT = find_default_root()
PANEL_SIZE   = 560        # 각 패널 '초기' 표시 크기 [px] — 창을 키우면 자동 확대
PANEL_MIN    = 320        # 최소 패널 크기
FPS_LIST     = [60, 30, 15, 5, 2, 1]
DEFAULT_FPS  = 30
DEFAULT_GAIN = 4.0        # 밝기 게인 초기값 — 이 데이터셋은 12-bit 중 상위 ~21% 만 쓰여
                          #   어둡게 보이므로 LUT 에서 증폭. [자동] 버튼이 현재 프레임에 맞춤.
HOT_THR      = 120        # 핫픽셀 판정 문턱 [DN] — 같은 색 이웃 중앙값보다
                          #   이만큼 밝으면 '반짝이'로 보고 이웃값으로 치환.
                          #   남으면 80 으로 낮추고, 실제 스펙클까지 지워지면 200 으로.
UM_PER_PX    = 2.20       # ★ 1픽셀당 실제 길이 [um/px] — 모든 측정의 환산 계수.
                          #   근거(실측): 1 m/min 영상에서 공작물 이동 3.79 px/frame,
                          #   2000fps 이므로 8.333 um/frame -> 8.333/3.79 = 2.20 um/px
                          #   (위상상관, 유리판 정지패턴 제거 후. 100um 깊이 = 45.5px)

# ----- 다크 테마 색 -----
C_BG     = '#14161a'      # 창 배경
C_PANEL  = '#1e2128'      # 패널 배경
C_LINE   = '#2e333d'      # 경계선
C_FG     = '#e8eaf0'      # 글자
C_SUB    = '#9aa3b2'      # 보조 글자
C_ACC_L  = '#4da3ff'      # 왼쪽 패널 강조 (파랑)
C_ACC_R  = '#ff7a4d'      # 오른쪽 패널 강조 (주황)
C_BTN    = '#262b34'      # 버튼 기본
C_BTN_HI = '#303743'      # 버튼 hover
C_DIS    = '#5a6170'      # 비활성 글자
# ================================================================


# ---------------- cine 판독 ----------------
class CineReader:
    """cine 1개. 순차 재생은 열린 제너레이터(~4ms), 점프는 재오픈."""

    def __init__(self, path):
        from pycine.file import read_header
        self.path = path
        h = read_header(path)
        self.n = int(h['cinefileheader'].ImageCount)
        self.setup = h['setup']
        self.fps = float(self.setup.FrameRate or self.setup.FrameRate16 or 0)   # 촬영 fps (시각 표시용)
        self.cfa = int(self.setup.CFA)
        self.bpp = None
        self._gen = None
        self._pos = -1
        self._lut = None
        self.gain = 1.0               # 밝기 게인 (LUT 에 반영)
        self.last_raw = None          # 마지막으로 읽은 raw 프레임 (자동 게인 계산용)

    def _open_at(self, idx):
        from pycine.raw import read_frames
        gen, setup, bpp = read_frames(self.path, start_frame=idx)
        self._gen, self._pos, self.bpp = gen, idx, int(bpp)

    def read(self, idx):
        idx = max(0, min(self.n - 1, idx))
        if self._gen is None or idx != self._pos:
            self._open_at(idx)
        try:
            fr = next(self._gen)
            self._pos = idx + 1
            self.last_raw = fr
            return fr
        except StopIteration:
            self._gen = None
            return None

    def _build_luts(self):
        maxv = (1 << self.bpp) - 1
        x = np.linspace(0.0, 1.0, maxv + 1) * self.gain
        gamma = float(getattr(self.setup, 'fGamma', 2.2)) or 2.2
        g = lambda v: np.clip(255.0 * np.power(np.clip(v, 0, 1), 1.0 / gamma),
                              0, 255).astype(np.uint8)
        self._lut = g(x)
        try:
            wb = self.setup.WBGain[0]
            rG, bG = float(wb.R), float(wb.B)
        except Exception:
            rG, bG = 1.0, 1.0
        self._lutR = g(x * rG)          # WB 를 감마 LUT 에 합침 (정수 인덱싱만)
        self._lutB = g(x * bG)

    def set_gain(self, gain):
        """밝기 게인 — LUT 에 곱해 두므로 프레임당 비용 0. bpp 를 알기 전이면 값만 저장."""
        self.gain = float(gain)
        if self.bpp is not None:
            self._build_luts()

    @staticmethod
    def _despeckle(raw):
        """핫픽셀(반짝이) 제거 — 디모자이크 '전' raw Bayer 에서.
        같은 색 이웃(±2px 상하좌우)의 중앙값보다 HOT_THR 이상 밝은
        '고립된 한 픽셀'만 이웃값으로 치환한다. 진짜 스펙클 하이라이트는
        여러 픽셀 덩어리라 이웃도 밝으므로 건드리지 않는다."""
        pad = np.pad(raw, 2, mode='edge')
        nb = np.stack([pad[:-4, 2:-2], pad[4:, 2:-2],
                       pad[2:-2, :-4], pad[2:-2, 4:]])   # 같은 Bayer 색 이웃 4개
        med = np.median(nb, axis=0)
        hot = raw.astype(np.int32) - med > HOT_THR
        out = raw.copy()
        out[hot] = med[hot].astype(raw.dtype)
        return out

    @staticmethod
    def _denoise(rgb, color):
        """노이즈 감소 (~8 ms) — 휘도는 에지 보존 bilateral, 색차는 가우시안(컬러 얼룩 제거)"""
        if not color:
            g = cv2.bilateralFilter(rgb[..., 0], 5, 25, 5)
            return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        y = cv2.bilateralFilter(ycc[..., 0], 5, 25, 5)
        c = cv2.GaussianBlur(ycc[..., 1:], (7, 7), 0)
        return cv2.cvtColor(np.dstack([y, c]), cv2.COLOR_YCrCb2RGB)

    def to_rgb8(self, fr, color=True, despeckle=True, denoise=True):
        if self.bpp is None:
            self.bpp = 12
        if self._lut is None:
            self._build_luts()
        maxv = (1 << self.bpp) - 1
        if despeckle:
            fr = self._despeckle(fr)
        raw16 = np.minimum(fr, maxv).astype(np.uint16, copy=False)
        if self.cfa == 0:                                   # 모노 센서
            out = cv2.cvtColor(self._lut[raw16], cv2.COLOR_GRAY2RGB)
            color = False
        else:
            # 흑백 표시도 디모자이크를 거친다 — raw Bayer 에 바로 LUT 를 씌우면
            # R/G/B 픽셀 감도 차이(WB 1.8배)가 격자 무늬 노이즈로 보였음
            rgb = cv2.cvtColor(raw16, cv2.COLOR_BAYER_GB2RGB_EA)   # 에지 보존 디모자이크
            out = np.empty(rgb.shape, np.uint8)
            out[..., 0] = self._lutR[rgb[..., 0]]
            out[..., 1] = self._lut[rgb[..., 1]]
            out[..., 2] = self._lutB[rgb[..., 2]]
            if not color:
                out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY),
                                   cv2.COLOR_GRAY2RGB)
        if denoise:
            out = self._denoise(out, color)
        return out


# ---------------- 조건 인덱스 ----------------
COND_PAT = re.compile(r'^\s*(-?\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\s*$')
REP_PAT  = re.compile(r'_(\d+)\.cine$', re.I)


def scan_root(root):
    """반환:
       cond[(angle,speed)][rep] = path      … 조건 버튼용
       extras[표시이름] = path              … 규칙 밖 파일 (직접 선택용)
    """
    cond, extras = {}, {}
    for dirpath, dirnames, filenames in os.walk(root):
        folder = os.path.basename(dirpath)
        m = COND_PAT.match(folder)
        for f in filenames:
            if os.path.splitext(f)[1].lower() != '.cine':
                continue
            path = os.path.join(dirpath, f)
            rep_m = REP_PAT.search(f)
            if m and rep_m:
                key = (float(m.group(1)), float(m.group(3)))
                cond.setdefault(key, {})[int(rep_m.group(1))] = path
            else:
                rel = os.path.relpath(path, root).replace('\\', '/')
                extras[rel] = path
    return cond, extras


# ---------------- 꾹 누르면 연속 실행되는 버튼 ----------------
def bind_hold(btn, fn, first_delay=350, interval=60):
    """버튼을 누르고 있으면 fn 이 연속 호출된다 (한 번 클릭 = 한 번 실행)."""
    state = {'job': None}

    def fire_repeat():
        fn()
        state['job'] = btn.after(interval, fire_repeat)

    def on_press(_):
        fn()                                     # 즉시 1회
        state['job'] = btn.after(first_delay, fire_repeat)

    def on_release(_):
        if state['job'] is not None:
            btn.after_cancel(state['job'])
            state['job'] = None

    btn.config(command=lambda: None)             # 기본 command 중복 방지
    btn.bind('<ButtonPress-1>', on_press)
    btn.bind('<ButtonRelease-1>', on_release)
    btn.bind('<Leave>', on_release)


# ---------------- 세그먼트 버튼 줄 ----------------
class SegRow:
    """[라벨] (버튼)(버튼)... — 하나만 선택되는 토글 버튼 줄"""

    def __init__(self, parent, label, accent, on_change, btn_w=3):
        self.accent, self.on_change, self.btn_w = accent, on_change, btn_w
        self.value = None
        self.btns = {}
        self.row = tk.Frame(parent, bg=C_PANEL)
        tk.Label(self.row, text=label, width=5, anchor='w',
                 bg=C_PANEL, fg=C_SUB,
                 font=('Malgun Gothic', 9)).pack(side='left')
        self.holder = tk.Frame(self.row, bg=C_PANEL)
        self.holder.pack(side='left', fill='x')

    def set_options(self, values, fmt=str, enabled=None):
        for b in self.btns.values():
            b.destroy()
        self.btns = {}
        for v in values:
            ok = (enabled is None) or (v in enabled)
            b = tk.Button(self.holder, text=fmt(v), width=self.btn_w,
                          font=('Malgun Gothic', 9, 'bold'),
                          bd=0, relief='flat', cursor='hand2',
                          bg=C_BTN, fg=C_FG if ok else C_DIS,
                          activebackground=C_BTN_HI, activeforeground=C_FG,
                          state='normal' if ok else 'disabled',
                          command=lambda vv=v: self.select(vv))
            b.pack(side='left', padx=1, pady=1)
            self.btns[v] = b
        if self.value not in self.btns or (enabled is not None
                                           and self.value not in enabled):
            self.value = None
        self._paint()

    def select(self, v, fire=True):
        self.value = v
        self._paint()
        if fire:
            self.on_change()

    def _paint(self):
        for v, b in self.btns.items():
            if v == self.value:
                b.config(bg=self.accent, fg='#ffffff')
            elif b['state'] == 'normal':
                b.config(bg=C_BTN, fg=C_FG)


# ---------------- 패널 ----------------
class Panel:
    def __init__(self, app, parent, side, accent):
        self.app, self.side, self.accent = app, side, accent
        self.reader = None
        self.i = 0
        self.playing = False
        self.want_seek = None
        self.t_last = None            # 실시간 재생 기준 시각 (프레임 스킵용)
        self._sld_guard = False
        self.t_dec = 0                # 시각 표시 소수 자릿수 (촬영 fps 로 결정)

        f = tk.Frame(parent, bg=C_PANEL, highlightthickness=1,
                     highlightbackground=C_LINE)
        self.frame = f
        pad = dict(padx=8)

        # --- 조건 버튼 3줄 ---
        self.rowA = SegRow(f, '각도',  accent, self.on_cond, btn_w=3)
        self.rowS = SegRow(f, '속도',  accent, self.on_cond, btn_w=4)
        self.rowR = SegRow(f, '실험',  accent, self.on_cond, btn_w=3)
        self.rowA.row.pack(fill='x', pady=(8, 0), **pad)
        self.rowS.row.pack(fill='x', pady=(2, 0), **pad)
        self.rowR.row.pack(fill='x', pady=(2, 0), **pad)

        # --- 선택된 파일 표시 + 직접 선택 ---
        sub = tk.Frame(f, bg=C_PANEL)
        sub.pack(fill='x', pady=(4, 0), **pad)
        self.lbl_file = tk.Label(sub, text='(조건을 선택하세요)', anchor='w',
                                 bg=C_PANEL, fg=C_SUB,
                                 font=('Malgun Gothic', 8))
        self.lbl_file.pack(side='left', fill='x', expand=True)
        self.cmb = ttk.Combobox(sub, state='readonly', width=18)
        self.cmb.set('직접 선택 ▾')
        self.cmb.pack(side='right')
        self.cmb.bind('<<ComboboxSelected>>', self.on_direct)

        # --- 화면 ---
        ps = app.panel_size
        self.canvas = tk.Canvas(f, width=ps, height=ps,
                                bg='black', highlightthickness=2,
                                highlightbackground=accent)
        self.canvas.pack(pady=6, **pad)
        self._imgid = self.canvas.create_image(ps // 2, ps // 2)
        self._photo = None

        # --- 📐 각도 측정 (앱의 '각도 측정' 체크 시 드래그로 직선) ---
        self.meas_items = []          # 확정된 선/텍스트 캔버스 아이템
        self._drag = None             # (각도) 드래그 중 임시
        self._poly = None             # (BUE) 진행 중 다각형 {pts, items, preview}
        self.disp_scale = 1.0         # 원본 px → 화면 px 배율 (decode 시 갱신)
        self.disp_off = (0, 0)        # 화면 안 이미지 좌상단 오프셋
        self.canvas.bind('<ButtonPress-1>', self.meas_press)
        self.canvas.bind('<B1-Motion>', self.meas_drag)
        self.canvas.bind('<Motion>', self.meas_drag)
        self.canvas.bind('<ButtonRelease-1>', self.meas_release)
        self.canvas.bind('<ButtonPress-3>', self.poly_close)   # 우클릭 = 다각형 닫기

        # --- 재생 줄 ---
        row = tk.Frame(f, bg=C_PANEL)
        row.pack(fill='x', pady=(0, 8), **pad)
        mk = lambda t, c, w=3: tk.Button(
            row, text=t, width=w, command=c, bd=0, relief='flat',
            cursor='hand2', bg=C_BTN, fg=C_FG,
            activebackground=C_BTN_HI, activeforeground=C_FG,
            font=('Malgun Gothic', 10, 'bold'))
        self.btn_play = mk('▶', self.toggle_play)
        self.btn_play.pack(side='left')
        b_prev = mk('−1', None); b_prev.pack(side='left', padx=2)
        b_next = mk('+1', None); b_next.pack(side='left')
        bind_hold(b_prev, lambda: self.step(-1))   # 꾹 누르면 연속 이동
        bind_hold(b_next, lambda: self.step(+1))
        self.sld = ttk.Scale(row, from_=0, to=0, command=self.on_slider,
                             style=f'{side}.Horizontal.TScale')
        self.sld.pack(side='left', fill='x', expand=True, padx=8)
        self.lbl = tk.Label(row, text='- / -', width=13, anchor='e',
                            bg=C_PANEL, fg=C_SUB,
                            font=('Consolas', 10))
        self.lbl.pack(side='right')

        # ★ 프레임 번호 직접 입력 (숫자 입력 후 Enter 또는 [이동])
        self.ent = tk.Entry(row, width=7, bg=C_BTN, fg=C_FG,
                            insertbackground=C_FG, relief='flat',
                            justify='right', font=('Consolas', 10))
        self.ent.pack(side='right', padx=(6, 2))
        self.ent.bind('<Return>', self.on_goto)
        tk.Button(row, text='이동', width=4, command=self.on_goto,
                  bd=0, relief='flat', cursor='hand2', bg=C_BTN, fg=C_FG,
                  activebackground=C_BTN_HI, activeforeground=C_FG,
                  font=('Malgun Gothic', 9)).pack(side='right')
        # ★ 현재 프레임의 실제 시각 [s] = (프레임번호−1) / 촬영 fps — 슬라이더 바로 옆
        self.lbl_t = tk.Label(row, text='', width=9, anchor='e',
                              bg=C_PANEL, fg=accent,
                              font=('Consolas', 10, 'bold'))
        self.lbl_t.pack(side='right', padx=(6, 4))

    # ---- 측정 도구 (각도 / BUE 크기 — 모드로 완전 분리) ----
    @staticmethod
    def _angle_of(x0, y0, x1, y1):
        """수평선 기준 예각 [0~90°] — 전단각. 화면 y 는 아래로 증가하므로 부호 반전.
        선을 어느 방향으로 긋든, 어느 쪽으로 기울든 같은 값이 나온다."""
        ang = math.degrees(math.atan2(-(y1 - y0), (x1 - x0))) % 180.0
        return 180.0 - ang if ang > 90.0 else ang

    def _to_src(self, x, y):
        """캔버스 좌표 → 원본 이미지 픽셀 좌표"""
        ox, oy = self.disp_off
        sc = max(self.disp_scale, 1e-9)
        return (x - ox) / sc, (y - oy) / sc

    def meas_press(self, e):
        mode = self.app.var_mode.get()
        if mode in ('angle', 'cal'):
            col = '#ffe14d' if mode == 'angle' else '#5bff8a'
            line = self.canvas.create_line(e.x, e.y, e.x, e.y,
                                           fill=col, width=2)
            txt = self.canvas.create_text(e.x + 10, e.y - 12, text='',
                                          fill=col, anchor='w',
                                          font=('Consolas', 12, 'bold'))
            self._drag = (e.x, e.y, line, txt)
        elif mode == 'bue':
            self._poly_add(e.x, e.y)

    def meas_drag(self, e):
        if self._drag:
            x0, y0, line, txt = self._drag
            self.canvas.coords(line, x0, y0, e.x, e.y)
            ang = self._angle_of(x0, y0, e.x, e.y)
            # 원본 픽셀 기준 길이 + 실측 환산 (칩 두께 측정용)
            sx0, sy0 = self._to_src(x0, y0)
            sx1, sy1 = self._to_src(e.x, e.y)
            L = math.hypot(sx1 - sx0, sy1 - sy0)
            um = self.app.um_per_px
            self.canvas.itemconfig(
                txt, text=f'{ang:.1f}\u00b0  {L:.0f}px ({L*um:.0f}\u00b5m)')
            self.canvas.coords(txt, (x0 + e.x) / 2 + 12, (y0 + e.y) / 2 - 12)
        elif self._poly:
            # 마지막 꼭짓점 → 마우스 미리보기 선 + 그 변의 길이
            px, py = self._poly['pts'][-1]
            self.canvas.coords(self._poly['preview'], px, py, e.x, e.y)
            L = self._src_len(px, py, e.x, e.y)
            self.canvas.itemconfig(
                self._poly['preview_txt'],
                text=f'{L:.0f}px ({L*self.app.um_per_px:.0f}\u00b5m)')
            self.canvas.coords(self._poly['preview_txt'], e.x + 10, e.y - 6)

    def meas_release(self, e):
        if not self._drag:
            return
        x0, y0, line, txt = self._drag
        mode = self.app.var_mode.get()
        self._drag = None
        if abs(e.x - x0) < 4 and abs(e.y - y0) < 4:      # 클릭만 한 경우
            self.canvas.delete(line); self.canvas.delete(txt)
            return

        sx0, sy0 = self._to_src(x0, y0)
        sx1, sy1 = self._to_src(e.x, e.y)
        L = math.hypot(sx1 - sx0, sy1 - sy0)

        if mode == 'cal':
            # ★ 스케일 보정: 아는 길이를 그으면 um/px 를 갱신
            self.canvas.delete(line); self.canvas.delete(txt)
            real = simpledialog.askfloat(
                '스케일 보정',
                f'방금 그은 선 = {L:.1f}px\n실제 길이 [µm]?  '
                '(절삭 전 가공 깊이 = 100)',
                initialvalue=100.0, minvalue=0.1, parent=self.app.tk)
            if real and L > 1:
                self.app.set_scale(real / L)
                self.app.log_memo(
                    f'[스케일] {L:.1f}px = {real:g}um -> '
                    f'{real/L:.3f} um/px')
            return

        ang = self._angle_of(x0, y0, e.x, e.y)
        um = self.app.um_per_px
        self.meas_items += [line, txt]
        self.app.log_memo(
            f'[{self.side}] fr{self.i+1}  선 {ang:.1f}deg  '
            f'L={L:.0f}px ({L*um:.0f}um)')

    # ---- BUE 다각형: 좌클릭 = 꼭짓점 추가, 우클릭 = 닫기 ----
    #   BUE 는 꼭짓점이 뚜렷하지 않은 뭉툭한 형상이라 변마다 길이를 따로 표시한다
    #   (공구와 만나는 면의 길이, 경사면(rake)까지의 거리 등을 변 단위로 읽기 위함)
    def _src_len(self, x0, y0, x1, y1):
        """캔버스 두 점 사이 거리를 원본 픽셀 단위로"""
        sx0, sy0 = self._to_src(x0, y0)
        sx1, sy1 = self._to_src(x1, y1)
        return math.hypot(sx1 - sx0, sy1 - sy0)

    def _seg_add(self, P, x0, y0, x1, y1):
        """변 (x0,y0)-(x1,y1) 의 길이 라벨 '번호: px (µm)' 을 만든다"""
        L = self._src_len(x0, y0, x1, y1)
        k = len(P['segs']) + 1
        txt = self.canvas.create_text(
            0, 0, text=f'{k}: {L:.0f}px ({L*self.app.um_per_px:.0f}\u00b5m)',
            fill='#ffd0d0', font=('Consolas', 9, 'bold'))
        bg = self.canvas.create_rectangle(0, 0, 0, 0, fill='#000000', outline='')
        self.canvas.tag_lower(bg, txt)
        P['segs'].append((bg, txt, L, (x0, y0, x1, y1)))
        P['items'] += [bg, txt]
        self._seg_place_all(P)

    def _seg_place_all(self, P):
        """모든 변 라벨을 변 중점에서 다각형 바깥쪽(무게중심 반대편)으로 띄워 놓는다"""
        pts = P['pts']
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(y for _, y in pts) / len(pts)
        for bg, txt, _, (x0, y0, x1, y1) in P['segs']:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            n = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / n, dx / n                   # 변의 법선
            if (mx - cx) * nx + (my - cy) * ny < 0:    # 무게중심 쪽을 향하면 뒤집기
                nx, ny = -nx, -ny
            self.canvas.coords(txt, mx + nx * 16, my + ny * 16)
            self.canvas.coords(bg, *self.canvas.bbox(txt))

    def _poly_add(self, x, y):
        if self._poly is None:
            pv = self.canvas.create_line(x, y, x, y, fill='#ff5b5b',
                                         width=2, dash=(4, 3))
            pt = self.canvas.create_text(x, y, text='', fill='#ffd0d0',
                                         anchor='sw', font=('Consolas', 9, 'bold'))
            self._poly = {'pts': [], 'items': [], 'segs': [],
                          'preview': pv, 'preview_txt': pt}
        P = self._poly
        P['pts'].append((x, y))
        P['items'].append(self.canvas.create_oval(x-3, y-3, x+3, y+3,
                                                  fill='#ff5b5b', outline=''))
        if len(P['pts']) > 1:
            (x0, y0) = P['pts'][-2]
            P['items'].append(self.canvas.create_line(x0, y0, x, y,
                                                      fill='#ff5b5b', width=2))
            self._seg_add(P, x0, y0, x, y)

    def poly_close(self, _=None):
        """우클릭 → 다각형 확정: 채우기 + 높이/밑변/면적 표시"""
        P = self._poly
        if not P or len(P['pts']) < 3:
            self.poly_cancel()
            return
        pts = P['pts']
        flat = [c for xy in pts for c in xy]
        poly = self.canvas.create_polygon(*flat, fill='#ff5b5b',
                                          stipple='gray25',
                                          outline='#ff5b5b', width=2)
        self._seg_add(P, *pts[-1], *pts[0])            # 닫는 변 (마지막 → 첫 꼭짓점)
        for sbg, stxt, _, _ in P['segs']:              # 라벨을 채우기 위로
            self.canvas.tag_raise(sbg); self.canvas.tag_raise(stxt)
        # --- 원본 픽셀 좌표로 환산해 계측 ---
        sp = [self._to_src(x, y) for (x, y) in pts]
        xs = [a for a, _ in sp];  ys = [b for _, b in sp]
        w_px = max(xs) - min(xs)
        h_px = max(ys) - min(ys)
        area = 0.0                                     # shoelace
        for k in range(len(sp)):
            x0, y0 = sp[k];  x1, y1 = sp[(k + 1) % len(sp)]
            area += x0 * y1 - x1 * y0
        area = abs(area) / 2.0
        um = self.app.um_per_px
        txt = (f'H {h_px:.0f}px ({h_px*um:.0f}\u00b5m)\n'
               f'W {w_px:.0f}px ({w_px*um:.0f}\u00b5m)\n'
               f'A {area:.0f}px\u00b2 ({area*um*um/1e6:.4f}mm\u00b2)')
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(y for _, y in pts) / len(pts)
        tid = self.canvas.create_text(cx, cy, text=txt, fill='#ffffff',
                                      font=('Consolas', 10, 'bold'),
                                      justify='center')
        bg = self.canvas.create_rectangle(self.canvas.bbox(tid),
                                          fill='#000000', outline='')
        self.canvas.tag_lower(bg, tid)
        self.canvas.delete(P['preview']); self.canvas.delete(P['preview_txt'])
        self.meas_items += P['items'] + [poly, bg, tid]
        self._poly = None
        # 메모장에 자동 기록 (변 길이는 꼭짓점을 찍은 순서대로 1, 2, … 마지막이 닫는 변)
        segs = ' '.join(f'{k}={L:.0f}px({L*um:.0f}um)'
                        for k, (_, _, L, _) in enumerate(P['segs'], 1))
        self.app.log_memo(
            f'[{self.side}] {os.path.basename(self.reader.path) if self.reader else "?"} '
            f'fr{self.i+1}  BUE H={h_px:.0f}px({h_px*um:.0f}um) '
            f'W={w_px:.0f}px({w_px*um:.0f}um) A={area:.0f}px2  변: {segs}')

    def poly_cancel(self, _=None):
        if self._poly:
            for it in self._poly['items']:
                self.canvas.delete(it)
            self.canvas.delete(self._poly['preview'])
            self.canvas.delete(self._poly['preview_txt'])
            self._poly = None

    def meas_clear(self):
        for it in self.meas_items:
            self.canvas.delete(it)
        self.meas_items = []
        if self._drag:
            self.canvas.delete(self._drag[2]); self.canvas.delete(self._drag[3])
            self._drag = None
        self.poly_cancel()

    def resize_canvas(self, ps):
        self.canvas.config(width=ps, height=ps)
        self.canvas.coords(self._imgid, ps // 2, ps // 2)
        self.meas_clear()                    # 배율이 바뀌면 선 좌표가 안 맞으므로 제거
        if self.reader:
            self.want_seek = self.i          # 현재 프레임을 새 크기로 다시 그림

    # ---- 조건 버튼 ----
    def refresh_buttons(self):
        cond = self.app.cond
        angs = sorted({k[0] for k in cond})
        self.rowA.set_options(angs, fmt=lambda v: f'{v:g}')
        self._refresh_dependent()

    def _refresh_dependent(self):
        cond = self.app.cond
        a = self.rowA.value
        spds_all = sorted({k[1] for k in cond})
        spds_ok = sorted({k[1] for k in cond if a is None or k[0] == a})
        self.rowS.set_options(spds_all, fmt=lambda v: f'{v:g}',
                              enabled=set(spds_ok))
        s = self.rowS.value
        reps_all = sorted({r for v in cond.values() for r in v})
        if a is not None and s is not None and (a, s) in cond:
            reps_ok = set(cond[(a, s)])
        else:
            reps_ok = set()
        self.rowR.set_options(reps_all, fmt=str,
                              enabled=reps_ok if (a is not None and s is not None)
                              else set(reps_all))

    def on_cond(self):
        self._refresh_dependent()
        a, s, r = self.rowA.value, self.rowS.value, self.rowR.value
        if a is None or s is None or r is None:
            return
        path = self.app.cond.get((a, s), {}).get(r)
        if path:
            self.load(path)

    def on_direct(self, _=None):
        name = self.cmb.get()
        path = self.app.extras.get(name)
        if path:
            self.rowA.value = self.rowS.value = self.rowR.value = None
            self.rowA._paint(); self.rowS._paint(); self.rowR._paint()
            self.load(path)

    def load(self, path):
        self.playing = False
        self.app.update_buttons()
        try:
            self.reader = CineReader(path)
        except Exception as e:
            self.lbl_file.config(text=f'열기 실패: {e}')
            return
        self.reader.set_gain(self.app.gain_value)
        self.i = 0
        self.sld.config(to=self.reader.n - 1)
        fps = self.reader.fps
        # 소수 자릿수: 1프레임 차이가 보이도록 (2000fps → 0.0005s → 4자리)
        self.t_dec = max(1, min(6, math.ceil(math.log10(fps)))) if fps > 0 else 0
        self.lbl_file.config(
            text=f'{os.path.basename(path)}   ({self.reader.n}프레임'
                 + (f' · {fps:g}fps' if fps > 0 else '') + ')')
        self.lbl_t.config(text='')
        self.want_seek = 0

    # ---- 재생 조작 ----
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
        self.want_seek = max(0, min(self.reader.n - 1, self.i + d))

    def on_goto(self, _=None):
        if not self.reader:
            return
        try:
            idx = int(self.ent.get().strip()) - 1     # 표시는 1-기준
        except ValueError:
            return
        self.playing = False
        self.app.update_buttons()
        self.want_seek = max(0, min(self.reader.n - 1, idx))

    def on_slider(self, v):
        if self._sld_guard or not self.reader:
            return
        self.playing = False
        self.app.update_buttons()
        self.want_seek = int(float(v))

    def show(self, idx, photo):
        self.i = idx
        self._photo = photo
        self.canvas.itemconfig(self._imgid, image=photo)
        self.lbl.config(text=f'{idx + 1}/{self.reader.n}')
        if self.reader.fps > 0:
            self.lbl_t.config(text=f'{idx / self.reader.fps:.{self.t_dec}f}s')
        self._sld_guard = True
        self.sld.set(idx)
        self._sld_guard = False


# ---------------- 앱 ----------------
class App:
    def __init__(self, root_dir):
        self.tk = tk.Tk()
        self.tk.title(APP_NAME)
        self.tk.configure(bg=C_BG)
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure('TCombobox', fieldbackground=C_BTN, background=C_BTN,
                        foreground=C_FG, arrowcolor=C_FG)
        for side, acc in (('L', C_ACC_L), ('R', C_ACC_R)):
            style.configure(f'{side}.Horizontal.TScale', troughcolor=C_BTN,
                            background=acc)

        self.cond, self.extras = {}, {}
        self.q = queue.Queue(maxsize=8)
        self.stop_flag = False
        self.panel_size = PANEL_SIZE
        self._resize_job = None

        # ---- 헤더 ----
        top = tk.Frame(self.tk, bg=C_BG)
        top.pack(fill='x', padx=10, pady=(8, 4))
        tk.Label(top, text=APP_NAME, bg=C_BG, fg=C_FG,
                 font=('Malgun Gothic', 13, 'bold')).pack(side='left')
        self.lbl_root = tk.Label(top, text='', bg=C_BG, fg=C_SUB,
                                 font=('Malgun Gothic', 9))
        self.lbl_root.pack(side='left', padx=12)
        mkb = lambda t, c: tk.Button(top, text=t, command=c, bd=0,
                                     relief='flat', cursor='hand2',
                                     bg=C_BTN, fg=C_FG, padx=10, pady=3,
                                     activebackground=C_BTN_HI,
                                     activeforeground=C_FG,
                                     font=('Malgun Gothic', 9))
        mkb('폴더 다시 선택', self.pick_root).pack(side='right', padx=2)
        mkb('🔄 새로고침', self.refresh).pack(side='right', padx=2)

        # ---- 본문: [패널 2개 | 우측 컨트롤] / 하단 메모 ----
        body = tk.Frame(self.tk, bg=C_BG)
        body.pack(fill='both', expand=True, padx=10)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        mid = tk.Frame(body, bg=C_BG)
        mid.grid(row=0, column=0, sticky='nw')
        self.L = Panel(self, mid, 'L', C_ACC_L)
        self.R = Panel(self, mid, 'R', C_ACC_R)
        self.L.frame.grid(row=0, column=0, padx=(0, 5))
        self.R.frame.grid(row=0, column=1, padx=(5, 0))

        # ===== 우측 컨트롤 컬럼 =====
        side = tk.Frame(body, bg=C_PANEL, highlightthickness=1,
                        highlightbackground=C_LINE)
        side.grid(row=0, column=1, sticky='ns', padx=(8, 0))

        def sec(title):
            tk.Label(side, text=title, bg=C_PANEL, fg=C_SUB, anchor='w',
                     font=('Malgun Gothic', 9, 'bold')
                     ).pack(fill='x', padx=10, pady=(10, 2))

        def sbtn(t, c=None, big=False):
            b = tk.Button(side, text=t, command=c, bd=0, relief='flat',
                          cursor='hand2', bg=C_BTN, fg=C_FG, pady=6,
                          activebackground=C_BTN_HI, activeforeground=C_FG,
                          font=('Malgun Gothic', 10, 'bold' if big else 'normal'))
            b.pack(fill='x', padx=10, pady=2)
            return b

        # --- 그림판 (측정) ---
        sec('🖌 그림판')
        self.var_mode = tk.StringVar(value='off')
        for txt, val in (('끄기', 'off'), ('📐 각도 측정', 'angle'),
                         ('△ BUE 크기 측정', 'bue'),
                         ('⚖ 스케일 보정', 'cal')):
            tk.Radiobutton(side, text=txt, value=val, variable=self.var_mode,
                           bg=C_PANEL, fg=C_FG, selectcolor=C_BTN,
                           activebackground=C_PANEL, activeforeground=C_FG,
                           anchor='w', font=('Malgun Gothic', 9)
                           ).pack(fill='x', padx=14)
        tk.Label(side, text='BUE: 좌클릭=꼭짓점, 우클릭=완성\n'
                 '     변마다 길이(px/\u00b5m) 표시\n'
                 '보정: 절삭깊이(100µm) 구간을 드래그',
                 bg=C_PANEL, fg=C_SUB, anchor='w', justify='left',
                 font=('Malgun Gothic', 8)).pack(fill='x', padx=14)
        self.um_per_px = UM_PER_PX
        self.lbl_scale = tk.Label(side, text=f'스케일 {UM_PER_PX:.3f} \u00b5m/px',
                                  bg=C_PANEL, fg='#5bff8a', anchor='w',
                                  font=('Consolas', 9, 'bold'))
        self.lbl_scale.pack(fill='x', padx=14)
        sbtn('선 지우기', lambda: (self.L.meas_clear(), self.R.meas_clear()))

        # --- 동시 재생 ---
        sec('▶ 동시 컨트롤')
        self.btn_all = sbtn('▶▶ 동시 재생', self.play_all, True)
        sbtn('⏸ 동시 정지', self.stop_all, True)
        b_ap = sbtn('◀ 1프레임');  b_an = sbtn('1프레임 ▶')
        bind_hold(b_ap, lambda: self.step_all(-1))
        bind_hold(b_an, lambda: self.step_all(+1))
        sbtn('⟲ 처음', self.home_all)

        # --- 속도 ---
        sec('속도')
        self.fps_value = float(DEFAULT_FPS)
        self.sld_fps = ttk.Scale(side, from_=1, to=1000, length=150,
                                 value=DEFAULT_FPS, command=self.on_fps)
        self.sld_fps.pack(fill='x', padx=12)
        self.lbl_fps = tk.Label(side, text=f'{DEFAULT_FPS:.0f} fps',
                                bg=C_PANEL, fg=C_FG,
                                font=('Consolas', 11, 'bold'))
        self.lbl_fps.pack()

        # --- 밝기 (게인) — 어두운 촬영본을 LUT 단계에서 증폭 (프레임당 비용 0) ---
        sec('밝기')
        self.gain_value = float(DEFAULT_GAIN)
        self._gain_job = None
        self._gain_guard = False
        self.sld_gain = ttk.Scale(side, from_=1, to=8, length=150,
                                  value=DEFAULT_GAIN, command=self.on_gain)
        self.sld_gain.pack(fill='x', padx=12)
        grow = tk.Frame(side, bg=C_PANEL)
        grow.pack(fill='x', padx=10)
        self.lbl_gain = tk.Label(grow, text=f'x{DEFAULT_GAIN:.1f}',
                                 bg=C_PANEL, fg=C_FG,
                                 font=('Consolas', 11, 'bold'))
        self.lbl_gain.pack(side='left', expand=True)
        tk.Button(grow, text='자동', command=self.auto_gain, bd=0, relief='flat',
                  cursor='hand2', bg=C_BTN, fg=C_FG, padx=10, pady=2,
                  activebackground=C_BTN_HI, activeforeground=C_FG,
                  font=('Malgun Gothic', 9)).pack(side='right')

        # --- 옵션 ---
        sec('옵션')
        self.var_loop = tk.BooleanVar(value=True)
        self.var_pix = tk.BooleanVar(value=False)
        self.var_color = tk.BooleanVar(value=True)
        self.var_despk = tk.BooleanVar(value=True)
        self.var_denoise = tk.BooleanVar(value=True)
        for txt, var in (('루프', self.var_loop), ('픽셀 선명', self.var_pix),
                         ('컬러', self.var_color),
                         ('반짝이 제거', self.var_despk),
                         ('노이즈 감소', self.var_denoise)):
            tk.Checkbutton(side, text=txt, variable=var, bg=C_PANEL, fg=C_FG,
                           selectcolor=C_BTN, activebackground=C_PANEL,
                           activeforeground=C_FG, anchor='w',
                           command=self.redraw_all,       # 토글 즉시 현재 프레임 갱신
                           font=('Malgun Gothic', 9)).pack(fill='x', padx=14)

        # ===== 메모장: 우측 컬럼 하단 (남는 공간 전부) =====
        sec('📝 메모 (자동 저장)')
        self.memo = tk.Text(side, width=26, bg='#12151a', fg=C_FG,
                            insertbackground=C_FG, relief='flat',
                            font=('Malgun Gothic', 9), wrap='word')
        self.memo.pack(fill='both', expand=True, padx=10, pady=(2, 8))
        tk.Label(side, text='Space 동시재생 · ←→ 1프레임 · Home 처음',
                 bg=C_PANEL, fg=C_SUB, font=('Malgun Gothic', 8)
                 ).pack(side='bottom', pady=(0, 6))
        self.memo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'memo.txt')
        try:
            if os.path.exists(self.memo_path):
                with open(self.memo_path, encoding='utf-8') as fmemo:
                    self.memo.insert('1.0', fmemo.read())
        except Exception:
            pass

        # ---- 단축키/종료 ----
        self.tk.bind('<space>', lambda e: self.toggle_all())
        self.tk.bind('<Right>', lambda e: self.step_all(+1))
        self.tk.bind('<Left>', lambda e: self.step_all(-1))
        self.tk.bind('<Home>', lambda e: self.home_all())
        self.tk.protocol('WM_DELETE_WINDOW', self.on_close)
        # 창 크기 변경 → 패널 자동 확대/축소 (전체화면 대응)
        self.tk.bind('<Configure>', self.on_resize)

        # ---- 데이터/워커 ----
        self.set_root(root_dir)
        threading.Thread(target=self.decode_loop, daemon=True).start()
        self.tk.after(15, self.poll_queue)

    # ---- 데이터 ----
    def set_root(self, root_dir):
        self.root_dir = root_dir
        self.cond, self.extras = scan_root(root_dir) if root_dir else ({}, {})
        nfile = sum(len(v) for v in self.cond.values()) + len(self.extras)
        self.lbl_root.config(
            text=f'{root_dir}   (조건 {len(self.cond)}개 · 파일 {nfile}개)')
        for p in (self.L, self.R):
            p.refresh_buttons()
            p.cmb.config(values=sorted(self.extras))
            p.cmb.set('직접 선택 ▾')

    def refresh(self):
        """디스크 재스캔 — 오늘 추가한 실험(_6, _7 ...)이 즉시 버튼에 반영된다.
           현재 선택(각도/속도/실험)과 재생 상태는 유지."""
        keep = []
        for p in (self.L, self.R):
            keep.append((p.rowA.value, p.rowS.value, p.rowR.value))
        self.cond, self.extras = scan_root(self.root_dir)
        nfile = sum(len(v) for v in self.cond.values()) + len(self.extras)
        self.lbl_root.config(
            text=f'{self.root_dir}   (조건 {len(self.cond)}개 · 파일 {nfile}개)')
        for p, (a, sp, r) in zip((self.L, self.R), keep):
            p.refresh_buttons()
            p.cmb.config(values=sorted(self.extras))
            if a is not None and a in p.rowA.btns:
                p.rowA.select(a, fire=False)
                p._refresh_dependent()
            if sp is not None and sp in p.rowS.btns:
                p.rowS.select(sp, fire=False)
                p._refresh_dependent()
            if r is not None and r in p.rowR.btns \
                    and p.rowR.btns[r]['state'] == 'normal':
                p.rowR.select(r, fire=False)

    def pick_root(self):
        d = filedialog.askdirectory(title='조건 폴더들이 든 상위 폴더 선택',
                                    initialdir=self.root_dir or DEFAULT_ROOT)
        if d:
            self.set_root(d)

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
                p.want_seek = max(0, min(p.reader.n - 1, p.i + d))

    def home_all(self):
        self.stop_all()
        for p in (self.L, self.R):
            if p.reader:
                p.want_seek = 0

    def on_resize(self, e):
        if e.widget is not self.tk:
            return
        if self._resize_job:
            self.tk.after_cancel(self._resize_job)
        self._resize_job = self.tk.after(150, self._apply_resize)   # 디바운스

    def _apply_resize(self):
        self._resize_job = None
        w = self.tk.winfo_width()
        h = self.tk.winfo_height()
        # 패널 하나가 쓸 수 있는 공간: 가로는 절반, 세로는 조건버튼/컨트롤 제외
        ps = min((w - 300) // 2, h - 240)    # 우측 컬럼(메모 포함)만 감안 — 화면 최대화
        ps = max(PANEL_MIN, ps)
        if abs(ps - self.panel_size) < 12:
            return
        self.panel_size = ps
        for p in (self.L, self.R):
            p.resize_canvas(ps)

    def set_scale(self, um_per_px):
        self.um_per_px = float(um_per_px)
        self.lbl_scale.config(
            text=f'스케일 {self.um_per_px:.3f} µm/px')

    def log_memo(self, line):
        self.memo.insert('end', line + '\n')
        self.memo.see('end')

    def save_memo(self):
        try:
            with open(self.memo_path, 'w', encoding='utf-8') as fmemo:
                fmemo.write(self.memo.get('1.0', 'end-1c'))
        except Exception:
            pass

    def on_fps(self, v):
        self.fps_value = max(1.0, float(v))
        self.lbl_fps.config(text=f'{self.fps_value:.0f} fps')

    # ---- 밝기 ----
    def on_gain(self, v):
        if self._gain_guard:                  # auto_gain 이 슬라이더 위치만 맞출 때
            return
        self.gain_value = max(1.0, float(v))
        self.lbl_gain.config(text=f'x{self.gain_value:.1f}')
        if self._gain_job:
            self.tk.after_cancel(self._gain_job)
        self._gain_job = self.tk.after(80, self._apply_gain)     # 드래그 디바운스

    def _apply_gain(self):
        self._gain_job = None
        for p in (self.L, self.R):
            if p.reader:
                p.reader.set_gain(self.gain_value)
        self.redraw_all()

    def auto_gain(self):
        """패널별로 현재 프레임의 상위 0.3% 가 흰색이 되도록 게인을 맞춘다 (좌우 따로).
           슬라이더를 직접 움직이면 다시 양쪽 공통 값이 된다."""
        gains = []
        for p in (self.L, self.R):
            r = p.reader
            if r and r.last_raw is not None and r.bpp:
                raw = r._despeckle(r.last_raw)           # 핫픽셀이 상위 0.3% 를 끌어올리므로 제외
                p997 = float(np.percentile(raw, 99.7))
                g = min(8.0, max(1.0, ((1 << r.bpp) - 1) / max(p997, 1.0)))
                r.set_gain(g)
                gains.append(g)
        if not gains:
            return
        self.gain_value = gains[0]
        self._gain_guard = True                  # 슬라이더 위치만 맞추고 재적용은 안 함
        self.sld_gain.set(gains[0])
        self._gain_guard = False
        self.lbl_gain.config(text=' | '.join(f'x{g:.1f}' for g in gains))
        self.redraw_all()

    def redraw_all(self):
        """옵션/게인이 바뀌면 멈춰 있는 현재 프레임을 다시 그린다"""
        for p in (self.L, self.R):
            if p.reader:
                p.want_seek = p.i

    def update_buttons(self):
        self.L.btn_play.config(text='⏸' if self.L.playing else '▶')
        self.R.btn_play.config(text='⏸' if self.R.playing else '▶')

    # ---- 디코드 워커 ----
    def decode_loop(self):
        next_t = time.time()
        while not self.stop_flag:
            fps = self.fps_value
            period = 1.0 / fps
            now = time.time()
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                self._serve_seeks()
                continue
            next_t = max(next_t + period, now - period)

            loop = self.var_loop.get()
            for p in (self.L, self.R):
                if p.want_seek is not None:
                    idx = p.want_seek
                    p.want_seek = None
                    p.t_last = None
                    self._decode_show(p, idx)
                elif p.playing and p.reader:
                    # ★ 실시간 재생: 경과 시간 x fps 만큼 전진.
                    #   설정 fps 가 디코딩 한계(~30/s)보다 높으면 중간 프레임을
                    #   건너뛰어 '실제 시간 비율'을 지킨다 (1000fps 도 유효).
                    now2 = time.time()
                    if p.t_last is None:
                        adv = 1
                    else:
                        adv = int((now2 - p.t_last) * fps)
                        adv = max(1, min(adv, int(fps)))   # 폭주 방지 (최대 1초분)
                    p.t_last = now2
                    nxt = p.i + adv
                    if nxt >= p.reader.n:
                        if loop:
                            nxt %= p.reader.n
                        else:
                            nxt = p.reader.n - 1
                            p.playing = False
                            if nxt == p.i:
                                continue
                    self._decode_show(p, nxt)
                else:
                    p.t_last = None

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
        try:
            fr = r.read(idx)
        except Exception:
            return
        if fr is None:
            return
        rgb = r.to_rgb8(fr, color=self.var_color.get(),
                        despeckle=self.var_despk.get(),
                        denoise=self.var_denoise.get())
        interp = cv2.INTER_NEAREST if self.var_pix.get() else cv2.INTER_LANCZOS4
        ps = self.panel_size
        h, w = rgb.shape[:2]
        scale = min(ps / w, ps / h)
        dw, dh = int(w * scale), int(h * scale)
        rgb = cv2.resize(rgb, (dw, dh), interpolation=interp)
        panel.disp_scale = scale                       # 측정 환산용
        panel.disp_off = ((ps - dw) // 2, (ps - dh) // 2)
        try:
            self.q.put((panel, idx, Image.fromarray(rgb)), timeout=0.2)
        except queue.Full:
            pass

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
        self.save_memo()
        self.stop_flag = True
        self.tk.after(60, self.tk.destroy)

    def run(self):
        self.tk.mainloop()


def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else None
    if not root_dir:
        if os.path.isdir(DEFAULT_ROOT):
            root_dir = DEFAULT_ROOT          # 기본 경로가 있으면 바로 시작
        else:
            r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)
            root_dir = filedialog.askdirectory(
                title='조건 폴더들이 든 상위 폴더 선택', initialdir=os.getcwd())
            r.destroy()
    if not root_dir:
        print('폴더를 선택하지 않았습니다.')
        return
    App(root_dir).run()


if __name__ == '__main__':
    main()
