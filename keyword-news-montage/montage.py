#!/usr/bin/env python3
# 키워드 신문 몽타주 렌더러 — python3 montage.py config.json out.mp4 [--stills 1,9,120] [--sheet]
# 1440x1080 · 24fps · 372프레임(15.5초). 가운데 고정 키워드 + 계속 바뀌는 한글 신문 화면 + 숫자 카운터 + 낱자·도장 엔딩.
import sys, os, json, math, wave, subprocess, urllib.request, datetime
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, M = 1440, 1080, 24, 220
LIME, STAGE = (216, 244, 80), (241, 241, 243)
HERE = os.path.dirname(os.path.abspath(__file__))

# ───────── 서체 (전부 한글 서체. 내려받기에 실패하면 시스템 Noto CJK KR 로 대체) ─────────
GF = 'https://raw.githubusercontent.com/google/fonts/main/ofl/'
DL = {'NM4': 'nanummyeongjo/NanumMyeongjo-Regular.ttf', 'NM8': 'nanummyeongjo/NanumMyeongjo-ExtraBold.ttf',
      'BH': 'blackhansans/BlackHanSans-Regular.ttf', 'DH': 'dohyeon/DoHyeon-Regular.ttf',
      'GB': 'gowunbatang/GowunBatang-Bold.ttf', 'NG': 'nanumgothic/NanumGothic-ExtraBold.ttf',
      'SM': 'songmyung/SongMyung-Regular.ttf', 'GA5': 'gothica1/GothicA1-Medium.ttf',
      'GA9': 'gothica1/GothicA1-Black.ttf', 'IB': 'ibmplexsanskr/IBMPlexSansKR-Bold.ttf'}
NOTO = {'NS3': 'NotoSansCJK-Light', 'NS4': 'NotoSansCJK-Regular', 'NS5': 'NotoSansCJK-Medium', 'NS7': 'NotoSansCJK-Bold',
        'NS9': 'NotoSansCJK-Black', 'NSf4': 'NotoSerifCJK-Regular', 'NSf6': 'NotoSerifCJK-SemiBold',
        'NSf7': 'NotoSerifCJK-Bold', 'NSf9': 'NotoSerifCJK-Black'}
FALL = {'NM4': 'NSf4', 'NM8': 'NSf9', 'GB': 'NSf7', 'SM': 'NSf6', 'BH': 'NS9', 'DH': 'NS7', 'NG': 'NS9', 'GA5': 'NS5', 'GA9': 'NS9', 'IB': 'NS7'}
FPATH, _fc = {}, {}

def load_fonts(fdir):
    os.makedirs(fdir, exist_ok=True)
    roots = ['/usr/share/fonts', '/usr/local/share/fonts', os.path.expanduser('~/.fonts'), 'C:/Windows/Fonts', fdir]
    found = {}
    for rt in roots:
        for dp, _, fs in os.walk(rt):
            for f in fs: found[os.path.splitext(f)[0]] = os.path.join(dp, f)
    for k, stem in NOTO.items():
        p = found.get(stem) or found.get(stem.replace('CJK', 'CJKkr'))
        if not p: continue
        idx = 0
        for i in range(8):                       # .ttc 안에서 한국어(KR) 면을 고른다
            try:
                if 'KR' in ImageFont.truetype(p, 20, index=i).getname()[0]: idx = i; break
            except Exception: break
        FPATH[k] = (p, idx)
    if 'NS4' not in FPATH:
        sys.exit('한글 서체(Noto Sans/Serif CJK KR)를 찾지 못했습니다. fonts-noto-cjk 를 설치하세요.')
    for k in NOTO: FPATH.setdefault(k, FPATH['NS4'])
    miss = []
    for k, u in DL.items():
        p = os.path.join(fdir, os.path.basename(u))
        if not os.path.exists(p) or os.path.getsize(p) < 50000:
            try: urllib.request.urlretrieve(GF + u, p)
            except Exception: p = None
        try:
            ImageFont.truetype(p, 20); FPATH[k] = (p, 0)
        except Exception:
            FPATH[k] = FPATH[FALL[k]]; miss.append(k)
    print('서체 %d종 준비' % len(FPATH) + (' (대체: %s)' % ','.join(miss) if miss else ''))

def font(k, px):
    q = max(4, round(px * 4) / 4)
    if (k, q) not in _fc:
        if len(_fc) > 3000: _fc.clear()
        p, i = FPATH[k]; _fc[(k, q)] = ImageFont.truetype(p, q, index=i)
    return _fc[(k, q)]

# ───────── 공용 수학 ─────────
def bez(x1, y1, x2, y2):
    def f(x):
        x = min(max(x, 0.0), 1.0); lo, hi = 0.0, 1.0
        for _ in range(28):
            t = (lo + hi) / 2
            if 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t * t * x2 + t ** 3 < x: lo = t
            else: hi = t
        t = (lo + hi) / 2
        return 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t * t * y2 + t ** 3
    return f
lerp = lambda a, b, t: a + (b - a) * t
clamp = lambda v, a=0.0, b=1.0: max(a, min(b, v))
sstep = lambda a, b, x: (lambda t: t * t * (3 - 2 * t))(clamp((x - a) / (b - a)))
mixc = lambda a, b, t: tuple(int(round(lerp(a[i], b[i], t))) for i in range(3))
def seg(L, a, b, va, vb, e=None):          # L 이 a→b 로 갈 때 va→vb
    t = clamp((L - a) / (b - a)); return lerp(va, vb, e(t) if e else t)
def keys(tab, f):                          # [(f, v...), ...] 직선 보간
    if f <= tab[0][0]: return tab[0][1:]
    for a, b in zip(tab, tab[1:]):
        if f <= b[0]:
            t = (f - a[0]) / (b[0] - a[0]); return tuple(lerp(x, y, t) for x, y in zip(a[1:], b[1:]))
    return tab[-1][1:]
def rng(seed):
    st = [seed & 0xFFFFFFFF]
    def r():
        st[0] = (st[0] + 0x6D2B79F5) & 0xFFFFFFFF; a = st[0]
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r

E32 = 60 / 88 / 8 * FPS
STARTS, _a = [], 0.0
for _j in range(114):
    STARTS.append(21 + int(math.floor(_a + .5))); _a += E32 if _j < 80 else E32 / 2
WHIP = 21 + int(math.floor(_a + .5))        # 219

# ───────── 내용 (config 로 덮어씀) ─────────
D = dict(
    outlets=['새결일보', '모루신문', '윤슬타임스', '다온경제', '너울뉴스', '가온매거진', '프레임와이즈', '데일리 마루', '누리브리프',
             '볕뉘일보', '아름드리 리뷰', '시그널 데일리', '코너오피스', '키프레임', '오프컷', '먼데이 브리프', '라티스', '블루필드',
             '웨이포인트', '선다이얼', '브라이트라인', '미리내 포스트', '해오름 저널', '글마루', '도담일보', '한울 위클리', '더 라운드업', '온새미로'],
    kickers=['기획', '심층', '단독', '문화', '사회', '경제', '트렌드', '인터뷰', '오피니언', '현장'],
    names=['한서윤', '문태오', '윤가람', '송리아', '오루다', '하준서', '나예린', '모지환', '이하늘', '손세미'],
    nav=['정치', '경제', '사회', '문화', '국제', '오피니언', '영상'],
    fill='관계자는 이번 변화가 현장에 빠르게 자리 잡고 있다고 말했다 전문가들은 관심이 분기마다 늘고 있으며 이런 흐름이 당분간 이어질 것으로 내다봤다 '
         '현장의 목소리는 조금씩 달랐지만 방향은 같았다 지난해부터 이어진 논의가 올해 들어 구체적인 움직임으로 바뀌고 있다는 분석이 나온다',
    date=datetime.date.today().strftime('%Y.%m.%d.'), seed=0, search_total=128,
    counter={'from': 212, 'to': 300, 'unit': '%', 'subtitle': ''},
    closing={'line': '', 'sub': '', 'stamp': ''}, photos='', bgm='', bgm_hit=0.0, style='web')
LIGHT = [(250, 250, 250), (245, 243, 236), (253, 241, 229), (236, 241, 247), (242, 242, 242), (248, 246, 240), (244, 240, 232), (238, 244, 238)]
DARK = [(16, 17, 20), (14, 22, 48), (20, 58, 46), (24, 24, 28), (42, 18, 26), (10, 10, 10)]
ACC = [(196, 28, 36), (40, 80, 160), (30, 120, 80), (150, 36, 70), (200, 120, 20), (110, 60, 170)]
SOLID = [(64, 140, 62), (14, 22, 48), (20, 58, 46), (176, 30, 38), (28, 30, 36), (30, 70, 160)]
HF = [('NSf7', 1), ('NSf9', 1), ('NSf6', 1), ('NM8', 1), ('GB', 1), ('SM', 1), ('NS5', 1), ('NS7', 1), ('NS9', 1), ('NS3', 1),
      ('BH', 1), ('DH', 1), ('NG', 1), ('GA9', 1), ('IB', 1), ('NS9', .84), ('NSf9', .88), ('GA5', 1)]
MF = ['NSf9', 'NS9', 'BH', 'NM8', 'GA9', 'DH', 'SM', 'GB']
PF = ['NS4', 'NSf4', 'NM4', 'NS5', 'NSf6', 'GA5', 'NS7', 'NS3', 'GB']
PHOTOS = []
PAPER = [(236, 234, 228), (232, 230, 222), (240, 238, 232), (228, 227, 221), (238, 235, 226), (233, 233, 230)]
PHF = [('NS9', .82), ('NS9', .9), ('BH', .95), ('GA9', .85), ('NG', .9), ('NSf9', .85), ('NM8', .92), ('DH', .95), ('NS7', .85), ('NSf9', 1)]
GRAIN = None
def paper(a):                                        # 종이 신문 질감: 살짝 번진 인쇄 + 종이 결
    global GRAIN
    if GRAIN is None: GRAIN = cv2.GaussianBlur(np.random.RandomState(5).normal(0, 7, (H + 2 * M, W + 2 * M, 1)).astype(np.float32), (0, 0), .8)[..., None]
    a = cv2.GaussianBlur(np.asarray(a, np.float32), (0, 0), .55)
    return np.clip(a + GRAIN[:a.shape[0], :a.shape[1]], 0, 255).astype(np.uint8)

def mkphoto(k):
    r = np.random.default_rng(100 + k); w, h = 900, 600
    pal = [((70, 110, 170), (200, 215, 235)), ((60, 30, 90), (240, 130, 50)), ((15, 15, 20), (120, 90, 70)), ((190, 200, 205), (90, 110, 120)),
           ((150, 120, 90), (230, 215, 190)), ((20, 24, 30), (90, 100, 120)), ((120, 90, 60), (225, 200, 160)), ((200, 110, 50), (240, 225, 200))][k % 8]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); yy /= h; xx /= w
    t = (yy * .7 + xx * .3)[..., None]
    img = np.array(pal[0], np.float32) * (1 - t) + np.array(pal[1], np.float32) * t
    def blob(cx, cy, rx, ry, col, a):
        nonlocal img
        m = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) ** 1.6)[..., None] * a
        img = img * (1 - m) + np.array(col, np.float32) * m
    for _ in range(12):
        c = np.array(pal[int(r.integers(2))], np.float32) * r.uniform(.5, 1.5)
        blob(r.uniform(0, 1), r.uniform(0, 1), r.uniform(.04, .2), r.uniform(.05, .25), np.clip(c, 0, 255), r.uniform(.3, .8))
    cx = r.uniform(.35, .65); skin = (205, 165, 140) if k % 2 == 0 else (150, 110, 90)
    blob(cx, 1.02, .26, .42, np.array(pal[0]) * .45, .95); blob(cx, .36, .085, .16, skin, .95); blob(cx, .27, .09, .09, (30, 24, 22), .85)
    img = cv2.GaussianBlur(img, (0, 0), 3) + r.normal(0, 5, (h, w, 1))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

# ───────── 그리기 문맥: 화면점 = (cx,cy) + (쪽점 − (720,539))·sc ─────────
class Ctx:
    def __init__(s, w, h, cx, cy, sc, bg, K):
        s.im = Image.new('RGB', (w, h), bg); s.d = ImageDraw.Draw(s.im); s.cx, s.cy, s.sc, s.K = cx, cy, sc, K
    X = lambda s, x: s.cx + (x - 720) * s.sc
    Y = lambda s, y: s.cy + (y - 539) * s.sc
    def rect(s, x, y, w, h, col, r=0, ow=0):
        b = [s.X(x), s.Y(y), s.X(x + w), s.Y(y + h)]
        if b[2] < b[0] + .5 or b[3] < b[1] + .5: return
        kw = dict(outline=col, width=max(1, round(ow * s.sc))) if ow else dict(fill=col)
        s.d.rounded_rectangle(b, r * s.sc, **kw) if r else s.d.rectangle(b, **kw)
    def circle(s, x, y, r, col, ow=0):
        b = [s.X(x - r), s.Y(y - r), s.X(x + r), s.Y(y + r)]
        s.d.ellipse(b, **(dict(outline=col, width=max(1, round(ow * s.sc))) if ow else dict(fill=col)))
    def tw(s, t, px, f, ls=0, sx=1):
        return (font(f, px * s.sc).getlength(t) / s.sc + ls * len(t)) * sx
    def text(s, t, px, y, x=0, f='NS4', col=(20, 20, 20), al='l', ls=0, sx=1):
        if not t: return 0
        F = font(f, px * s.sc); w = s.tw(t, px, f, ls, sx)
        if al == 'r': x -= w
        elif al == 'c': x -= w / 2
        px0, py0 = s.X(x), s.Y(y)
        if px0 > s.im.width or px0 + w * s.sc < 0 or py0 < -px * s.sc or py0 > s.im.height + px * s.sc * 1.2: return w
        if sx != 1:                                   # 가로배율(장체): 따로 그려 가로로만 줄인다
            tw_, th = int(F.getlength(t)) + 8, int(px * s.sc * 1.5)
            m = Image.new('L', (tw_, th), 0); ImageDraw.Draw(m).text((2, th * .75), t, font=F, fill=255, anchor='ls')
            m = m.resize((max(1, int(tw_ * sx)), th), Image.LANCZOS)
            s.im.paste(col, (int(px0 - 2 * sx), int(py0 - th * .75)), m)
        elif ls:
            for ch in t:
                s.d.text((px0, py0), ch, font=F, fill=col, anchor='ls'); px0 += F.getlength(ch) + ls * s.sc
        else: s.d.text((px0, py0), t, font=F, fill=col, anchor='ls')
        return w
    def kwline(s, pre, post, px, f, col, kcol=None, sx=1, hl=True, box=LIME):
        """키워드 줄: 키워드 잉크의 중심이 정확히 (720,539) 에 온다. 줄 시작 x0 과 기준선 y 를 돌려준다."""
        F = font(f, px * s.sc); l, t, r, b = [v / s.sc for v in F.getbbox(s.K, anchor='ls')]
        l, r = l * sx, r * sx; adv = s.tw(s.K, px, f, 0, sx)
        xk, y = 720 - (l + r) / 2, 539 - (t + b) / 2
        hh = max(1.10 * px, (b - t) + .18 * px); pad = .07 * px
        if hl: s.rect(720 - (r - l) / 2 - pad, 539 - hh / 2, (r - l) + 2 * pad, hh, box)
        wp = s.tw(pre, px, f, 0, sx); x0 = xk - wp
        s.text(pre, px, y, x0, f, col, sx=sx); s.text(s.K, px, y, xk, f, kcol or col, sx=sx); s.text(post, px, y, xk + adv, f, col, sx=sx)
        return x0, y
    def rich(s, line, px, y, x, f, col, pale=(222, 240, 150)):   # 본문 속 {K} 에 옅은 라임 상자
        for i, part in enumerate(line.split('{K}')):
            if i:
                w = s.tw(s.K, px, f); s.rect(x - 4, y - .86 * px, w + 8, 1.08 * px, pale)
                x += s.text(s.K, px, y, x, f, (16, 16, 16))
            x += s.text(part, px, y, x, f, col)
    def body(s, x0, y0, n, gap, width, col, px, seed, f='NS4'):
        ws = D['fill'].split(); k = seed * 7
        for i in range(n):
            ln = ''
            while s.tw(ln, px, f) < width: ln += ws[k % len(ws)] + ' '; k += 1
            s.text(ln, px, y0 + i * gap, x0, f, col)
    def photo(s, k, x, y, w, h, focus=(.5, .5), mul=None, circle=False, zoom=1, gray=False):
        b = [s.X(x), s.Y(y), s.X(x + w), s.Y(y + h)]
        if b[2] < 0 or b[3] < 0 or b[0] > s.im.width or b[1] > s.im.height: return
        tw_, th = max(2, int(b[2] - b[0])), max(2, int(b[3] - b[1])); p = PHOTOS[k % len(PHOTOS)]
        sc = max(tw_ / p.width, th / p.height) * zoom; cw, ch = tw_ / sc, th / sc
        l = clamp(focus[0] * p.width - cw / 2, 0, p.width - cw); t = clamp(focus[1] * p.height - ch / 2, 0, p.height - ch)
        im = p.resize((tw_, th), Image.BILINEAR, box=(l, t, l + cw, t + ch))
        if gray: im = im.convert('L').convert('RGB')
        if mul: im = Image.fromarray((np.asarray(im, np.float32) * (np.array(mul, np.float32) / 255)).astype(np.uint8))
        mask = None
        if circle:
            mask = Image.new('L', (tw_ * 2, th * 2), 0); ImageDraw.Draw(mask).ellipse([0, 0, tw_ * 2 - 1, th * 2 - 1], fill=255)
            mask = mask.resize((tw_, th), Image.LANCZOS)
        s.im.paste(im, (int(b[0]), int(b[1])), mask)
    def avatar(s, k, cx, cy, r, zoom=1.4): s.photo(k, cx - r, cy - r, 2 * r, 2 * r, (.5, .33), None, True, zoom)

# ───────── 쪽 템플릿 (기준 px 90) ─────────
def page_A(c, hl):                                   # 도입부 첫 쪽: 검은 머리띠 신문
    pre, post, l2 = HEADS[0]; ink = (20, 20, 20)
    c.rect(-3000, -3000, 9000, 3385, (16, 16, 16))
    c.text(D['outlets'][0], 72, 362, 1394, 'NSf9', (245, 245, 245), 'r')
    for yy in (330, 344, 358): c.rect(30, yy - 3, 32, 6, (221, 221, 221))
    x0, y = c.kwline(pre, post, 96, 'NSf6', ink, hl=hl)
    c.text(D['kickers'][0] + '  >  ' + D['kickers'][1], 20, y - 121, x0 + 30, 'NS5', (120, 120, 120), ls=1)
    c.text(l2, 96, y + 112, x0, 'NSf6', ink)
    c.text('%s 기자   ·   %s' % (D['names'][0], D['kickers'][0]), 22, y + 196, x0, 'NS4', (70, 70, 70))
    c.text('입력 %s 오전 09:00   수정 %s 오전 11:49' % (D['date'], D['date']), 19, y + 250, x0, 'NS4', (110, 110, 110))
    c.rect(x0 + 1240, y + 168, 140, 34, (236, 236, 236))
    for i in range(3): c.circle(x0 + 30 + 72 * i, y + 350, 7, (90, 110, 200))
    c.body(x0, y + 450, 6, 46, 1250, (70, 70, 70), 26, 1)
    c.photo(0, x0 + 20, y + 750, 1270, 900, (.5, .3))

def page_T(c):                                       # 마지막 컷: 하이라이트 초근접
    c.rect(-3000, 256, 9000, 570, LIME)
    px = clamp(1500 / max(c.tw(c.K, 100, 'NSf6') / 100, .1), 260, 500)
    F = font('NSf6', px * c.sc); l, t, r, b = [v / c.sc for v in F.getbbox(c.K, anchor='ls')]
    c.text(c.K, px, 541 - (t + b) / 2, 720 - (l + r) / 2, 'NSf6', (14, 14, 14))

def page_gen(mk, i):                                 # 가상 매체 기사 첫 화면 (틀 8종)
    r = rng(D['seed'] * 131 + 1000 + i * 7919); pick = lambda a: a[int(r() * len(a))]
    dark = r() < .32; bg = pick(DARK) if dark else pick(LIGHT)
    ink = (238, 238, 242) if dark else (16 + int(r() * 30), 16 + int(r() * 24), 20 + int(r() * 30))
    mut = (150, 150, 158) if dark else (118, 118, 122); acc = pick(ACC)
    hf, sx = pick(HF); mf = pick(MF); kick = pick(D['kickers']); v = int(r() * 8)
    al = pick('lcr'); ms = 46 + int(r() * 50); navY = 230 + int(r() * 40); name = pick(D['names']); fy = .3 + r() * .4
    pre, post, l2 = HEADS[(i * 7 + 1) % len(HEADS)]; outlet = D['outlets'][(i * 5) % len(D['outlets'])]
    mx = {'l': 120, 'c': 720, 'r': 1320}[al]; line = (40, 40, 40) if not dark else (60, 60, 66)
    if v == 5: bg = pick(SOLID); dark = True; ink = (250, 250, 250); mut = (200, 215, 205)
    if v == 6: bg = (232, 234, 240); dark = False; ink = (20, 20, 20); mut = (120, 120, 130); hf, sx = pick(['NS4', 'NS5', 'GA5']), 1
    if v == 7: bg = (245, 243, 236); dark = False; mut = (90, 90, 90)
    kc = (14, 14, 14) if dark else ink
    c = mk(bg)
    if v == 2:
        c.photo(i, -400, -330, 2300, 600, (.5, fy), None if dark else (170, 170, 176))
        c.text(outlet, ms, 240, mx, mf, (250, 250, 250), al)
    elif v == 4:
        band = pick([(16, 16, 16), acc, (16, 38, 78)])
        c.rect(-3000, -3000, 9000, 3250, band); c.rect(-3000, 250, 9000, 62, mixc(bg, (128, 128, 128), .08))
        c.text(outlet, ms + 14, 204, mx, mf, (250, 250, 250), al); x = 150
        for n in D['nav']: x += c.text(n, 22, 290, x, 'NS5', mut) + 44
    elif v == 6:
        c.rect(60, 220, 1320, 760, (255, 255, 255), 18)
        c.avatar(i, 180, 310, 46); c.text(name, 28, 304, 250, 'NS7', (20, 20, 20)); c.text('@' + outlet.replace(' ', '') + '  ·  12분', 21, 338, 250, 'NS4', mut)
        c.text('♡ 2.4천      ↻ 318      공유', 23, 920, 180, 'NS4', mut)
    elif v == 7:
        c.text('%s  —  제 %d호' % (outlet, 100 + i * 3), 22, 300, 720, 'NS5', (60, 60, 60), 'c', ls=4)
        c.rect(200, 329, 1040, 2, (60, 60, 60)); c.rect(200, 338, 1040, 1, (60, 60, 60))
    elif v == 5:
        for k in range(6): c.circle(632 + 108 * k, 152, 12, (255, 255, 255), 2.5)
    else:
        c.text(outlet, ms, 196, mx, mf, ink, al)
        c.rect(-3000, navY - 34, 9000, 2 if not dark else 1, line); x = 140 if al == 'r' else 420
        for n in D['nav']: x += c.text(n, 20, navY, x, 'NS5', mut) + 40
    x0, y = c.kwline(pre, post, 90, hf, ink, kc, sx)
    if v not in (5, 6, 7): c.text(kick, 19, y - 116, x0, 'NS7', acc, ls=2.5)
    c.text(l2, 90, y + 104, x0, hf, mut if v in (5, 7) else ink, sx=sx)
    by = y + 178
    if v in (5, 6): return c
    if v == 3:
        c.avatar(i + 3, x0 + 24, by - 8, 24); c.text(name + ' 기자', 21, by, x0 + 62, 'NS7', mut)
        c.rect(x0 - 60, 300, 2, 900, (210, 210, 210) if not dark else (70, 70, 76)); c.body(x0, by + 70, 5, 44, 1150, mut, 25, i + 3)
    else:
        c.text('%s 기자   ·   %s' % (name, D['date']), 21, by, x0, 'NS4', mut)
        if v in (0, 4) and r() < .75: c.photo(i, x0, by + 40, 1100, 560, (.5, fy))
        else: c.body(x0, by + 70, 7, 44, 1150, mut, 25, i)
    return c

def page_print(mk, i, hl=True, ext=0):                 # 종이 신문 지면: 머리띠 · 어깨제목 · 굵은 제목 · 다단 본문 · 괘선
    r = rng(D['seed'] * 131 + 9000 + i * 7919); pick = lambda a: a[int(r() * len(a))]
    bg = pick(PAPER); g = 26 + int(r() * 14); ink = (g, g, g - 2); mut = (96, 94, 90)
    hf, sx = pick(PHF); v = int(r() * 4); cw = 226 + int(r() * 44); bpx = 19 + int(r() * 3); gap = 26
    two = r() < .6; y2o = 300 + int(r() * 90); pho = r() < .45; pcol = 1 + int(r() * 3); hf2, sx2 = pick(PHF); cen = r() < .5
    if i < 0: v, two, pho = 0, True, True
    pre, post, l2 = HEADS[(i * 7 + 1) % len(HEADS)] if i >= 0 else HEADS[0]
    h2 = HEADS[(i * 7 + 4) % len(HEADS)]; h3 = HEADS[(i * 7 + 9) % len(HEADS)]; h4 = HEADS[(i * 7 + 13) % len(HEADS)]
    outlet = D['outlets'][(i * 5) % len(D['outlets'])] if i >= 0 else D['outlets'][0]
    c = mk(bg); rev = v == 2
    if rev: c.rect(-4000, 539 - 74, 12000, 148, (22, 22, 22))
    x0, y = c.kwline(pre, post, 90, hf, (244, 244, 240) if rev else ink, (16, 16, 16) if rev else None, sx, hl)
    over = h3[2]; ox = 720 - c.tw(over, 46, 'NS5') / 2 if cen else x0
    c.text(over, 46, y - (132 if rev else 112), ox, 'NS5' if hf.startswith('NS') and not hf.startswith('NSf') else 'NSf6', ink)
    c.text(l2, 42, y + (112 if rev else 92), x0 + 30, 'NS7', ink); c.rect(-4000, y + (132 if rev else 114), 12000, 1.5, ink)
    blocked = []; yb = y + (190 if rev else 168); top2 = None
    if v in (0, 3):                                  # 지면 머리띠: 날짜 · 단기 · 제호 · 면
        try: dt = datetime.date(*[int(t) for t in D['date'].strip('.').split('.')])
        except Exception: dt = datetime.date.today()
        yh = y - 222
        c.text('%d년 %d월 %d일  %s요일' % (dt.year, dt.month, dt.day, '월화수목금토일'[dt.weekday()]), 22, yh, x0 - 40, 'NS5', ink)
        c.text('단기 %d년   제%d호' % (dt.year + 2333, 12000 + (i + 2) * 37), 20, yh, x0 + 420, 'NS4', mut)
        c.text(outlet, 30, yh + 2, x0 + 900, 'NS9', ink); c.text('%s  %d' % (pick(D['kickers']), 2 + (i + 3) % 28), 22, yh, x0 + 1240, 'NS7', ink)
        c.rect(-4000, yh + 16, 12000, 2.5, ink)
        if v == 3: c.rect(x0 - 200, yh + 34, 96, 40, (22, 22, 22)); c.text(pick(D['kickers']), 22, yh + 63, x0 - 152, 'NS7', (240, 240, 236), 'c')
    else:
        top2 = y - 190; c.rect(-4000, top2 + 14, 12000, 4, ink)
    step = cw + gap; xs = x0 - math.ceil((x0 + 300) / step) * step; xmax = 1720 + ext * 1000; bot = 1340 + ext * 760
    if two:                                          # 아래쪽 둘째 기사
        y2 = y + y2o; c.rect(-4000, y2, 12000, 3.5, ink)
        c.text(h2[2], 66, y2 + 92, x0 - 30, hf2, ink, sx=sx2); c.text(h4[2], 30, y2 + 142, x0 - 10, 'NS7', ink)
        blocked.append((-9000, y2 - 16, 9000, y2 + 172))
    if pho:
        px_, py_ = x0 + pcol * step, yb + 6; c.photo(i + 2, px_, py_, cw * 2 + gap, 176, (.5, .35), None, False, 1, True)
        c.text(h3[0] + KW + h3[1], 15, py_ + 198, px_, 'NS5', mut); blocked.append((px_ - 4, py_ - 26, px_ + cw * 2 + gap + 4, py_ + 214))
    S = (D['fill'] + ' ') * 4; n = max(6, int(cw / bpx) + 1); k = int(r() * 60); lh = bpx * 1.58; x = xs
    while x < xmax:
        for a, b in ([(top2 - 470 - ext * 300, top2 - 8)] if top2 else []) + [(yb, bot)]:
            yy = a
            while yy < b:
                if not any(bx0 < x + cw and x < bx1 and by0 < yy and yy - bpx < by1 for bx0, by0, bx1, by1 in blocked):
                    m = n
                    while m < n + 7 and c.tw(S[k:k + m + 1].strip(), bpx, 'NSf4') <= cw: m += 1
                    ln = S[k:k + m].strip(); k = (k + m) % (len(S) - n - 9)
                    c.text(ln, bpx, yy, x, 'NSf4', ink, ls=clamp((cw - c.tw(ln, bpx, 'NSf4')) / max(len(ln) - 1, 1), 0, 7))
                yy += lh
        if v == 3: c.rect(x - gap / 2, yb - 20, 1, bot - yb, mut)
        x += step
    return c

def page_pas(mk, i):                                 # 큰 본문 클로즈업 (문장 한가운데의 키워드)
    r = rng(D['seed'] * 131 + 5000 + i * 104729); pick = lambda a: a[int(r() * len(a))]
    dark = r() < .36; bg = pick(DARK[:4] + [(26, 26, 30)]) if dark else pick([(246, 246, 246), (244, 244, 244), (246, 246, 244), (242, 242, 242), (240, 240, 240), (243, 235, 218)])
    ink = (228, 228, 232) if dark else (24 + int(r() * 30),) * 3; f = pick(PF); gap = 150 + int(r() * 40); v = int(r() * 6)
    ab, pre, post, be = PASS[(i * 5 + 2) % len(PASS)]; other = PASS[(i * 5 + 5) % len(PASS)]
    ab = (other[3][-1:] + ab)[-2:]; be = (be + other[0] + other[3])[:3]
    if D['style'] != 'web': dark = False; bg = PAPER[i % len(PAPER)]; ink = (30, 30, 28); f = ['NSf4', 'NM4', 'NSf6', 'NS4', 'GB', 'NSf4'][i % 6]
    c = mk(bg); line = (200, 200, 200) if not dark else (74, 74, 80)
    x0, y = c.kwline(pre, post, 90, f, ink, (16, 16, 16) if dark else None)
    if v == 0: c.rect(-3000, y - gap * 2.6, 9000, 3, line)
    elif v == 1 and not dark and D['style'] == 'web': c.rect(457, -400, 2400, y - gap * 1.25 + 400, (8, 8, 8)); ab = []
    elif v == 2:
        c.text('%s 오후 4:00' % D['date'], 60, y - gap * 2.45, x0, 'NS7', ink); c.rect(x0 - 5, y - gap * 2.45 + 70, 3000, 3, line); ab = ab[-1:]
    elif v == 3:
        c.avatar(i, x0 + 80, y - gap * 2.6, 84, 1.8); c.text(pick(D['names']) + ' 기자', 54, y - gap * 2.6 + 20, x0 + 200, 'NS7', (190, 40, 50)); ab = ab[-1:]
    for k, ln in enumerate(reversed(ab)): c.rich(ln, 90, y - gap * (k + 1), x0, f, ink)
    for k, ln in enumerate(be):
        c.rich(ln, 90, y + gap * (k + 1), x0, f, ink)
        if v == 4 and k == 0: c.rect(x0, y + gap + 20, c.tw(ln.replace('{K}', c.K), 90, f), 4, ink)
    return c

# ───────── 렌즈 (술통 왜곡 + 방사 흐림 + 가장자리 어둡게) ─────────
_gy, _gx = np.mgrid[0:H, 0:W].astype(np.float32); _dx, _dy = _gx - 720, _gy - 539
_r2 = (_dx ** 2 + _dy ** 2) / 900.0 ** 2
_vgm = np.clip((_r2 - .25) / 1.05, 0, 1); _vgm = (_vgm * _vgm * (3 - 2 * _vgm))[..., None]
def lens(canvas, sc, K, B0, B1, VG):
    qx, qy = _dx * (1 + K * _r2), _dy * (1 + K * _r2); bl = B0 + B1 * _r2
    n = int(clamp(math.ceil(float(bl.max()) * 900 / 1.5), 1, 24)); acc = np.zeros((H, W, 3), np.float32)
    for i in range(n):
        f = (1 - bl * (i / max(n - 1, 1))) / sc
        acc += cv2.remap(canvas, 720 + M + qx * f, 539 + M + qy * f, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return acc / n * (1 - VG * _vgm)
eK, eB, eV = bez(.5, .35, .66, .65), bez(0, .07, .36, .95), bez(.5, .47, 0, .6)

def searchbar(fr, rect, txt, match, alpha=1.0):
    x0, y0, x1, y1 = rect
    if y1 <= 0: return fr
    S = 3; h = y1 - y0; s = h / 60; w = x1 - x0; pad = 6
    im = Image.new('RGBA', (int((w + 2 * pad) * S), int((h + 2 * pad) * S)), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    P = lambda x, y: ((x + pad) * S, (y + pad) * S)
    d.rounded_rectangle([P(0, 0), P(w, h)], h / 2 * S, fill=(24, 24, 26), outline=(70, 70, 74), width=int(2 * s * S))
    cx, cy = 34 * s, h / 2 - 2 * s
    d.ellipse([P(cx - 8 * s, cy - 8 * s), P(cx + 8 * s, cy + 8 * s)], outline=(210, 210, 210), width=int(2.6 * s * S))
    d.line([P(40 * s, h / 2 + 4 * s), P(46 * s, h / 2 + 10 * s)], fill=(210, 210, 210), width=int(2.6 * s * S))
    d.text(P(58 * s, h / 2 + 7 * s), txt, font=font('NS5', 20 * s * S), fill=(230, 230, 230), anchor='ls')
    if match: d.text(P(w - 46 * s, h / 2 + 6 * s), match, font=font('NS5', 17 * s * S), fill=(200, 200, 200), anchor='rs')
    im = im.resize((im.width // S, im.height // S), Image.LANCZOS)
    if alpha < 1: im.putalpha(im.getchannel('A').point(lambda v: int(v * alpha)))
    out = Image.fromarray(fr.astype(np.uint8)).convert('RGBA'); out.alpha_composite(im, (int(round(x0 - pad)), int(round(y0 - pad))))
    return np.asarray(out.convert('RGB'), np.float32)

# ───────── 본편 프레임 ─────────
_cut = {}
def kpx(j): return (93 + 32 * j / 113) * KS
def cut_canvas(j):
    if j not in _cut:
        _cut.clear(); mk = lambda bg: Ctx(W + 2 * M, H + 2 * M, 720 + M, 539 + M, kpx(j) / 90, bg, KW)
        if j == 113: c = mk((14, 14, 14)); c.sc = 1; page_T(c)
        elif j == 112 or KIND[j]: c = page_pas(mk, j)
        elif D['style'] == 'print' or (D['style'] == 'mix' and j % 2): c = page_print(mk, j)
        else: c = page_gen(mk, j)
        _cut[j] = paper(c.im) if D['style'] != 'web' and j != 113 else np.asarray(c.im)
    return _cut[j]

def frame_intro(L):
    cy = seg(L, 1, 9, 470, 285, bez(.22, .82, .74, .98)); cx = 464; sc = .55
    if L >= 9:
        cx = seg(L, 9, 16, 464, 719, bez(.59, -.02, .44, .95)); sc = seg(L, 9, 16, .55, 1.04, bez(.71, .08, .40, .76))
        cy = 285 - 6 * clamp(L - 9) if L <= 10 else seg(L, 10, 16, 279, 539, bez(.5, .12, .39, .95))
    if L > 16: cx = seg(L, 16, 20, 719, 720); sc = seg(L, 16, 20, 1.04, 1.045)
    if D['style'] == 'web':
        c = Ctx(W, H, cx, cy, sc * KS, (250, 250, 250), KW); page_A(c, L >= 9); fr = np.asarray(c.im, np.float32)
    else:
        c = page_print(lambda bg: Ctx(W, H, cx, cy, sc * KS * 96 / 90, bg, KW), -1, L >= 9, 1); fr = paper(c.im).astype(np.float32)
    vg = seg(L, 14, 19, 0, .04) if L < 20 else .045
    if vg > 0: fr = fr * (1 - vg * _vgm)
    if L <= 9: rc = (seg(L, 1, 9, 450, 449), seg(L, 1, 9, 207, 43, bez(.09, .08, .2, .89))); rc = (rc[0], rc[1], 1010, rc[1] + 60)
    elif L < 20: rc = tuple(seg(L, 9, 19, a, b) for a, b in zip((449, 43, 1010, 103), (448, 30, 1010, 92)))
    else: rc = (440, 29, 1010, 90)
    n = len(KW) if L >= 8 else int(round(len(KW) * (0, .1, .3, .4, .6, .7, .9)[L - 1]))
    return searchbar(fr, rc, KW[:n] + ('|' if L in (2, 5, 6) else ''), '1/%d' % D['search_total'] if L >= 9 else '', .35 if L == 1 else 1)

def frame_montage(L):
    j = max(k for k in range(114) if STARTS[k] <= L); tp = L - STARTS[j]; u = j / 113
    K, B0, B1, VG = .23 * eK(u), 0, .022 * eB(u), .06 + .18 * eV(u)
    if j == 112: B0, B1 = .05, .06
    fr = lens(cut_canvas(j), 1 + (.08 if j == 113 else .012) * tp, K, B0, B1, VG)
    p = min(j / 45.2, 1)
    rc = (lerp(448, 397, bez(.14, .54, 0, .16)(p)), lerp(30, -70, bez(1, .5, .86, .59)(p)),
          lerp(1010, 1043, bez(.12, -.1, .15, 0)(p)), lerp(92, -10, bez(1, .42, .88, .59)(p)))
    return searchbar(fr, rc, KW, '%d/%d' % (j + 2, D['search_total']))

# ───────── 휩 + 카운터 ─────────
CF = 'NS9'
CFRAC = [0, .114, .227, .352, .466, .58, .693, .784, .864, .932, .977, 1]
CCOL = [(232, 22, 122), (227, 78, 88), (214, 192, 46), (208, 249, 22), (214, 192, 46), (228, 80, 90), (235, 19, 112), (228, 80, 90)]
SUBC = [(232, 22, 122), (214, 190, 50), (212, 242, 30), (226, 122, 60), (232, 22, 122), (226, 110, 100)]
def cval(k):
    a, b = D['counter']['from'], D['counter']['to']; v = a + (b - a) * CFRAC[min(k, 11)]
    isint = float(a).is_integer() and float(b).is_integer()
    return ('{:,}'.format(int(round(v))) if isint else '%.1f' % v) + D['counter']['unit']
def num_layout(txt, px):                                  # 숫자는 모두 «0» 폭으로(값이 바뀌어도 덩어리가 덜 출렁인다)
    F = font(CF, px); a0 = F.getlength('0'); xs, x = [], 0.0
    for ch in txt: xs.append(x); x += (a0 if ch.isdigit() else F.getlength(ch)) - 12 * px / 434
    l = F.getbbox(txt[0], anchor='ls')[0]; rgt = xs[-1] + F.getbbox(txt[-1], anchor='ls')[2]
    return xs, (l + rgt) / 2, rgt - l
def draw_num(d, txt, cols, cx, cy, s, dx=0):
    px = NPX * s; xs, mid, _ = num_layout(txt, px); F = font(CF, px)
    for ch, x, col in zip(txt, xs, cols): d.text((cx + dx * s + x - mid, cy + .35 * NPX * s), ch, font=F, fill=col, anchor='ls')
def frame_counter(L):
    k = L - WHIP; bg = PAPER[0] if D['style'] != 'web' else (250, 250, 250); ink = (16, 16, 16); txt = cval(k)
    pp = (lambda a: paper(np.clip(a, 0, 255).astype(np.uint8)).astype(np.float32)) if D['style'] != 'web' else (lambda a: a)
    if k == 0:                                            # 휩: 분홍 숫자를 44벌 겹친 방사 번짐
        m = Image.new('L', (W, H), 0); draw_num(ImageDraw.Draw(m), txt, [255] * len(txt), 680, 470, 1.85)
        m = np.asarray(m, np.float32) / 255; acc = np.zeros((H, W), np.float32)
        for q in range(44):
            sk = math.exp(lerp(math.log(4.2), math.log(1.85), 1 - (1 - q / 43) ** 2.5)) / 1.85
            acc += cv2.warpAffine(m, np.float32([[sk, 0, 680 * (1 - sk)], [0, sk, 470 * (1 - sk)]]), (W, H), flags=cv2.INTER_LINEAR)
        a = (acc / 44)[..., None]
        return pp(np.array(bg, np.float32) * (1 - a) + np.array(CCOL[0], np.float32) * a)
    s = seg(L, 219, 231, 3.2, 1.0, bez(.19, 1.12, .54, .96)) if L <= 231 else seg(L, 231, 244, 1.0, .977)
    cx = seg(L, 219, 225, 680, 717, bez(.22, .65, 0, .49)) if L <= 225 else seg(L, 225, 231, 717, 717.5)
    cy = seg(L, 219, 225, 470, 518, bez(.8, 1.05, .2, .81)) if L <= 225 else seg(L, 225, 231, 518, 519)
    im = Image.new('RGB', (W, H), bg); d = ImageDraw.Draw(im); n = len(txt)
    if k == 7: draw_num(d, cval(6), [mixc(bg, (150, 150, 154), .30)] * 9, cx, cy, s, -18)
    if k == 8: draw_num(d, txt, [mixc(bg, (110, 110, 114), .55)] * 9, cx, cy, s, -24)
    cols = [mixc((0, 0, 0), CCOL[k], clamp(i * 3 / max(n - 1, 1) - k * .5 + 1)) for i in range(n)] if k <= 7 else [ink] * n
    draw_num(d, txt, cols, cx, cy, s)
    sub = D['counter']['subtitle']
    if sub and L >= 225:
        vis = int(round(len(sub) * (.14, .52, .67, .86, .95, 1)[min(L - 225, 5)])); F = font('NS7', SPX * s)
        x = cx - (F.getlength(sub) + 1.2 * s * len(sub)) / 2
        for i, ch in enumerate(sub[:vis]):
            d.text((x, cy + 243 * s), ch, font=F, fill=SUBC[L - 225] if (L <= 230 and i >= vis - 4) else ink, anchor='ls'); x += F.getlength(ch) + 1.2 * s
    return pp(np.asarray(im, np.float32))

# ───────── 엔딩: 낱자 활자판 → 다시 모임 → 문구 · 도장 ─────────
CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'; JUNG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
JONG = [''] + list('ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ')
CNAME = dict(zip(CHO, ['기역', '쌍기역', '니은', '디귿', '쌍디귿', '리을', '미음', '비읍', '쌍비읍', '시옷', '쌍시옷', '이응', '지읒', '쌍지읒', '치읓', '키읔', '티읕', '피읖', '히읗']))
def decomp(ch):
    o = ord(ch) - 0xAC00
    return [CHO[o // 588], JUNG[o % 588 // 28]] + ([JONG[o % 28]] if o % 28 else []) if 0 <= o < 11172 else [ch]
def jname(j): return CNAME.get(j) or (chr(0xAC00 + 11 * 588 + JUNG.index(j) * 28) if j in JUNG else '')
eo = lambda u: 1 - (1 - clamp(u)) ** 3
eio = lambda u: (lambda t: 4 * t ** 3 if t < .5 else 1 - (-2 * t + 2) ** 3 / 2)(clamp(u))
def eob(u, k=1.4): u = clamp(u); return 1 + (k + 1) * (u - 1) ** 3 + k * (u - 1) ** 2
def mkseal(txt):
    ch = [c for c in txt if c.strip()][:4]; n = len(ch)
    w, h = {1: (300, 300), 2: (180, 320), 3: (150, 380), 4: (300, 300)}[n]
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([6, 6, w - 6, h - 6], 20, fill=(188, 30, 36, 255)); d.rounded_rectangle([24, 24, w - 24, h - 24], 10, outline=(250, 236, 230, 255), width=7)
    iw, ih = w - 70, h - 70
    cells = {1: [(0, 0, 1, 1)], 2: [(0, 0, 1, 2), (0, 1, 1, 2)], 3: [(0, 0, 1, 3), (0, 1, 1, 3), (0, 2, 1, 3)],
             4: [(0, 0, 2, 2), (1, 0, 2, 2), (0, 1, 2, 2), (1, 1, 2, 2)]}[n]
    for c_, (cx_, cy_, nx, ny) in zip(ch, cells):
        cw, chh = iw / nx, ih / ny; F = font('NSf9', min(cw, chh) * .9); l, t, r, b = F.getbbox(c_, anchor='ls')
        d.text((35 + cw * (cx_ + .5) - (l + r) / 2, 35 + chh * (cy_ + .5) - (t + b) / 2), c_, font=F, fill=(250, 236, 230, 255), anchor='ls')
    a = np.asarray(im).copy(); nz = cv2.GaussianBlur(np.random.RandomState(3).rand(h, w).astype(np.float32), (0, 0), 2.2)
    a[..., 3] = (a[..., 3] * (nz > np.percentile(nz, 7))).astype(np.uint8)
    return Image.fromarray(a)
def build_end():
    F = 'NSf9'; px = min(230, 1050 / (font(F, 100).getlength(KW) / 100)); Fk = font(F, px)
    l, t, r, b = Fk.getbbox(KW, anchor='ls'); cy = 440; x0, base = 720 - (l + r) / 2, cy - (t + b) / 2
    syl = []
    for i, ch in enumerate(KW):
        if not ch.strip(): continue
        x = x0 + Fk.getlength(KW[:i]); bl, bt, br, bb = Fk.getbbox(ch, anchor='ls')
        syl.append(dict(ch=ch, cx=x + (bl + br) / 2, cy=base + (bt + bb) / 2, j=decomp(ch)))
    tiles = [(j, jname(j), si) for si, sy in enumerate(syl) for j in sy['j']]; N, G = len(tiles), len(syl)
    ts = min(128, (1320 - 16 * (N - G) - 44 * (G - 1)) / N); x = 720 - (N * ts + 16 * (N - G) + 44 * (G - 1)) / 2; pos, prev = [], None
    for j, nm, si in tiles:
        if prev is not None: x += 44 if si != prev else 16
        pos.append(x + ts / 2); x += ts; prev = si
    cl = D['closing']
    try: dt = datetime.date(*[int(v) for v in D['date'].strip('.').split('.')])
    except Exception: dt = datetime.date.today()
    return dict(F=F, px=px, x0=x0, base=base, box=(x0 + l, base + t, x0 + r, base + b), cy=cy, syl=syl, tiles=tiles, pos=pos, ts=ts, N=N,
                step=min(.085, .8 / max(N, 1)), line=cl.get('line', ''), sub=cl.get('sub') or '%d년 %d월 %d일' % (dt.year, dt.month, dt.day),
                seal=mkseal(cl.get('stamp') or ''.join(c for c in KW if '가' <= c <= '힣')[:4] or KW[:4]))
def glyph(d, ch, x, y, F, col):
    l, t, r, b = F.getbbox(ch, anchor='ls'); d.text((x - (l + r) / 2, y - (t + b) / 2), ch, font=F, fill=col, anchor='ls')
def frame_ending(n, last):
    E = END; c = n / FPS; pr = D['style'] != 'web'; bg = PAPER[0] if pr else STAGE; ink = (26, 26, 24); mut = (110, 108, 104)
    im = Image.new('RGB', (W, H), bg); d = ImageDraw.Draw(im); px = E['px']; F = font(E['F'], px)
    kl, kt, kr, kb = E['box']; cy = E['cy']; pad = .07 * px; hh = max(1.1 * px, (kb - kt) + .18 * px)
    col = lambda a, cc=ink: mixc(bg, cc, clamp(a))
    if c < .25:                                       # 1 · 카운터 화면이 가로 한 줄로 눌린다
        u = clamp(c / .25) ** 3; h = max(2, int(H * (1 - u))); im.paste(last.resize((W, h), Image.LANCZOS), (0, int(lerp(H / 2, cy, u) - h / 2)))
        if c > .19: d.rectangle([0, cy - 3, W, cy + 3], fill=LIME)
    elif c < .95:                                     # 2 · 그 줄이 하이라이트 상자로 자라고 키워드가 켜진다
        u = eo((c - .25) / .3); bw, bh = lerp(W, kr - kl + 2 * pad, u), lerp(6, hh, u)
        d.rectangle([720 - bw / 2, cy - bh / 2, 720 + bw / 2, cy + bh / 2], fill=LIME)
        if c >= .42: d.text((E['x0'], E['base']), KW, font=F, fill=ink, anchor='ls')
    elif c < 2.75:                                    # 3 · 낱자로 흩어져 활자판처럼 늘어섰다가 다시 모인다
        ts, back = E['ts'], eio((c - 2.3) / .4); bx = 1 - clamp((c - .95) / .15)
        if bx > 0: d.rectangle([kl - pad, cy - hh / 2, kr + pad, cy + hh / 2], fill=col(bx, LIME))
        tb = clamp((c - 1.15) / .15) * (1 - clamp((c - 2.3) / .15)); cur = int(math.floor((c - 1.45) / E['step'])) if c >= 1.45 else -1
        Fc = font('NS5', max(20, ts * .22))
        for k, (j, nm, si) in enumerate(E['tiles']):
            x = E['pos'][k]
            if tb > 0:
                if k == cur: d.rounded_rectangle([x - ts / 2, cy - ts / 2, x + ts / 2, cy + ts / 2], ts * .12, fill=col(tb, LIME))
                d.rounded_rectangle([x - ts / 2, cy - ts / 2, x + ts / 2, cy + ts / 2], ts * .12, outline=col(tb * .9), width=3)
            ca = clamp((c - 1.35 - .04 * k) / .1) * (1 - clamp((c - 2.3) / .12))
            if ca > 0 and nm: glyph(d, nm, x, cy + ts / 2 + max(26, ts * .3), Fc, col(ca, mut))
        sa = max(1 - clamp((c - .95) / .1), clamp((c - 2.62) / .13))
        if sa > 0: d.text((E['x0'], E['base']), KW, font=F, fill=col(sa), anchor='ls')
        for k, (j, nm, si) in enumerate(E['tiles']):
            sy = E['syl'][si]; a = eob((c - .95 - .025 * k) / .38) * (1 - back)
            ja = min(clamp((c - .95) / .1), 1 - clamp((c - 2.62) / .13))
            if ja <= 0: continue
            glyph(d, j, lerp(sy['cx'], E['pos'][k], a), lerp(sy['cy'], cy, a), font(E['F'], lerp(px * .5, E['ts'] * .74, a)), col(ja))
    else:                                             # 4 · 형광펜 밑줄 → 문구 타이핑 → 도장
        u = eo((c - 2.75) / .35); d.rectangle([kl - pad, cy - .05 * (kb - kt), lerp(kl - pad, kr + pad, u), kb + .07 * px], fill=LIME)
        d.text((E['x0'], E['base']), KW, font=F, fill=ink, anchor='ls')
        if E['line']:
            Fl = font('NSf6', 48); vis = E['line'][:max(0, int((c - 3.1) * 30))]; lx = 720 - Fl.getlength(E['line']) / 2
            d.text((lx, kb + 128), vis, font=Fl, fill=ink, anchor='ls')
        sa = clamp((c - 3.6) / .2)
        if sa > 0: glyph(d, E['sub'], 720, kb + 128 + (62 if E['line'] else 0) - 12, font('NS5', 24), col(sa, mut))
        k = c - 3.55
        if k >= 0:
            sc = keys([(0, 1.7), (.08, .93), (.14, 1.02), (.2, 1)], k)[0] * 175 / E['seal'].height
            sl = E['seal'].resize((max(2, int(E['seal'].width * sc)), max(2, int(E['seal'].height * sc))), Image.LANCZOS).rotate(6, Image.BICUBIC, expand=True)
            if k < .04: sl.putalpha(sl.getchannel('A').point(lambda v: int(v * k / .04)))
            sx_, sy_ = min(kr + 105, W - 95), kt + 28; im.paste(sl, (int(sx_ - sl.width / 2), int(sy_ - sl.height / 2)), sl)
    a = np.asarray(im).astype(np.float32)
    if 0 <= c - 3.55 < .09: a = np.roll(a, (3, -2), (0, 1))
    if c > 4.73: a = lerp(a, np.array(bg, np.float32), sstep(4.73, 5.08, c))
    return paper(a.astype(np.uint8)).astype(np.float32) if pr else a

# ───────── 사운드 (전부 합성. 88 BPM, 강박이 휩 프레임에 떨어진다) ─────────
SR = 48000
def synth_audio(path):
    N = int(15.5 * SR); mix = np.zeros((N, 2), np.float32); rs = np.random.RandomState(7)
    tt = lambda dur: np.arange(int(dur * SR), dtype=np.float32) / SR
    def add(t0, sig, db=0.0, pan=0.0):
        i = int(round(t0 * SR));
        if i < 0: sig = sig[-i:]; i = 0
        sig = sig[:max(0, N - i)] * 10 ** (db / 20)
        mix[i:i + len(sig), 0] += sig * (1 - max(pan, 0)); mix[i:i + len(sig), 1] += sig * (1 + min(pan, 0))
    hp = lambda x, k: x - np.convolve(x, np.ones(k, np.float32) / k, 'same')
    def kick(): t = tt(.32); return np.sin(2 * np.pi * (45 * t + (120 - 45) * .035 * (1 - np.exp(-t / .035)))) * np.exp(-t / .11)
    def snare(): t = tt(.2); return (hp(rs.randn(len(t)).astype(np.float32), 24) * .5 + np.sin(2 * np.pi * 185 * t) * .5) * np.exp(-t / .055)
    def hat(d=.018): t = tt(.07); return hp(rs.randn(len(t)).astype(np.float32), 5) * np.exp(-t / d) * .35
    def bass(fq, dur): t = tt(dur); return np.tanh(2.2 * np.sin(2 * np.pi * fq * t) + .5 * np.sin(4 * np.pi * fq * t)) * np.minimum(1, t / .01) * np.exp(-t / (dur * .6)) * .6
    def stab(fs, dur=.7):
        t = tt(dur); s = sum(np.sin(2 * np.pi * f * t) + .3 * np.sin(4 * np.pi * f * t) for f in fs); return s / len(fs) * np.exp(-t / .22) * .35
    B = 60 / 88; st = B / 4; hit = (WHIP - 1) / FPS; bgm = np.zeros(N, np.float32)
    def badd(t0, sig, g=1.0):
        i = int(round(t0 * SR))
        if 0 <= i < N: sig = sig[:N - i]; bgm[i:i + len(sig)] += sig * g
    user = D.get('bgm') and os.path.exists(D['bgm'])
    if user:                                              # 사용자 음악: bgm_hit(초) 지점이 휩 프레임에 오도록 놓는다
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(max(0, D['bgm_hit'] - hit)), '-i', D['bgm'], '-t', '15.5', '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True).stdout
        u = np.frombuffer(raw, np.float32); off = int(max(0, hit - D['bgm_hit']) * SR); u = u[:N - off]; bgm[off:off + len(u)] = u / (np.abs(u).max() + 1e-6) * .8
    else:
        for bar in range(-4, 2):
            t0 = hit + bar * 4 * B; full = bar >= 0; last = bar == -1
            for s_ in range(16):
                t = t0 + s_ * st
                if last and s_ >= 8: badd(t, hat(.01), .5 + s_ * .05); badd(t + st / 2, hat(.01), .4 + s_ * .05); continue
                if s_ in (0, 7, 10): badd(t, kick(), 1.0 if s_ == 0 else .8); badd(t, bass(55 if s_ != 7 else 65.4, st * (5 if s_ == 0 else 2.5)), 1.0 if full else .75)
                if s_ in (4, 12): badd(t, snare(), .8)
                if s_ % 2 == 0 or full: badd(t, hat(.03 if s_ % 4 == 2 else .015), 1.0 if s_ % 2 == 0 else .5)
                if s_ in (0, 6): badd(t, stab((220, 261.6, 329.6, 493.9) if bar % 2 == 0 else (174.6, 220, 261.6, 392)), 1.0 if full else .7)
        t = tt(1.6); badd(hit, np.sin(2 * np.pi * (38 * t + (80 - 38) * .09 * (1 - np.exp(-t / .09)))) * np.exp(-t / .5), 1.3)
        t = tt(2.2); badd(hit, hp(rs.randn(len(t)).astype(np.float32), 6) * np.exp(-t / .55), .35)
    tg = np.arange(N) / SR; env = np.minimum(1, tg / .03); fo = (tg - 10.717) / (13.067 - 10.717)
    env = env * np.where(fo < 0, 1, np.where(fo > 1, 0, (1 - np.clip(fo, 0, 1)) ** 1.6)); bgm *= env * .9
    mix[:, 0] += bgm; mix[:, 1] += bgm
    def click(): t = tt(.12); e = np.exp(-t / .012); e[-int(.06 * SR):] *= np.linspace(1, 0, int(.06 * SR)); return (hp(rs.randn(len(t)).astype(np.float32), 9) * .8 + np.sin(2 * np.pi * 2300 * t) * .4) * e
    for j, Ls in enumerate(STARTS): add((Ls - 1) / FPS, click(), -19 if j < 80 else -18, (.35 if j % 2 else -.35))
    for L in range(2, 9):                                 # 타이핑
        t = tt(.06); add((L - 1) / FPS + rs.rand() * .01, (np.sin(2 * np.pi * 320 * t) + hp(rs.randn(len(t)).astype(np.float32), 30) * .6) * np.exp(-t / .012), -13)
    t = tt(1.5); ph = 2 * np.pi * np.cumsum(180 * (10 ** (t / 1.5))) / SR; e = (t / 1.5) ** 3     # 라이저(피크가 휩에)
    add(hit - 1.5, (np.sin(ph) * .5 + hp(rs.randn(len(t)).astype(np.float32), 12) * .7) * e, -9)
    for k in range(11):                                   # 카운터 틱
        t = tt(.03); add(hit + (k + 1) / FPS, np.sin(2 * np.pi * (1500 + 90 * k) * t) * np.exp(-t / .008), -15)
    T0 = 244 / FPS
    if True:                                              # 엔딩 효과음
        nzf = lambda n_, k_: hp(rs.randn(n_).astype(np.float32), k_)
        def clack(g=1.0):
            t = tt(.07); return (nzf(len(t), 4) * .6 + np.sin(2 * np.pi * 2600 * t) * .5 + np.sin(2 * np.pi * 3900 * t) * .3) * np.exp(-t / .014) * g
        t = tt(.25); add(T0, nzf(len(t), 8) * (t / .25) ** 2.5, -14)                                   # 눌림 휙
        t = tt(.05); add(T0 + .42, np.sin(2 * np.pi * 900 * t) * np.exp(-t / .012), -15)               # 하이라이트 켜짐
        for k in range(END['N']): add(T0 + .95 + .025 * k + .3, clack(), -17, (k / max(END['N'] - 1, 1) - .5) * .8)   # 낱자 활자가 놓임
        for k in range(END['N']):
            t = tt(.03); add(T0 + 1.45 + END['step'] * k, np.sin(2 * np.pi * 1800 * t) * np.exp(-t / .006), -25)
        t = tt(.4); add(T0 + 2.3, nzf(len(t), 10) * (t / .4) ** 2, -20)                                # 다시 모임
        add(T0 + 2.71, clack(1.4), -10); t = tt(.32); add(T0 + 2.75, nzf(len(t), 40) * np.sin(np.pi * t / .32), -22)   # 형광펜
        for i, ch in enumerate(END['line']):
            if ch.strip(): t = tt(.03); add(T0 + 3.1 + i / 30, nzf(len(t), 3) * np.exp(-t / .005), -24)
        t = tt(.5); add(T0 + 3.55, (np.sin(2 * np.pi * 62 * t) * np.exp(-t / .09) + nzf(len(t), 6) * np.exp(-t / .012) * .8), -3)   # 도장 쿵
    raw = path + '.raw.wav'; w = wave.open(raw, 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(mix / (np.abs(mix).max() + 1e-6) * .7, -1, 1) * 32767).astype('<i2').tobytes()); w.close()
    def lufs(p):
        e = subprocess.run(['ffmpeg', '-hide_banner', '-i', p, '-af', 'ebur128', '-f', 'null', '-'], capture_output=True, text=True).stderr
        try: return float(e[e.rindex('I:') + 2:].split()[0])
        except Exception: return -14.0
    g = -14 - lufs(raw)
    for _ in range(3):                                    # 이득 → 리미터(천장 −3.9 dB) → 다시 재서 −14 LUFS 로 수렴
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', raw, '-af', 'volume=%.2fdB,alimiter=limit=0.64:attack=3:release=60:level=disabled' % g, '-ar', str(SR), path], check=True)
        d = -14 - lufs(path)
        if abs(d) < .25: break
        g += d
    os.remove(raw)

# ───────── 실행 ─────────
def frame(g, state):                                       # g = 1…372 (전체 프레임 번호)
    if g <= 20: return frame_intro(g)
    if g < WHIP: return frame_montage(g)
    if g <= 244: return frame_counter(g)
    return frame_ending(g - 245, state['last'])

def main():
    global KW, KS, HEADS, PASS, KIND, NPX, SPX, END
    if len(sys.argv) < 3: sys.exit(__doc__ or 'python3 montage.py config.json out.mp4')
    cfg = json.load(open(sys.argv[1], encoding='utf-8')); out = sys.argv[2]
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(D.get(k), dict): D[k].update(v)
        elif v not in (None, '', []): D[k] = v
    KW = D['keyword'].strip(); load_fonts(os.path.join(HERE, 'fonts'))
    D['outlets'] = (D['outlets'] * 28)[:max(28, len(D['outlets']))]
    def split(s):
        a, b = s.split('{K}', 1); return a, b
    HEADS = []
    for h in D['headlines']:
        l1, _, l2 = h.partition('|'); HEADS.append(split(l1) + (l2.strip(),))
    PASS = []
    for p in D.get('passages') or [h.replace('|', ' |') for h in D['headlines']]:
        ls = [x.strip() for x in p.split('|')]; k = next(i for i, x in enumerate(ls) if '{K}' in x)
        PASS.append((ls[:k],) + split(ls[k]) + (ls[k + 1:],))
    if len(HEADS) % 7 == 0: HEADS.append(HEADS[0])
    if len(PASS) % 5 == 0: PASS.append(PASS[0])
    if not D['counter']['subtitle']: D['counter']['subtitle'] = D.get('topic') or KW
    wk = font('NS7', 100).getlength(KW); KS = min(1.0, 600 / wk)                 # 긴 키워드는 전체를 줄인다
    r = rng(D['seed'] * 977 + 3); KIND = [r() < .06 + .8 * clamp((j / 113 - .3) / .7) ** 1.25 for j in range(114)]
    pd = D.get('photos')
    if pd and os.path.isdir(pd):
        for f in sorted(os.listdir(pd)):
            try: PHOTOS.append(Image.open(os.path.join(pd, f)).convert('RGB'))
            except Exception: pass
    PHOTOS.extend(mkphoto(k) for k in range(len(PHOTOS), 8))
    NPX = 434 * min(1.2, 1052 / num_layout(cval(11), 434)[2]); SPX = min(50, 1150 / max(font('NS7', 50).getlength(D['counter']['subtitle']) / 50 + 1.2 * len(D['counter']['subtitle']) / 50, 1))
    END = build_end(); state = {'last': None}
    def get(g):
        if g > 244 and state['last'] is None:
            state['last'] = Image.fromarray(np.clip(frame_counter(244), 0, 255).astype(np.uint8))
        return np.clip(frame(g, state), 0, 255).astype(np.uint8)
    base = os.path.splitext(out)[0]; os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if '--stills' in sys.argv:
        for g in sys.argv[sys.argv.index('--stills') + 1].split(','):
            Image.fromarray(get(int(g))).save('%s_f%03d.png' % (base, int(g)))
        return
    if '--sheet' in sys.argv:                                # 검수용: 컷 첫 프레임 30장을 한 장에
        gs = [1, 9, 16] + [STARTS[j] for j in range(0, 114, 5)] + [STARTS[112], STARTS[113], 219, 222, 226, 244]
        sh = Image.new('RGB', (6 * 360, 5 * 270), (0, 0, 0))
        for i, g in enumerate(gs[:30]): sh.paste(Image.fromarray(get(g)).resize((360, 270), Image.LANCZOS), (i % 6 * 360, i // 6 * 270))
        sh.save(base + '_sheet.jpg', quality=88); print('검수 시트:', base + '_sheet.jpg'); return
    wav = base + '.audio.wav'; synth_audio(wav)
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-', '-i', wav,
                           '-c:v', 'libx264', '-preset', 'medium', '-crf', '15', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '320k', '-t', '15.5', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for g in range(1, 373):
        ff.stdin.write(get(g).tobytes())
        if g % 62 == 0: print('렌더 %d/372' % g, flush=True)
    ff.stdin.close(); ff.wait(); os.remove(wav); print('완료:', out)

if __name__ == '__main__': main()
