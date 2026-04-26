"""
commons.py
----------
Shared global state replacing Fortran COMMON blocks.

Fortran COMMON blocks translated:
  /sixs_atm/      -> atm
  /sixs_del/      -> delta_sigma
  /sixs_aer/      -> aer
  /sixs_disc/     -> disc
  /sixs_ffu/      -> ffu
  /sixs_ier/      -> ier
  /sixs_trunc/    -> trunc
  /sixs_planesim/ -> planesim
"""

import numpy as np

# /sixs_atm/ z(34),p(34),t(34),wh(34),wo(34)
class _Atm:
    def __init__(self):
        self.z  = np.zeros(34, dtype=np.float32)
        self.p  = np.zeros(34, dtype=np.float32)
        self.t  = np.zeros(34, dtype=np.float32)
        self.wh = np.zeros(34, dtype=np.float32)
        self.wo = np.zeros(34, dtype=np.float32)

atm = _Atm()

# /sixs_del/ delta, sigma
class _Del:
    def __init__(self):
        self.delta = 0.0
        self.sigma = 0.0

delta_sigma = _Del()

# /sixs_aer/ ext(10),ome(10),gasym(10),phase(10)
class _Aer:
    def __init__(self):
        self.ext   = np.zeros(10, dtype=np.float32)
        self.ome   = np.zeros(10, dtype=np.float32)
        self.gasym = np.zeros(10, dtype=np.float32)
        self.phase = np.zeros(10, dtype=np.float32)

aer = _Aer()

# /sixs_disc/ roatm(3,10),dtdir(3,10),dtdif(3,10),
#             utdir(3,10),utdif(3,10),sphal(3,10),
#             wldis(10),trayl(10),traypl(10)
class _Disc:
    def __init__(self):
        self.roatm  = np.zeros((3, 10), dtype=np.float32)
        self.dtdir  = np.zeros((3, 10), dtype=np.float32)
        self.dtdif  = np.zeros((3, 10), dtype=np.float32)
        self.utdir  = np.zeros((3, 10), dtype=np.float32)
        self.utdif  = np.zeros((3, 10), dtype=np.float32)
        self.sphal  = np.zeros((3, 10), dtype=np.float32)
        self.wldis  = np.zeros(10, dtype=np.float32)
        self.trayl  = np.zeros(10, dtype=np.float32)
        self.traypl = np.zeros(10, dtype=np.float32)

disc = _Disc()

# /sixs_ffu/ s(1501),wlinf,wlsup
class _Ffu:
    def __init__(self):
        self.s     = np.zeros(1501, dtype=np.float32)
        self.wlinf = 0.0
        self.wlsup = 0.0

ffu = _Ffu()

# /sixs_ier/ iwr,ier
class _Ier:
    def __init__(self):
        self.iwr = 6     # stdout unit in Fortran
        self.ier = False

ier = _Ier()

# /sixs_trunc/ pha(1:83),betal(0:80)
class _Trunc:
    def __init__(self):
        self.pha   = np.zeros(83, dtype=np.float32)
        self.betal = np.zeros(81, dtype=np.float32)   # index 0..80

trunc = _Trunc()

# /sixs_planesim/ zpl(34),ppl(34),tpl(34),whpl(34),wopl(34)
class _PlaneSim:
    def __init__(self):
        self.zpl  = np.zeros(34, dtype=np.float32)
        self.ppl  = np.zeros(34, dtype=np.float32)
        self.tpl  = np.zeros(34, dtype=np.float32)
        self.whpl = np.zeros(34, dtype=np.float32)
        self.wopl = np.zeros(34, dtype=np.float32)

planesim = _PlaneSim()

# /mie_in/ rmax,rmin,icp,rn(10,4),ri(10,4),x1(4),x2(4),x3(4),
#           cij(4),irsunph,rsunph(50),nrsunph(50)
#
# Populated by sixs_main.py for iaer=8-11 (Mie size distributions):
#   iaer=8  log-normal:      rmin rmax icp / per mode: x1 x2 cij rn(10) ri(10)
#   iaer=9  modified gamma:  rmin rmax / x1 x2 x3 / rn(10) / ri(10)
#   iaer=10 Junge power-law: rmin rmax / x1 / rn(10) / ri(10)
#   iaer=11 sun photometer:  irsunph / per point: rsunph nrsunph / rn(10) / ri(10)
class _MieIn:
    def __init__(self):
        self.rmax     = 0.0
        self.rmin     = 0.0
        self.icp      = 1
        self.rn       = np.zeros((10, 4), dtype=np.float64)  # real refractive index
        self.ri       = np.zeros((10, 4), dtype=np.float64)  # imaginary refractive index
        self.x1       = np.zeros(4,  dtype=np.float64)   # distribution param 1
        self.x2       = np.zeros(4,  dtype=np.float64)   # distribution param 2
        self.x3       = np.zeros(4,  dtype=np.float64)   # distribution param 3
        self.cij      = np.zeros(4,  dtype=np.float64)   # component volume fractions
        self.irsunph  = 0
        self.rsunph   = np.zeros(50, dtype=np.float64)   # sun-photometer radii (µm)
        self.nrsunph  = np.zeros(50, dtype=np.float64)   # sun-photometer dV/dlogr

mie_in = _MieIn()
