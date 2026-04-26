"""
aktool.py
---------
Translated from AKTOOL.f

Kuusk MultiSpectral Reflectance Model (MSRM93) for vegetation canopy BRDF.
Reference: A. Kuusk, "A multispectral canopy reflectance model",
           Remote Sens. Environ., 50(2):75-82, 1994.
           Internet: andres@aai.ee

Parameters fed to akbrdf / akalbe:
    eei   – clumping parameter ε  (0 < ε ≤ 1; 0 = random, 1 = clumped)
    thmi  – mean leaf inclination angle (degrees)
    uli   – leaf area index
    sli   – leaf size parameter (relative leaf radius)
    rsl1i – soil brightness parameter
    wlmoy – wavelength (µm), range 0.404–2.500
    rnci  – leaf refractive index coefficient (≈1.4)
    cabi  – chlorophyll a+b content (µg/cm²)
    cwi   – leaf water equivalent thickness (cm)
    vaii  – effective number of elementary layers N (PROSPECT)
"""

import math
import numpy as np

# ── constants ────────────────────────────────────────────────────────────────
_PI   = 3.141592653589793
_PI12 = 1.570796326794895   # π/2
_DR   = 1.745329251994330e-2  # degrees → radians

# ── spectral tables (200 points, 404–800 nm step 4 nm, 801–2500 nm step 17 nm)
# refractive index of leaf material (Jacquemoud & Baret 1990)
_REFR = [
    1.5123,1.5094,1.5070,1.5050,1.5032,1.5019,1.5007,1.4997,1.4988,1.4980,1.4969,
    1.4959,1.4951,1.4943,1.4937,1.4930,1.4925,1.4920,1.4915,1.4910,1.4904,1.4899,
    1.4893,1.4887,1.4880,1.4873,1.4865,1.4856,1.4846,1.4836,1.4825,1.4813,1.4801,
    1.4788,1.4774,1.4761,1.4746,1.4732,1.4717,1.4701,1.4685,1.4670,1.4654,1.4639,
    1.4624,1.4609,1.4595,1.4582,1.4570,1.4559,1.4548,1.4538,1.4528,1.4519,1.4510,
    1.4502,1.4495,1.4489,1.4484,1.4480,1.4477,1.4474,1.4472,1.4470,1.4468,1.4467,
    1.4465,1.4463,1.4461,1.4458,1.4456,1.4453,1.4450,1.4447,1.4444,1.4440,1.4435,
    1.4430,1.4423,1.4417,1.4409,1.4402,1.4394,1.4387,1.4380,1.4374,1.4368,1.4363,
    1.4357,1.4352,1.4348,1.4345,1.4342,1.4341,1.4340,1.4340,1.4341,1.4342,1.4343,1.4345,
    1.4347,1.4348,1.4347,1.4345,1.4341,1.4336,1.4331,1.4324,1.4317,1.4308,1.4297,
    1.4284,1.4269,1.4253,1.4235,1.4216,1.4196,1.4176,1.4156,1.4137,1.4118,1.4100,
    1.4082,1.4065,1.4047,1.4029,1.4011,1.3993,1.3975,1.3958,1.3940,1.3923,1.3906,
    1.3888,1.3870,1.3851,1.3830,1.3808,1.3784,1.3758,1.3731,1.3703,1.3676,1.3648,
    1.3620,1.3592,1.3565,1.3537,1.3510,1.3484,1.3458,1.3433,1.3410,1.3388,1.3368,
    1.3350,1.3333,1.3317,1.3303,1.3289,1.3275,1.3263,1.3251,1.3239,1.3228,1.3217,
    1.3205,1.3194,1.3182,1.3169,1.3155,1.3140,1.3123,1.3105,1.3086,1.3066,1.3046,
    1.3026,1.3005,1.2985,1.2964,1.2944,1.2923,1.2902,1.2882,1.2863,1.2844,1.2826,
    1.2808,1.2793,1.2781,1.2765,1.2750,1.2738,1.2728,1.2719,1.2712,1.2708,1.2712,1.2736,
]

# brown/dry leaf absorption (albino leaf)
_KE = (
    [.1104,.0893,.0714,.0567,.0442,.0348,.0279,.0232,.0197,.0173,.0154,.0142,.0120,.0108,
    .0093,.0092,.0092,.0092,.0092,.0092,.0091,.0091,.0091,.0091,.0091,.0090,.0090,.0090,
    .0090,.0090,.0089,.0089,.0089,.0089,.0088,.0088,.0088,.0088,.0088,.0087,.0087,.0087,
    .0087,.0087,.0086,.0086,.0086,.0086,.0086,.0085,.0085,.0085,.0085,.0085,.0084,.0084,
    .0084,.0084,.0084,.0083,.0083,.0083,.0082,.0082,.0082,.0082,.0082,.0081,.0081,.0081,
    .0081,.0081,.0080,.0080,.0080,.0080,.0080,.0079,.0079,.0079,.0079,.0079,.0078,.0078,
    .0078,.0078,.0078,.0077,.0077,.0077,.0077,.0077,.0076,.0076,.0076,.0076,.0076,.0075,
    .0075,.0075,.0074,.0073,.0072,.0071,.0070,.0069,.0068,.0068,.0067,.0066,.0065,.0064,
    .0063,.0062,.0062,.0061,.0060,.0059,.0058,.0057,.0056,.0056,.0054,.0053,.0053,.0052,
    .0051,.0050,.0049,.0048,.0047,.0047,.0046,.0045,.0044,.0043,.0042,.0041,.0040,.0039,
    .0039,.0037,.0037,.0036,.0035,.0034,.0033,.0032,.0031,.0031,.0030,.0029,.0028,.0027,
    .0026,.0025,.0025,.0024,.0023,.0022,.0021,.0020,.0019,.0019,.0018,.0017,.0016,.0015,
    .0014,.0014,.0013,.0012,.0010,.0010,.0009,.0008,.0007,.0006,.0006,.0005,.0004,.0003,
    .0002,.0002,.0001] + [0.0]*15
)

# chlorophyll a+b specific absorption (µg/cm² units)
_KAB = [
    .04664,.04684,.04568,.04482,.04344,.04257,.04287,.04189,.04116,.03847,.03409,.03213,
    .03096,.03116,.03051,.03061,.02998,.02965,.02913,.02902,.02769,.02707,.02539,.02409,
    .02150,.01807,.01566,.01317,.01095,.00929,.00849,.00803,.00788,.00757,.00734,.00713,
    .00692,.00693,.00716,.00758,.00815,.00877,.00938,.00976,.01041,.01089,.01105,.01127,
    .01170,.01222,.01280,.01374,.01441,.01462,.01495,.01499,.01506,.01580,.01686,.01810,
    .01961,.02112,.02336,.02702,.02880,.02992,.03142,.03171,.02961,.02621,.02078,.01518,
    .01020,.00718,.00519,.00390,.00298,.00218,.00163,.00116,.00083,.00057,.00039,.00027,
    .00014,.00011,.00009,.00005,]+[0.0]*112

# leaf water specific absorption (cm units)
_KW = [0.0]*111 + [
    .100,.200,.278,.206,.253,.260,.313,.285,.653,.614,.769,.901,.872,.812,.733,.724,.855,
    .900,1.028,1.500,2.026,2.334,3.636,8.942,14.880,17.838,19.497,19.419,17.999,12.024,
    10.709,8.384,7.081,6.155,5.619,5.112,4.512,4.313,4.064,3.804,3.709,3.877,4.348,
    4.574,5.029,5.804,6.345,5.823,5.886,6.315,8.432,15.588,32.247,51.050,58.694,55.135,
    50.454,42.433,40.670,36.030,29.771,25.153,24.378,22.008,20.608,18.576,17.257,15.921,
    14.864,12.861,12.773,12.426,13.090,14.013,15.066,15.857,16.776,19.113,21.066,22.125,
    26.438,28.391,28.920,31.754,36.375,40.056,41.019,45.471,43.126,
]

# Price soil basis functions (1990)
_PHIS1 = [
    .088,.095,.102,.109,.116,.123,.130,.136,.143,.150,.157,.164,.171,.178,.185,.192,.199,
    .206,.213,.220,.227,.233,.240,.247,.254,.261,.268,.275,.282,.289,.295,.302,.309,.316,
    .326,.335,.345,.356,.366,.376,.386,.395,.404,.412,.421,.429,.436,.443,.450,.457,.464,
    .470,.476,.483,.489,.495,.502,.508,.514,.520,.526,.532,.538,.543,.549,.555,.561,.568,
    .574,.580,.587,.594,.601,.608,.615,.622,.629,.637,.644,.652,.659,.667,.674,.681,.689,
    .696,.702,.709,.716,.723,.729,.735,.742,.748,.754,.760,.766,.771,.777,.782,
    .802,.819,.832,.842,.854,.868,.883,.899,.917,.935,.954,.974,.993,1.012,1.030,1.047,
    1.063,1.078,1.091,1.102,1.111,1.118,1.126,1.137,1.150,1.163,1.176,1.187,1.192,1.188,
    1.177,1.159,1.134,1.090,.979,.830,.764,.744,.748,.777,.823,.878,.932,.983,1.026,1.062,
    1.091,1.115,1.133,1.147,1.156,1.161,1.162,1.158,1.149,1.132,1.109,1.087,1.072,1.056,
    1.035,.989,.886,.659,.456,.350,.323,.335,.361,.396,.438,.484,.530,.576,.622,.664,.705,
    .740,.768,.788,.800,.802,.796,.794,.797,.789,.779,.756,.725,.715,.675,.635,.585,.535,
    .485,.435,.385,.335,.285,.235,
]
_PHIS2 = [
    .249,.245,.241,.237,.232,.228,.222,.217,.211,.205,.199,.193,.186,.179,.171,.163,.155,
    .147,.139,.130,.121,.111,.102,.092,.081,.071,.060,.049,.038,.026,.014,.002,-.011,-.024,
    -.037,-.050,-.064,-.078,-.092,-.107,-.121,-.137,-.152,-.168,-.184,-.200,-.216,-.232,
    -.246,-.259,-.270,-.280,-.289,-.297,-.303,-.308,-.313,-.317,-.322,-.325,-.329,-.332,
    -.335,-.338,-.340,-.342,-.345,-.347,-.350,-.352,-.355,-.358,-.360,-.363,-.366,-.369,
    -.372,-.374,-.377,-.378,-.380,-.381,-.382,-.382,-.383,-.382,-.382,-.381,-.380,-.378,
    -.376,-.373,-.370,-.367,-.363,-.359,-.354,-.349,-.344,-.338,
    -.310,-.283,-.258,-.234,-.212,-.190,-.167,-.143,-.118,-.092,-.066,-.039,-.014,.011,
    .034,.057,.083,.114,.151,.192,.233,.272,.311,.348,.380,.407,.438,.476,.521,.570,.624,
    .674,.708,.766,.824,.853,.854,.852,.858,.881,.916,.947,.973,.997,1.017,1.036,1.052,
    1.067,1.082,1.095,1.107,1.119,1.131,1.142,1.154,1.166,1.175,1.179,1.178,1.172,1.162,
    1.148,1.083,.900,.678,.538,.499,.515,.552,.598,.653,.716,.777,.834,.886,.932,.973,
    1.007,1.036,1.058,1.075,1.086,1.091,1.091,1.086,1.076,1.060,1.039,1.012,.980,.943,
    .900,.852,.799,.740,.676,.606,.532,.451,.366,
]
_PHIS3 = [
    -.417,-.384,-.351,-.318,-.285,-.253,-.221,-.189,-.157,-.126,-.095,-.064,-.033,-.003,
    .027,.057,.087,.117,.146,.175,.204,.232,.260,.289,.316,.344,.371,.399,.425,.452,.478,
    .505,.525,.545,.566,.587,.606,.626,.652,.676,.699,.722,.744,.764,.784,.804,.822,.839,
    .856,.872,.886,.900,.913,.926,.937,.948,.957,.966,.974,.981,.988,.993,.998,1.002,1.006,
    1.009,1.012,1.014,1.016,1.017,1.018,1.018,1.018,1.017,1.016,1.014,1.012,1.010,1.007,
    1.003,.999,.995,.990,.984,.978,.972,.965,.957,.949,.941,.932,.923,.913,.902,.891,.880,
    .868,.855,.842,.829,
    .766,.694,.620,.550,.484,.421,.361,.303,.247,.190,.134,.079,.023,-.031,-.086,-.140,
    -.190,-.235,-.275,-.310,-.340,-.367,-.394,-.422,-.452,-.484,-.513,-.541,-.565,-.578,
    -.575,-.556,-.525,-.468,-.323,-.115,-.018,.002,-.003,-.029,-.076,-.142,-.211,-.274,
    -.333,-.386,-.432,-.471,-.503,-.528,-.544,-.551,-.549,-.538,-.517,-.491,-.463,-.436,
    -.419,-.417,-.401,-.348,-.216,.014,.160,.203,.209,.210,.207,.200,.189,.174,.155,.132,
    .105,.075,.043,.013,-.012,-.035,-.053,-.068,-.078,-.082,-.080,-.073,-.060,-.041,-.017,
    .006,.035,.065,.097,.125,.168,.180,.168,.125,.097,.065,
]
_PHIS4 = [
    .067,.077,.086,.094,.102,.111,.118,.126,.133,.140,.146,.152,.158,.164,.169,.174,.179,
    .184,.188,.192,.195,.198,.201,.204,.206,.208,.210,.212,.213,.214,.214,.214,.214,.214,
    .213,.212,.211,.210,.210,.209,.207,.205,.202,.198,.194,.189,.184,.179,.173,.167,.161,
    .155,.149,.143,.136,.130,.123,.116,.108,.101,.093,.085,.077,.068,.060,.051,.043,.034,
    .026,.018,.010,.002,-.006,-.014,-.022,-.030,-.037,-.045,-.052,-.060,-.067,-.074,-.081,
    -.087,-.093,-.098,-.103,-.108,-.112,-.116,-.120,-.123,-.126,-.129,-.132,-.134,-.136,
    -.138,-.140,-.141,
    -.147,-.152,-.158,-.166,-.170,-.165,-.157,-.151,-.144,-.128,-.104,-.078,-.049,-.009,
    .038,.082,.122,.169,.222,.272,.317,.364,.413,.469,.532,.591,.642,.694,.748,.790,.810,
    .817,.819,.740,.494,.215,.110,.125,.155,.204,.291,.408,.521,.627,.724,.811,.884,.940,
    .987,1.025,1.053,1.071,1.077,1.072,1.046,.996,.941,.892,.857,.842,.809,.713,.509,.055,
    -.236,-.324,-.336,-.320,-.308,-.294,-.275,-.248,-.205,-.144,-.094,-.048,.005,.058,.105,
    .132,.123,.079,.045,.024,.014,.018,.022,-.010,-.042,-.054,-.055,-.060,-.060,-.055,-.050,
    -.046,-.042,-.038,-.034,-.030,
]

# Gauss quadrature tables (dakg): hardcoded for nq=2,4,6,8,10,12,14,16,20,48
_DAKG = {
    2:  ([-.577350269189626,  .577350269189626],
         [1.0, 1.0]),
    4:  ([-.861136311594053, -.339981043584856, .339981043584856, .861136311594053],
         [.347854845137454, .652145154862546, .652145154862546, .347854845137454]),
    6:  ([-.932469514203152, -.661209386466265, -.238619186083197,
           .238619186083197,  .661209386466265,  .932469514203152],
         [.171324492379170, .360761573048139, .467913934572691,
          .467913934572691, .360761573048139, .171324492379170]),
    8:  ([-.960289856497536,-.796666477413627,-.525532409916329,-.183434642495650,
           .183434642495650, .525532409916329, .796666477413627, .960289856497536],
         [.101228536290376,.222381034453374,.313706645877887,.362683783378362,
          .362683783378362,.313706645877887,.222381034453374,.101228536290376]),
    10: ([-.973906528517172,-.865063366688985,-.679409568299024,-.433395394129247,-.148874338981631,
           .148874338981631, .433395394129247, .679409568299024, .865063366688985, .973906528517172],
         [.0666713443086881,.149451349150580,.219086362515982,.269266719309996,.295524224714753,
          .295524224714753, .269266719309996, .219086362515982, .149451349150580, .0666713443086881]),
    12: ([-.981560634246719,-.904117256370475,-.769902674194305,-.587317954286617,-.367831498998180,-.125233408511469,
           .125233408511469, .367831498998180, .587317954286617, .769902674194305, .904117256370475, .981560634246719],
         [.0471753363865118,.106939325995318,.160078328543346,.203167426723066,.233492536538355,.249147045813402,
          .249147045813402, .233492536538355, .203167426723066, .160078328543346, .106939325995318, .0471753363865118]),
    16: ([-.989400934991650,-.944575023073233,-.865631202387832,-.755404408355003,
          -.617876244402643,-.458016777657227,-.281603550779259,-.0950125098376374,
           .0950125098376374,.281603550779259, .458016777657227, .617876244402643,
           .755404408355003, .865631202387832, .944575023073233, .989400934991650],
         [.0271524594117541,.0622535239386479,.0951585116824928,.124628971255534,
          .149595988816577, .169156519395003, .182603415044924, .189450610455068,
          .189450610455068, .182603415044924, .169156519395003, .149595988816577,
          .124628971255534, .0951585116824928,.0622535239386479,.0271524594117541]),
    20: ([-.993128599185095,-.963971927277914,-.912234428251326,-.839116971822219,
          -.746331906460151,-.636053680726515,-.510867001950827,-.373706088715420,
          -.227785851141645,-.0765265211334973,
           .0765265211334973, .227785851141645, .373706088715420, .510867001950827,
           .636053680726515, .746331906460151, .839116971822219, .912234428251326,
           .963971927277914, .993128599185095],
         [.0176140071391521,.0406014298003869,.0626720483341091,.0832767415767047,
          .101930119817240, .118194531961518, .131688638449177, .142096109318382,
          .149172986472604, .152753387130726, .152753387130726, .149172986472604,
          .142096109318382, .131688638449177, .118194531961518, .101930119817240,
          .0832767415767047,.0626720483341091,.0406014298003869,.0176140071391521]),
}


def _dakg(nq):
    """Return Gauss quadrature nodes and weights for nq points (-1..1)."""
    if nq in _DAKG:
        return list(_DAKG[nq][0]), list(_DAKG[nq][1])
    raise ValueError(f"dakg: unsupported nq={nq}")


def _integr(xx):
    """(1 - exp(-xx)) / xx  with safe handling of small xx."""
    if xx < 1e-10:
        return 1.0
    return (1.0 - math.exp(-min(xx, 87.0))) / xx


# ── state: all Fortran COMMONs collected into a simple namespace ─────────────
class _State:
    __slots__ = [
        # /count/
        'jl','jj','lg','jg','lf','nnx','n1','n2','u1','u2','a1','a2',
        # /leafin/
        'nnl','vai','kk',
        # /leafout/
        'refl','tran',
        # /dat/ (read-only tables, set once)
        # /soildata/
        'rsl1','rsl2','rsl3','rsl4','th2','rsl','rsoil','rr1soil','rrsoil',
        # /aaa/
        'rrl','ttl','ul','sl','clmp','clmp1','bi','bd','bqint',
        # /ggg/
        'gr','gt','g','g1','th','sth','cth','th1','sth1','cth1',
        'phi','sp','cp','th22','st','ct','st1','ct1','t10','t11','e1','e2',
        's2','s3','ctg','ctg1','ctt1','stt1','calph','alp2','salp2','calp2',
        'alph','salph','alpp','difmy','difsig',
        # /cfresn/
        'rn','rk',
        # /ladak/
        'ee','thm','sthm','cthm',
        # /msrmdata/
        'th10','rncoef','cab','cw','bq',
        # NAG
        'inex',
        # tav
        '_teta','_ref','_tau',
    ]
    def __init__(self):
        for s in self.__slots__:
            object.__setattr__(self, s, 0.0)
        self.u1 = [0.0]*10; self.u2 = [0.0]*10
        self.a1 = [0.0]*10; self.a2 = [0.0]*10


_s = _State()


# ── subroutine: tav ──────────────────────────────────────────────────────────
def _tav(teta_deg, ref):
    """Transmittance through a dielectric interface (Stern 1964 / Allen 1973)."""
    eps  = 1e-6
    teta = teta_deg * _DR
    r2   = ref**2
    rp   = r2 + 1.0
    rm   = r2 - 1.0
    a    = (ref + 1.0)**2 / 2.0
    k    = -(r2 - 1.0)**2 / 4.0

    if abs(teta) <= eps:
        return 4.0 * ref / (ref + 1.0)**2

    ds = math.sin(teta)
    b1 = 0.0 if abs(teta - _PI12) <= eps else math.sqrt((ds**2 - rp/2.0)**2 + k)
    b2 = ds**2 - rp/2.0
    b  = b1 - b2

    ts  = (k**2/(6.0*b**3) + k/b - b/2.0) - (k**2/(6.0*a**3) + k/a - a/2.0)
    tp1 = -2.0*r2*(b - a)/rp**2
    tp2 = -2.0*r2*rp*math.log(b/a)/rm**2
    tp3 = r2*(1.0/b - 1.0/a)/2.0
    tp4 = 16.0*r2**2*(r2**2+1.0)*math.log((2.0*rp*b - rm**2)/(2.0*rp*a - rm**2))/(rp**3*rm**2)
    tp5 = 16.0*r2**3*(1.0/(2.0*rp*b - rm**2) - 1.0/(2.0*rp*a - rm**2))/rp**3
    tp  = tp1 + tp2 + tp3 + tp4 + tp5
    return (ts + tp) / (2.0 * ds**2)


# ── subroutine: s13aaf (exponential integral E1) ─────────────────────────────
def _s13aaf(k):
    """Exponential integral ∫(exp(-t)/t, t=k..∞)."""
    if k <= 0.0:
        return 0.0
    if k >= 85.0:
        return 0.0
    if k <= 4.0:
        x = 0.5*k - 1.0
        y = (((((((((((((((
            -3.60311230482612224e-13*x + 3.46348526554087424e-12)*x
            -2.99627399604128973e-11)*x + 2.57747807106988589e-10)*x
            -2.09330568435488303e-9)*x  + 1.59501329936987818e-8)*x
            -1.13717900285428895e-7)*x  + 7.55292885309152956e-7)*x
            -4.64980751480619431e-6)*x  + 2.63830365675408129e-5)*x
            -1.37089870978830576e-4)*x  + 6.47686503728103400e-4)*x
            -2.76060141343627983e-3)*x  + 1.05306034687449505e-2)*x
            -3.57191348753631956e-2)*x  + 1.07774527938978692e-1)*x
        y = (y*x + 8.64664716763387311e-1)*x + 7.42047691268006429e-1
        return y - math.log(k)
    else:
        x = 14.5/(k + 3.25) - 1.0
        y = (((((((((((((((-1.62806570868460749e-12*x
            -8.95400579318284288e-13)*x - 4.08352702838151578e-12)*x
            -1.45132988248537498e-11)*x - 8.35086918940757852e-11)*x
            -2.13638678953766289e-10)*x - 1.10302431467069770e-9)*x
            -3.67128915633455484e-9)*x  - 1.66980544304104726e-8)*x
            -6.11774386401295125e-8)*x  - 2.70306163610271497e-7)*x
            -1.05565006992891261e-6)*x  - 4.72090467203711484e-6)*x
            -1.95076375089955937e-5)*x  - 9.16450482931221453e-5)*x
            -4.05892130452128677e-4)*x  - 2.14213055000334718e-3
        y = ((y*x - 1.06374875116569657e-2)*x - 8.50699154984571871e-2)*x + 9.23755307807784058e-1
        return math.exp(-k)*y/k


# ── subroutine: leaf (PROSPECT) ───────────────────────────────────────────────
def _leaf():
    """Leaf reflectance and transmittance via PROSPECT model."""
    k  = _s.kk
    nn = _s.nnl
    N  = _s.vai

    if k <= 0.0:
        k_use = 1.0
    else:
        inex  = _s13aaf(k)
        k_use = (1.0 - k)*math.exp(-k) + k**2*inex

    t1 = _tav(90.0, nn)
    t2 = _tav(59.0, nn)
    x1 = 1.0 - t1
    x2 = t1**2 * k_use**2 * (nn**2 - t1)
    x3 = t1**2 * k_use * nn**2
    x4 = nn**4 - k_use**2 * (nn**2 - t1)**2
    x5 = t2/t1
    x6 = x5*(t1 - 1.0) + 1.0 - t2
    r  = x1 + x2/x4
    t  = x3/x4
    ra = x5*r + x6
    ta = x5*t

    delta = (t**2 - r**2 - 1.0)**2 - 4.0*r**2
    sq    = math.sqrt(max(delta, 0.0))
    alfa  = (1.0 + r**2 - t**2 + sq)/(2.0*r)
    beta  = (1.0 + r**2 - t**2 - sq)/(2.0*r)
    va    = alfa
    vb    = math.sqrt(max(beta*(alfa-r)/(alfa*(beta-r)), 0.0))
    n1    = N - 1.0

    def pw(base, exp_):
        if base <= 0.0:
            return 0.0
        return math.exp(exp_*math.log(base))

    vbn1 = pw(vb, n1); vbn1m = pw(vb, -n1)
    s1 = ra*(va*vbn1 - va**(-1)*vbn1m) + (ta*t - ra*r)*(vbn1 - vbn1m)
    s2 = ta*(va - va**(-1))
    s3 = va*vbn1 - va**(-1)*vbn1m - r*(vbn1 - vbn1m)
    _s.refl = s1/s3
    _s.tran = s2/s3


# ── subroutine: soilspec ─────────────────────────────────────────────────────
def _soilspec():
    jl = _s.jl
    _s.rsl = (_s.rsl1*_PHIS1[jl] + _s.rsl2*_PHIS2[jl]
             + _s.rsl3*_PHIS3[jl] + _s.rsl4*_PHIS4[jl])


# ── subroutine: soil ─────────────────────────────────────────────────────────
def _soil():
    th1 = _s.th1; th  = _s.th; th2 = _s.th2; cp = _s.cp; rsl = _s.rsl
    x   = th1**2
    a   = x*7.702 - 4.3
    b   = th1*7.363
    c   = 16.41 - x*4.3
    cts = 16.41 - th2**2*4.3
    x2  = rsl/cts
    _s.rsoil   = ((a*th + b*cp)*th + c)*x2
    _s.rr1soil = (0.7337*a + c)*x2
    _s.rrsoil  = 14.25*x2


# ── subroutine: glak (elliptical LAD) ────────────────────────────────────────
def _glak(th):
    """Elliptical leaf angle distribution projection function."""
    ee  = _s.ee; thm = _s.thm
    eps = 0.1
    if ee < eps:
        return 1.0
    ee = min(ee, 0.999999)
    u1 = ee*_s.cthm; u3 = ee*_s.sthm
    u2 = math.sqrt(max(0.0, 1.0 - u1**2))
    u4 = math.sqrt(max(0.0, 1.0 - u3**2))
    x  = math.log((u4 + u1)/(u2 - u3)) if (u2 - u3) > 1e-30 else 0.0
    x1 = math.atan2(u3, u4) - math.atan2(u1, u2)
    x2 = _s.sthm*x - _s.cthm*x1
    bb = ee/x2 if abs(x2) > 1e-30 else 1.0
    denom = 1.0 - (ee*math.cos(thm - th))**2
    return bb/math.sqrt(max(denom, 1e-30))


# ── subroutine: gmf (Fresnel leaf specular reflection) ───────────────────────
def _gmf():
    """Fresnel reflection for specular component."""
    ca  = _s.calp2; rn = _s.rn; rk = _s.rk
    x2  = ca**2
    ag  = 2.0*x2 - 1.0 + rn**2
    bg  = 1.0 + (ag - 2.0)*x2
    xy  = ag - x2
    cg  = 2.0*ca*math.sqrt(max(xy, 0.0))
    sa2 = 1.0 - x2
    denom = (ag + cg)
    if abs(denom) < 1e-30:
        return 0.0
    y = (ag - cg)*bg / ((bg + sa2*cg)*denom)
    yy = math.sqrt(max(sa2, 0.0)) / _PI12 / max(ca, 1e-10) * rk
    return math.exp(-yy)*y


# ── subroutine: gmd92 (G-functions and phase function) ───────────────────────
def _gmd92():
    """Phase function components and G-functions for elliptical LAD."""
    ee   = _s.ee
    alph = _s.alph; alp2 = _s.alp2; salph = _s.salph; alpp = _s.alpp
    calp2 = _s.calp2; salp2 = _s.salp2
    th22 = _s.th22; t10 = _s.t10; t11 = _s.t11
    st   = _s.st; ct = _s.ct; st1 = _s.st1; ct1 = _s.ct1
    ctt1 = _s.ctt1; stt1 = _s.stt1
    difmy = _s.difmy; difsig = _s.difsig
    phi = _s.phi; sp = _s.sp; cp = _s.cp
    cthm = _s.cthm; sthm = _s.sthm
    ctg  = _s.ctg; ctg1 = _s.ctg1
    calph = _s.calph
    pi13 = 0.1061032953; pi12 = 0.159154943; pi14 = 0.636619773
    pi4  = 2*_PI; eps5 = 0.01

    # spherical / isotropic (e=0) components
    gr0 = (salph + alpp*math.cos(alph))*pi13
    gt0 = (salph - alph*math.cos(alph))*pi13

    if ee < 0.4:
        _s.gr = gr0; _s.gt = gt0; _s.g = 0.5; _s.g1 = 0.5
        return

    # elliptical corrections
    sg = sg1 = sgmr = sgmt = 0.0

    def pp_from_fa_fb(fa, fb):
        x = fb - fa
        if x <= eps5:
            return 0.0
        if (pi4 - x) <= eps5:
            x1 = sthm**2
            pp = calph*x1 + ctt1*(2.0 - 3.0*x1)
            return pp*_PI
        sfa = math.sin(fa); sfb = math.sin(fb)
        cfa = math.cos(fa); cfb = math.cos(fb)
        pp = x*ctt1*cthm**2
        y1 = x + sfb*cfb - sfa*cfa
        xv = cfa - cfb
        y1 = y1*cp + sp*xv*(cfa + cfb)
        pp = pp + stt1*0.5*y1*sthm**2
        y1 = _s.s2*(sfb - sfa) + _s.s3*xv
        pp = pp + y1*sthm*cthm
        return pp

    if th22 >= t11:
        # case 61 (tl1=0, tl2=pi/2-th1): 130 branch (phi=alph)
        x1 = sthm**2
        pp = calph*x1 + ctt1*(2.0 - 3.0*x1)
        pp *= _PI
        if pp > 0: sgmr += pp
        if pp < 0: sgmt -= pp
        y1 = ct1*cthm; sg1 += abs(y1)
    elif th22 >= t10:
        # case 62
        x2 = cthm/sthm if sthm > 1e-30 else 1e30
        x  = -ctg1*x2 if abs(ctg1) < 1e30 else 0.0
        x1 = math.sqrt(max(0.0, 1.0 - x**2))
        fa = math.atan2(x1, x)
        fb = pi4 - fa
        pp = pp_from_fa_fb(fa, fb)
        if pp > 0: sgmr += pp
        if pp < 0: sgmt -= pp
        # case 73 (0 to pi)
        pp2 = alph*ctt1*cthm**2 + stt1*0.5*_PI*sthm**2*calph + \
              (_s.s2*salph + _s.s3*(math.cos(alph)-1))*sthm*cthm - pp
        if pp2 > 0: sgmr += pp2
        if pp2 < 0: sgmt -= pp2
    elif th22 >= 0:
        # case 63 (most general)
        x2 = cthm/sthm if sthm > 1e-30 else 1e30
        pts = []
        for ctg_v in [ctg1, ctg]:
            x  = -ctg_v*x2 if abs(ctg_v) < 1e30 else 0.0
            x1 = math.sqrt(max(0.0, 1.0 - x**2))
            pts.append(math.atan2(x1, x))
        fa2, fa3 = pts
        fb2 = pi4 - fa2
        fa4 = phi - fa3; 
        if fa4 < 0: fa4 += pi4
        fb4 = phi + fa3
        fs = sorted([fa4, fa2, fb2, fb4])
        fs = [fs[0] - pi4] + fs
        for ii in range(4):
            pp = pp_from_fa_fb(fs[ii], fs[ii+1])
            if pp > 0: sgmr += pp
            if pp < 0: sgmt -= pp
        x1 = ct*cthm; x2_ = st*sthm/x1 if abs(x1) > 1e-10 else 0.0
        xx = math.sqrt(max(0.0, x2_**2 - 1.0))
        x_ = math.atan2(1.0, xx)
        x_ = (x_ + xx)*x1
        sg += abs(x_*pi14)
    else:
        # case 50→52: tl1=0, tl2=pi/2-th (sg accumulates for view angle)
        x2 = ct1*cthm
        if abs(x2) > 1e-10:
            x1 = st1*sthm/x2
            xx = math.sqrt(max(0.0, x1**2 - 1.0))
            x_ = math.atan2(1.0, xx)
            x_ = (x_ + xx)*x2
            sg1 += abs(x_*pi14)
        y1 = ct*cthm; sg += abs(y1)

    gr1 = sgmr*pi12; gt1 = sgmt*pi12
    _s.gr = gr0 - 0.0102 + (1.742*difmy - 0.4557*difsig)*(gr1 - gr0)
    _s.gt = gt0 + 0.00653 + (0.2693*difmy + 5.821*difsig)*(gt1 - gt0)
    _s.g  = (2.653*difmy + 1.432*difsig)*(sg  - 0.5) + 0.50072
    _s.g1 = (2.653*difmy + 1.432*difsig)*(sg1 - 0.5) + 0.50072


# ── subroutine: difr92 (SAIL diffuse fluxes) ─────────────────────────────────
def _difr92():
    """Diffuse reflectance via SAIL two-stream equations."""
    rrl = _s.rrl; ttl = _s.ttl; ul = _s.ul
    g   = _s.g;   g1  = _s.g1;  cth = _s.cth; cth1 = _s.cth1
    difmy = _s.difmy; difsig = _s.difsig
    cthm = _s.cthm; rrsoil = _s.rrsoil

    rtp  = (rrl + ttl)/2.0
    ks   = g1*ul/max(cth1, 1e-10)
    ko   = g *ul/max(cth,  1e-10)
    gg   = (1.289*difmy - 1.816*difsig)*(cthm**2 - 1.0/3.0) + 0.31823
    bf   = (rrl - ttl)/2.0*ul*gg
    att  = (1.0 - rtp)*ul + bf
    sig  = rtp*ul + bf
    sb   = ks*rtp + bf; sf = ks*rtp - bf
    ub   = ko*rtp + bf; uf = ko*rtp - bf
    m    = math.sqrt(max(att**2 - sig**2, 0.0))
    h1   = (att + m)/sig if abs(sig) > 1e-30 else 1.0
    h2   = 1.0/h1
    denom1 = m**2 - ks**2
    c = (sf*sig - sb*(ks - att))/denom1 if abs(denom1) > 1e-30 else 0.0
    d = (sb*sig + sf*(ks + att))/denom1 if abs(denom1) > 1e-30 else 0.0
    epso = -d
    epss = (rrsoil*(d + 1.0) - c)*math.exp(-min(ks, 87.0))
    m11  = h1; m12 = h2
    m21  = (1.0 - rrsoil*h1)*math.exp(-min(m, 87.0))
    m22  = (1.0 - rrsoil*h2)*math.exp(min(m, 87.0))
    det  = m11*m22 - m12*m21
    if abs(det) < 1e-20:
        _s.bd = _s.rrsoil * math.exp(-min(ko,87.0))
        return
    a_   = (m22*epso - m12*epss)/det
    b_   = (-m21*epso + m11*epss)/det
    ep   = _integr(ko + m); em = _integr(ko - m); ek = _integr(ko + ks)
    gp   = a_*ep + b_*em + c*ek
    gm   = h1*a_*ep + h2*b_*em + d*ek
    ems  = h1*a_*math.exp(-min(m,87.0)) + h2*b_*math.exp(-min(-m,87.0)) + d*math.exp(-min(ks,87.0))
    _s.bd = uf*gp + ub*gm + rrsoil*ems*math.exp(-min(ko,87.0))


# ── subroutine: biz (single-scatter BRDF) ─────────────────────────────────────
def _biz():
    """Single-scattering bidirectional term."""
    eps = 1e-4; eps3 = 0.01

    _soil()
    if _s.ul <= eps:
        _s.bi = _s.rsoil
        return

    # Arrange sun (th) and view (th1) so that t10 = smaller, t11 = larger
    if _s.th1 >= _s.th:
        t10 = _s.th; t11 = _s.th1
        st  = _s.sth; ct  = _s.cth
        st1 = _s.sth1; ct1 = _s.cth1
        jj  = 0
    else:
        t10 = _s.th1; t11 = _s.th
        st  = _s.sth1; ct  = _s.cth1
        st1 = _s.sth;  ct1 = _s.cth
        jj  = 1

    _s.t10 = t10; _s.t11 = t11
    _s.st  = st;  _s.ct  = ct
    _s.st1 = st1; _s.ct1 = ct1
    _s.jj  = jj
    _s.ctt1  = ct*ct1; _s.stt1 = st*st1
    _s.calph = _s.stt1*_s.cp + _s.ctt1
    _s.calph = max(-1.0, min(1.0, _s.calph))
    _s.alph  = math.acos(_s.calph)
    _s.alp2  = _s.alph*0.5
    _s.e1    = st*ct1; _s.e2 = ct*st1
    _s.s2    = _s.e1*_s.cp + _s.e2
    _s.s3    = _s.e1*_s.sp
    _s.ctg   = ct/st   if abs(st)  > 1e-30 else 1e30
    _s.ctg1  = ct1/st1 if abs(st1) > 1e-30 else 1e30
    _s.salph = math.sin(_s.alph)
    _s.alpp  = _PI - _s.alph
    _s.salp2 = math.sin(_s.alp2); _s.calp2 = math.cos(_s.alp2)

    gf_val = _gmf()

    thp = 0.0
    if _s.ee > eps3:
        y4 = abs(_s.cth + _s.cth1)*0.5/max(_s.calp2, 1e-10)
        if y4 < 1.0:
            thp = math.acos(y4)
    glthp = _glak(thp)
    gf_val = gf_val * glthp * 0.125

    _gmd92()
    gammd = _s.gr*_s.rrl + _s.gt*_s.ttl

    # restore after possible jj swap
    _s.t11 = _s.th1; _s.st = _s.sth; _s.st1 = _s.sth1
    _s.ct  = _s.cth; _s.ct1 = _s.cth1; _s.t10 = _s.th
    if jj == 1:
        _s.g, _s.g1 = _s.g1, _s.g

    gg   = _s.g*_s.g1
    g_c  = _s.g*_s.clmp; g1_c = _s.g1*_s.clmp1
    gg1  = g_c*_s.ct1 + g1_c*_s.cth
    sct  = math.sqrt(max(_s.ctt1, 0.0))
    alpd = _s.alp2/_s.sl if _s.sl > 1e-10 else 0.0
    bam  = alpd*sct/_s.ul if _s.ul > 1e-10 else 0.0

    if _s.ctt1 > eps:
        gma  = alpd/sct
        ulg  = gg1/_s.ctt1*_s.ul
    else:
        gma = 0.0; ulg = _s.ul

    ulg1 = ulg*0.5
    xx1  = ulg + gma

    _MAX_ARG = 87.0
    if xx1 > _MAX_ARG or _s.ctt1 <= eps:
        easte = easte2 = easte4 = bs1 = 0.0
    else:
        easte  = math.exp(-min(ulg, _MAX_ARG))
        easte2 = math.exp(-min(ulg1 + gma, _MAX_ARG))
        easte4 = math.exp(-min(ulg + gma, _MAX_ARG))
        bs1    = (easte + easte2 - easte4)*_s.rsoil

    xx1  = (1.0 - easte)/max(gg1, 1e-30)
    gg1h = gg1*0.5 + bam
    if gg1h < 1e-30 or gg1 + bam < 1e-30:
        xx2 = 0.0
    else:
        xx2  = (1.0 - easte2)/gg1h - (1.0 - easte4)/(gg1 + bam)
    bc1d  = xx1*gammd
    bc1hs = xx2*(gammd + gf_val)
    bcsp  = xx1*gf_val
    result = bc1d + bcsp + bc1hs + bs1
    # Sanity: reflectance must be in [0, 1]
    _s.bi = max(0.0, min(1.0, result))


# ── subroutine: msrm ─────────────────────────────────────────────────────────
def _msrm():
    """Compute total canopy BRDF (single + diffuse scattering)."""
    rrls = _s.rrl
    _biz()
    _s.rrl = _s.refl
    _difr92()
    _s.rrl = rrls
    _s.bq  = _s.bi + _s.bd


# ── subroutine: akd (hemispherical integration for albedo) ────────────────────
def _akd():
    """Integrate msrm over view hemisphere for albedo calculation."""
    eps = 0.005; pi1 = _PI12
    n1 = _s.n1; n2 = _s.n2
    u1 = _s.u1; u2 = _s.u2; a1 = _s.a1; a2 = _s.a2

    _s.bqint = 0.0

    if _s.th <= eps:
        _s.phi = 0.0; _s.sp = 0.0; _s.cp = 1.0
        for i2 in range(n2):
            _s.th1  = (1.0 - u2[i2])*pi1
            _s.sth1 = math.sin(_s.th1); _s.cth1 = math.cos(_s.th1)
            rrls = _s.rrl
            _biz()
            _s.rrl = _s.refl
            _difr92()
            _s.rrl = rrls
            _s.bqint += a2[i2]*(_s.bi + _s.bd)*_s.sth1*_s.cth1
        _s.bqint *= _PI
        return

    # Precompute quadrature angles
    tt3 = [(u1[i]*_s.th)                      for i in range(n1)]
    tt2 = [(u2[i]*(_s.th - pi1) + pi1)        for i in range(n2)]

    for j in range(n1):
        _s.phi  = (1.0 - u1[j])*_PI
        _s.sp   = math.sin(_s.phi); _s.cp = math.cos(_s.phi)
        bd1 = bd2 = 0.0
        for i1 in range(n1):
            _s.th1  = tt3[i1]
            _s.sth1 = math.sin(_s.th1); _s.cth1 = math.cos(_s.th1)
            rrls = _s.rrl; _biz(); _s.rrl = _s.refl; _difr92(); _s.rrl = rrls
            bd1 += a1[i1]*(_s.bi + _s.bd)*_s.sth1*_s.cth1
        for i2 in range(n2):
            _s.th1  = tt2[i2]
            _s.sth1 = math.sin(_s.th1); _s.cth1 = math.cos(_s.th1)
            rrls = _s.rrl; _biz(); _s.rrl = _s.refl; _difr92(); _s.rrl = rrls
            bd2 += a2[i2]*(_s.bi + _s.bd)*_s.sth1*_s.cth1
        _s.bqint += ((pi1 - _s.th)*bd2 + _s.th*bd1)*a1[j]
    _s.bqint = _s.bqint * 2.0


# ── shared setup (called by both akbrdf and akalbe) ──────────────────────────
def _setup(eei, thmi, uli, sli, rsl1i, wlmoy, rnci, cabi, cwi, vaii):
    """Initialise state from input parameters."""
    dr = _DR
    clz = 0.9; clx = 0.1

    _s.ee   = eei
    _s.thm  = thmi*dr
    _s.ul   = uli
    _s.sl   = sli
    _s.rsl1 = rsl1i
    _s.rsl2 = -0.48*rsl1i + 0.0862
    _s.rsl3 = 0.0; _s.rsl4 = 0.0
    _s.th2  = 45.0*dr

    rlambda = wlmoy*1000.0
    if rlambda <= 800.0:
        jl = round((rlambda - 400.0)/4.0)
    else:
        jl = round((rlambda - 800.0)/17.0) + 100
    jl = max(0, min(199, jl))
    _s.jl = jl

    _s.rncoef = rnci
    _s.cab = cabi; _s.cw = cwi; _s.vai = vaii
    _s.nnl = _REFR[jl]
    _s.kk  = _KE[jl] + cabi*_KAB[jl] + cwi*_KW[jl]

    _leaf()
    _s.rn  = rnci*_s.nnl
    _s.rrl = _s.refl - ((1.0 - _s.rn)/(1.0 + _s.rn))**2
    _s.ttl = _s.tran

    _soilspec()
    _s.cthm = math.cos(_s.thm); _s.sthm = math.sin(_s.thm)
    _s.th22 = _PI12 - _s.thm
    eps4 = 1e-3
    if abs(_s.th22) < eps4:
        _s.th22 = 0.0

    eln = -math.log(max(1.0 - eei, 1e-10))
    _s.difmy  = abs(0.059*eln*(_s.thm - 1.02) + 0.02)
    _s.difsig = abs(0.01771 - 0.0216*eln*(_s.thm - 0.256))
    _s.rk = 0.0  # default: no hair


def _clamp(xx, sl, clz, clx):
    eps = 1e-5
    tgt = math.tan(xx)
    arg = tgt*clx/sl if sl > 1e-10 else 0.0
    return clz if arg < eps else 1.0 - (1.0 - clz)*_integr(arg)


# ── public interface: akbrdf ─────────────────────────────────────────────────
def akbrdf(eei, thmi, uli, sli, rsl1i, wlmoy, rnci, cabi, cwi, vaii, mu, np_, rm, rp):
    """
    Kuusk canopy BRDF at Gauss quadrature angles.

    Parameters
    ----------
    eei   : float – clumping (ε); 0=random, 1=clumped
    thmi  : float – mean leaf inclination (degrees)
    uli   : float – leaf area index
    sli   : float – leaf size parameter
    rsl1i : float – soil brightness
    wlmoy : float – wavelength (µm), 0.404–2.500
    rnci  : float – leaf refractive index coefficient (~1.4)
    cabi  : float – chlorophyll a+b (µg/cm²)
    cwi   : float – leaf water thickness (cm)
    vaii  : float – number of leaf layers N (PROSPECT)
    mu, np_, rm, rp : quadrature arrays (same convention as other BRDF models)

    Returns
    -------
    brdfint : 2-D array, shape (2*mu+1, np_)
    """
    clz = 0.9; clx = 0.1; eps = 1e-5; pir = math.pi

    _setup(eei, thmi, uli, sli, rsl1i, wlmoy, rnci, cabi, cwi, vaii)

    def ji(k): return k + mu
    brdfint = np.zeros((2*mu+1, np_))

    mu1    = rm[ji(0)]          # = -xmus (rm(0) in Fortran)
    th10_0 = math.acos(max(-1.0, min(1.0, mu1)))  # solar zenith
    rrls0  = _s.rrl

    for k in range(np_):
        for j in range(1, mu+1):
            mu2 = rm[ji(j)]
            fi  = rm[ji(-mu)] if j == mu else rp[k] + rm[ji(-mu)]
            if fi < 0:   fi += 2*pir
            if fi > 2*pir: fi -= 2*pir
            if fi > pir: fi = 2*pir - fi  # symmetry

            th10 = math.acos(max(-1.0, min(1.0, mu1)))
            _s.th10 = th10
            _s.clmp1 = _clamp(th10, _s.sl, clz, clx)

            _s.phi  = fi
            _s.sp   = math.sin(fi); _s.cp = math.cos(fi)
            _s.th1  = th10
            _s.sth1 = math.sin(th10); _s.cth1 = math.cos(th10)

            th = math.acos(max(-1.0, min(1.0, mu2)))
            _s.clmp = _clamp(th, _s.sl, clz, clx)
            _s.th   = th
            _s.sth  = math.sin(th); _s.cth = math.cos(th)

            _s.rrl = rrls0
            _msrm()
            brdfint[ji(j), k] = max(0.0, _s.bq)

    _s.rrl = rrls0
    return brdfint


# ── public interface: akalbe ─────────────────────────────────────────────────
def akalbe(eei, thmi, uli, sli, rsl1i, wlmoy, rnci, cabi, cwi, vaii):
    """
    Kuusk canopy hemispherical albedo.

    Parameters
    ----------
    (same as akbrdf minus mu, np_, rm, rp)

    Returns
    -------
    albbrdf : float – hemispherical (spherical) albedo
    """
    _setup(eei, thmi, uli, sli, rsl1i, wlmoy, rnci, cabi, cwi, vaii)

    n2 = 8; n1 = 6
    _s.n1 = n1; _s.n2 = n2

    uu, aa = _dakg(n2*2)
    n_full = n2*2  # total points
    # Fortran: u2(i)=uu(ng-i), i=1..n2 → Python: uu[n_full-i] for i=1..n2
    for i in range(n2):
        _s.a2[i] = aa[i]
        _s.u2[i] = uu[n_full - 1 - i]

    uu, aa = _dakg(n1*2)
    n_full = n1*2
    for i in range(n1):
        _s.a1[i] = aa[i]
        _s.u1[i] = uu[n_full - 1 - i]

    rrls0 = _s.rrl
    bdd   = 0.0
    for i2 in range(n2):
        _s.th  = (1.0 - _s.u2[i2])*_PI12
        _s.sth = math.sin(_s.th); _s.cth = math.cos(_s.th)
        _s.rrl = rrls0
        _akd()
        bdd += _s.a2[i2]*_s.bqint*_s.sth*_s.cth

    _s.rrl = rrls0
    return float(bdd*_PI)
