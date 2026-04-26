      subroutine interp (iaer,idatmp,wl,taer55,taer55p,xmud,
     a                   romix,rorayl,roaero,phaa,phar,tsca,
     a                   tray,trayp,taer,taerp,dtott,utott,
     a                   astot,asray,asaer,
     a                   utotr,utota,dtotr,dtota)
 
      common /sixs_aer/ext(10),ome(10),gasym(10),phase(10)
      common /sixs_disc/ roatm(3,10),dtdir(3,10),dtdif(3,10),
     a utdir(3,10),utdif(3,10),sphal(3,10),wldis(10),trayl(10),
     a traypl(10)
      common /sixs_del/ delta,sigma
      Real wl,taer55,taer55p
      Real xmud,romix,rorayl,roaero,phaa,phar,tsca,tray
      Real trayp,taer,taerp,dtott,utott,astot,asray,asaer,utotr
      Real utota,dtotr,dtota,ext,ome,gasym,phase,roatm,dtdir
      Real dtdif,utdir,utdif,sphal,wldis,trayl,traypl,delta,sigma
      Real alphaa,betaa,alphar,betar,alphac,betac,coef,wlinf,d2
      Real drinf,drsup,dtinf,dtsup,dtotc,dainf,dasup,urinf,ursup
      Real utinf,utsup,utotc,uainf,uasup,arinf,arsup,atinf,atsup
      Real aainf,aasup
      Integer iaer,idatmp,linf,ll,lsup

 
c     that for the atmosphere :
c     the reflectances
c                     rayleigh                             = rorayl
c                     aerosols                             = roaero
c                     mixing                               = romix
c     the downward transmittances
c                     rayleigh                             = dtotr
c                     aerosols                             = dtota
c                     total                                = dtott
c     the upward transmittances
c                     rayleigh                             = utotr
c                     aerosols                             = utota
c                     total                                = utott
c     the spherical albedos
c                     rayleigh                             = asray
c                     aerosols                             = asaer
c                     total                                = astot
c     the optical thickness of total atmosphere
c                     rayleigh                             = tray
c                     aerosols                             = taer
c     the optical thickness of the atmosphere above the plane
c                     rayleigh                             = trayp
c                     aerosols                             = taerp
c     the tsca of the aerosols (god dammed it)
c                     total atmosphere                     = tsca
      
      linf=1
      do 81 ll=1,9
      if(wl.gt.wldis(ll).and.wl.le.wldis(ll+1)) linf=ll
   81 continue
      if(wl.gt.wldis(10)) linf=9
      lsup=linf+1
 
c     interpolation in function of wavelength for scattering
c     atmospheric functions from discrete values at wldis
 
      alphaa=0.
      betaa=0.
      alphar=0.
      betar=0.
      alphac=0.
      betac=0.
      phaa=0.
      roaero=0.
      dtota=1.
      utota=1.
      asaer=0.
      taer=0.
      taerp=0.
      coef=alog(wldis(lsup)/wldis(linf))
      wlinf=wldis(linf)
c
      if(iaer.eq.0) goto 1240
      alphaa=alog(phase(lsup)/phase(linf))/coef
      betaa=phase(linf)/(wlinf**(alphaa))
      phaa=betaa*(wl**alphaa)
 1240 d2=2.+delta
      phar=(2.*(1.-delta)/d2)*.75*(1.+xmud*xmud)+3.*delta/d2
      if (idatmp.eq.0) then
         betar=0.
         betaa=0.
         betac=0.
         goto 1234
      endif
      if(roatm(1,linf).lt..001) then
	rorayl=roatm(1,linf)+(roatm(1,lsup)-roatm(1,linf))
     s     *(wl-wldis(linf))/(wldis(lsup)-wldis(linf))
	else
        alphar=alog(roatm(1,lsup)/roatm(1,linf))/ coef
        betar=roatm(1,linf)/(wlinf**(alphar))
	rorayl=betar*(wl**alphar)
      endif
      if(roatm(2,linf).lt..001) then
        romix=roatm(2,linf)+(roatm(2,lsup)-roatm(2,linf))
     s     *(wl-wldis(linf))/(wldis(lsup)-wldis(linf))
        else
        alphac=alog(roatm(2,lsup)/roatm(2,linf))/coef
        betac=roatm(2,linf)/(wlinf**(alphac))
	romix=betac*(wl**alphac)
      endif
      if(iaer.eq.0) goto 1234
      if(roatm(3,linf).lt..001) then
	roaero=roatm(3,linf)+(roatm(3,lsup)-roatm(3,linf))
     s     *(wl-wldis(linf))/(wldis(lsup)-wldis(linf))
	else
        alphaa=alog(roatm(3,lsup)/roatm(3,linf))/coef
        betaa=roatm(3,linf)/(wlinf**(alphaa))
        roaero=betaa*(wl**alphaa)
      endif
 1234 continue
c
      alphar=alog(trayl(lsup)/trayl(linf))/coef
      betar=trayl(linf)/(wlinf**(alphar))
      tray=betar*(wl**alphar)
      if (idatmp.ne.0.) then
         alphar=alog(traypl(lsup)/traypl(linf))/coef
         betar=traypl(linf)/(wlinf**(alphar))
         trayp=betar*(wl**alphar)
         else
         trayp=0.
         endif
c
      if(iaer.eq.0) goto 1235
      alphaa=alog(ext(lsup)*ome(lsup)/(ext(linf)*ome(linf)))/coef
      betaa=ext(linf)*ome(linf)/(wlinf**(alphaa))
      tsca=taer55*betaa*(wl**alphaa)/ext(4)
      alphaa=alog(ext(lsup)/ext(linf))/coef
      betaa=ext(linf)/(wlinf**(alphaa))
      taerp=taer55p*betaa*(wl**alphaa)/ext(4)
      taer=taer55*betaa*(wl**alphaa)/ext(4)
c
 1235 drinf=dtdif(1,linf)+dtdir(1,linf)
      drsup=dtdif(1,lsup)+dtdir(1,lsup)
      alphar=alog(drsup/drinf)/coef
      betar=drinf/(wlinf**(alphar))
      dtotr=betar*(wl**alphar)
      dtinf=dtdif(2,linf)+dtdir(2,linf)
      dtsup=dtdif(2,lsup)+dtdir(2,lsup)
      alphac=alog((dtsup*drinf)/(dtinf*drsup))/coef
      betac=(dtinf/drinf)/(wlinf**(alphac))
      dtotc=betac*(wl**alphac)
      dainf=dtdif(3,linf)+dtdir(3,linf)
      dasup=dtdif(3,lsup)+dtdir(3,lsup)
      if(iaer.eq.0) goto 1236
      alphaa=alog(dasup/dainf)/coef
      betaa=dainf/(wlinf**(alphaa))
      dtota=betaa*(wl**alphaa)
 1236 dtott=dtotc*dtotr
      urinf=utdif(1,linf)+utdir(1,linf)
      ursup=utdif(1,lsup)+utdir(1,lsup)
      alphar=alog(ursup/urinf)/ coef
      betar=urinf/(wlinf**(alphar))
      utotr=betar*(wl**alphar)
      utinf=utdif(2,linf)+utdir(2,linf)
      utsup=utdif(2,lsup)+utdir(2,lsup)
      alphac=alog((utsup*urinf)/(utinf*ursup))/ coef
      betac=(utinf/urinf)/(wlinf**(alphac))
      utotc=betac*(wl**alphac)
      uainf=utdif(3,linf)+utdir(3,linf)
      uasup=utdif(3,lsup)+utdir(3,lsup)
      if(iaer.eq.0) goto 1237
      alphaa=alog(uasup/uainf)/ coef
      betaa=uainf/(wlinf**(alphaa))
      utota=betaa*(wl**alphaa)
 1237 utott=utotc*utotr
      arinf=sphal(1,linf)
      arsup=sphal(1,lsup)
      alphar=alog(arsup/arinf)/ coef
      betar=arinf/(wlinf**(alphar))
      asray=betar*(wl**alphar)
      atinf=sphal(2,linf)
      atsup=sphal(2,lsup)
      alphac=alog(atsup/atinf)/coef
      betac=atinf/(wlinf**(alphac))
      astot=betac*(wl**alphac)
      aainf=sphal(3,linf)
      aasup=sphal(3,lsup)
      if(iaer.eq.0) goto 1239
      alphaa=alog(aasup/aainf)/coef
      betaa=aainf/(wlinf**(alphaa))
      asaer=betaa*(wl**alphaa)
 1239 return
      end
