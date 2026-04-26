      subroutine specinterp(wl,taer55,taer55p,
     s     tamoy,tamoyp,pizmoy,pizmoyp)
      real wl,taer55,taer55p,tamoy,tamoyp,pizmoy,pizmoyp,roatm
      real dtdir,dtdif,utdir,utdif,sphal,wldis,trayl,traypl
      real ext,ome,gasym,phase,pha,betal,phasel,cgaus,pdgs,coef
      real wlinf,alphaa,betaa,tsca,coeff
      integer linf,ll,lsup,k
      common /sixs_disc/ roatm(3,10),dtdir(3,10),dtdif(3,10),
     s utdir(3,10),utdif(3,10),sphal(3,10),wldis(10),trayl(10),
     s traypl(10)
      common /sixs_aer/ext(10),ome(10),gasym(10),phase(10)
      common /sixs_trunc/pha(83),betal(0:80)
      common /sixs_sos/phasel(10,83),cgaus(83),pdgs(83)
      linf=1
      do 80 ll=1,9
      if(wl.ge.wldis(ll).and.wl.le.wldis(ll+1)) linf=ll
   80 continue
      if(wl.gt.wldis(10)) linf=9
      lsup=linf+1
      coef=alog(wldis(lsup)/wldis(linf))
      wlinf=wldis(linf)
      alphaa=alog(ext(lsup)*ome(lsup)/(ext(linf)*ome(linf)))/coef
      betaa=ext(linf)*ome(linf)/(wlinf**(alphaa))
      tsca=taer55*betaa*(wl**alphaa)/ext(4)
      alphaa=alog(ext(lsup)/(ext(linf)))/coef
      betaa=ext(linf)/(wlinf**(alphaa))
      tamoy=taer55*betaa*(wl**alphaa)/ext(4)
      tamoyp=taer55p*betaa*(wl**alphaa)/ext(4)
      pizmoy=tsca/tamoy
      pizmoyp=pizmoy
      do 81 k=1,83
      alphaa=alog(phasel(lsup,k)/phasel(linf,k))/coef
      betaa=phasel(linf,k)/(wlinf**(alphaa))
 81   pha(k)=betaa*(wl**alphaa)
      call trunca(coeff)
      tamoy=tamoy*(1.-pizmoy*coeff)
      tamoyp=tamoyp*(1.-pizmoyp*coeff)
      pizmoy=pizmoy*(1.-coeff)/(1.-pizmoy*coeff)
      return
      end
