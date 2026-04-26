      subroutine discom (idatmp,iaer,xmus,xmuv,phi,
     a                   taer55,taer55p,palt,
     a                 phirad,nt,mu,np,rm,gb,rp,
     a                   ftray,xlm1,xlm2)
      integer mu,np
      real rm(-mu:mu),rp(np),gb(-mu:mu)
      real ftray,xlm1(-mu:mu,np),xlm2(-mu:mu,np)
      real xmus,xmuv,phi
      real taer55,taer55p,palt,phirad,ext,ome,gasym,phase,roatm
      real dtdir,dtdif,utdir,utdif,sphal,wldis,trayl,traypl,s
      real wlinf,wlsup,phasel,pdgs,cgaus,pha,betal,wl,tray,trayp,taer
      real taerp,piza,tamoy,tamoyp,pizmoy,rorayl
      real roaero,romix,ddirtt,ddiftt,udirtt,udiftt,sphalbt,ddirtr
      real ddiftr,udirtr,udiftr,sphalbr,ddirta,ddifta,udirta,udifta
      real sphalba,coeff
      integer idatmp,iaer,nt,l,k
      common /sixs_aer/ext(10),ome(10),gasym(10),phase(10)
      common /sixs_disc/ roatm(3,10),dtdir(3,10),dtdif(3,10),
     a utdir(3,10),utdif(3,10),sphal(3,10),wldis(10),trayl(10),
     a traypl(10)
      common /sixs_ffu/s(1501),wlinf,wlsup
      common /sixs_sos/phasel(10,83),cgaus(83),pdgs(83)
      common /sixs_trunc/pha(83),betal(0:80)

c     computation of all scattering parameters at wavelength
c     discrete values,so we
c     can interpolate at any wavelength
 
      do 50 l=1,10
      wl=wldis(l)
      if ((wlsup.lt.wldis(1)).and.(l.le.2)) goto 30
      if (wlinf.gt.wldis(10).and.(l.ge.9)) goto 30
      if ((l.lt.10).and.(wldis(l).lt.wlinf).and.
     a     (wldis(l+1).lt.wlinf))
     a     goto 50
      if ((l.gt.1).and.(wldis(l).gt.wlsup).and.
     a      (wldis(l-1).gt.wlsup))
     a     goto 50
 
c     computation of rayleigh optical depth at wl
 
 30   call odrayl(wl,
     a           tray)
 
c plane case discussed here above
 
      if (idatmp.eq.0.or.idatmp.eq.4) then
	  if (idatmp.eq.4) trayp=tray
	  if (idatmp.eq.0) trayp=0.
	  else
          trayp=tray*ftray
      endif
      trayl(l)=tray
      traypl(l)=trayp
 
c     computation of aerosol optical properties at wl
 
      taer=taer55*ext(l)/ext(4)
      taerp=taer55p*ext(l)/ext(4)
      piza=ome(l)
c
c     computation of atmospheric reflectances
c               rorayl is rayleigh ref
c               roaero is aerosol ref
c     call plegen to decompose aerosol phase function in Betal
      if (iaer.ne.0) then
      do k=1,83
      pha(k)=phasel(l,k)
      enddo
      call trunca(coeff)
      endif
      tamoy=taer*(1.-piza*coeff)
      tamoyp=taerp*(1.-piza*coeff)
      pizmoy=piza*(1.-coeff)/(1.-piza*coeff)
c     do i=0,80
c     write(6,'(A5,I2.2,1X,E13.7)') 'betal',i,betal(i)
c     enddo
c
      call atmref(iaer,tamoy,tray,pizmoy,tamoyp,trayp,palt,
     a               phi,xmus,xmuv,
     s               phirad,nt,mu,np,rm,gb,rp,
     a                   rorayl,roaero,romix,xlm1,xlm2)
c     computation of scattering transmitances (direct and diffuse)
c     first time for rayleigh ,next total (rayleigh+aerosols)
      call scatra (tamoy,tamoyp,tray,trayp,pizmoy,
     a      palt,nt,mu,rm,gb,xmus,xmuv,
     a             ddirtt,ddiftt,udirtt,udiftt,sphalbt,
     a             ddirtr,ddiftr,udirtr,udiftr,sphalbr,
     a             ddirta,ddifta,udirta,udifta,sphalba)
      roatm(1,l)=rorayl
      roatm(2,l)=romix
      roatm(3,l)=roaero
      dtdir(1,l)=ddirtr
      dtdif(1,l)=ddiftr
      dtdir(2,l)=ddirtt
      dtdif(2,l)=ddiftt
      dtdir(3,l)=ddirta
      dtdif(3,l)=ddifta
      utdir(1,l)=udirtr
      utdif(1,l)=udiftr
      utdir(2,l)=udirtt
      utdif(2,l)=udiftt
      utdir(3,l)=udirta
      utdif(3,l)=udifta
      sphal(1,l)=sphalbr
      sphal(2,l)=sphalbt
      sphal(3,l)=sphalba
   50 continue
      return
      end
