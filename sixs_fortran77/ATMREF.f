      subroutine atmref (iaer,tamoy,trmoy,pizmoy,tamoyp,trmoyp,palt,
     s               phi,xmus,xmuv,
     s               phirad,nt,mu,np,rm,gb,rp,
     a                   rorayl,roaero,romix,xlm1,xlm2)
      integer mu,np
      real rm(-mu:mu),rp(np),gb(-mu:mu),xlm1(-mu:mu,np)
      real xlm2(-mu:mu,np)
      real tamoy,trmoy,pizmoy,tamoyp,trmoyp,palt
      real phi,xmus,xmuv,phirad
      real rorayl,roaero,romix,delta,sigma,tamol,tamolp
      integer iaer,nt
 
      common /sixs_del/ delta,sigma
c
c     atmospheric reflectances
c
      rorayl=0.
      roaero=0.
c
c     rayleigh reflectance 3 cases (satellite,plane,ground)
      if(palt.lt.900..and.palt.gt.0.0)then
        rm(-mu)=-xmuv
        rm(mu)=xmuv
        rm(0)=-xmus
        tamol=0.
        tamolp=0.
      call os(tamol,trmoy,pizmoy,tamolp,trmoyp,palt,
     s               phirad,nt,mu,np,rm,gb,rp,
     s                     xlm1)
        rorayl=xlm1(-mu,1)/xmus
        else
        if (palt.le.0.0) then
           rorayl=0.
           else
           call chand(phi,xmuv,xmus,trmoy,rorayl)
           endif
        endif
c
      if (iaer.eq.0) then
         romix=rorayl
         return
         endif
c
c     rayleigh+aerosol=romix,aerosol=roaero reflectance computed
c     using sucessive order of scattering method
c     3 cases: satellite,plane,ground
      if(palt.gt.0.0) then
        rm(-mu)=-xmuv
        rm(mu)=xmuv
        rm(0)=-xmus
        call os(tamoy,trmoy,pizmoy,tamoyp,trmoyp,palt,
     s               phirad,nt,mu,np,rm,gb,rp,
     s                     xlm2)
        romix=(xlm2(-mu,1)/xmus)
        tamol=0.
        tamolp=0.
        call os(tamoy,tamol,pizmoy,tamoyp,tamolp,palt,
     s               phirad,nt,mu,np,rm,gb,rp,
     s                     xlm2)
        roaero=(xlm2(-mu,1)/xmus)
        else
        roaero=0.
        romix=0.
        endif
      return
      end
