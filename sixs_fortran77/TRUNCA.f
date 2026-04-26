      subroutine trunca(coeff)
      real aa,x1,x2,a,x,rm,z1
      real cosang(80),weight(80),ptemp(83),pl(-1:81)
      real rmu(83),ga(83)
      integer nbmu,nang,k,j,kk,i
      real pha,betal,coeff
      common /sixs_trunc/pha(1:83),betal(0:80)
      nbmu=83
      nang=80
      do k=1,nbmu
      ptemp(k)=pha(k)
      enddo
      call gauss(-1.,1.,cosang,weight,nang)
      do 1 j=1,40
      rmu(j+1)=cosang(j)
      ga(j+1)=weight(j)
   1  continue
      rmu(1)=-1.0
      ga(1)=0.
      rmu(42)=0.
      ga(42)=0.
      do 2 j=41,80
      rmu(j+2)=cosang(j)
      ga(j+2)=weight(j)
   2  continue
      rmu(83)=1.0
      ga(83)=0.
      do 3 j=1,nbmu
      if((rmu(j).gt.0.8)) then
      go to 20
      else
      k=j-1
      endif
   3  continue
  20  continue
      do 4 j=1,nbmu
      if((rmu(j).gt.0.94)) then
      go to 21
      else
      kk=j-1
      endif
   4  continue
  21  continue
      aa=(alog10(pha(kk))-alog10(pha(k)))/
     a       (acos(rmu(kk))-acos(rmu(k)))
      x1=alog10(pha(kk))
      x2=acos(rmu(kk))
      do 5 j=kk+1,nbmu
      if(abs(rmu(j)-1.).le.1d-08) a=x1-aa*x2
      a=x1+aa*(acos(rmu(j))-x2)
      ptemp(j)=10**a
    5 continue
      do i=1,83
      pha(i)=ptemp(i)
      enddo
c
      do 10 k=0,80
      betal(k)=0.
   10 continue
      do 11 j=1,83
      x=pha(j)*ga(j)
      rm=rmu(j)
      pl(-1)=0.
      pl(0)=1.
      do 12 k=0,80
      pl(k+1)=((2*k+1.)*rm*pl(k)-k*pl(k-1))/(k+1.)
      betal(k)=betal(k)+x*pl(k)
  12  continue
  11  continue
      do 13 k=0,80
      betal(k)=(2*k+1.)*0.5*betal(k)
  13  continue
      z1=betal(0)
      coeff=1.-z1
      do k=0,80
      betal(k)=betal(k)/z1
      enddo
      return
      end
