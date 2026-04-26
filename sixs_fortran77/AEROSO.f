      subroutine aeroso (iaer,co,xmud,wldis,FILE)

      double precision cij(4),vi(4),nis,sumni,ni(4)
      real co(4),dd(4,10),ci(4),ex(4,10),sc(4,10),asy(4,10)
      real pha(5,10,83),sca(10),wldis(10)
      real ex2(1,10),sc2(1,10),asy2(1,10)
      real ex3(1,10),sc3(1,10),asy3(1,10)
      real ex4(1,10),sc4(1,10),asy4(1,10)
      real xmud,ext,ome,gasym,phase,ph,phasel,cgaus,pdgs
      real coef,sigm,pi
      integer i,j,k,l,j1,j2,iaer,icp
      character FILE*80
c sra basic components for aerosol model, extinction coefficients are 
c in km-1.
c     dust-like = 1
c     water-soluble = 2
c     oceanique = 3
c     soot = 4
 
      data vi /113.983516,113.983516d-06,5.1444150196,
     a          59.77353425d-06/
      data ni /54.734,1.86855d+06,276.05,1.80582d+06/
 
c     i: 1=dust-like 2=water-soluble 3=oceanic 4=soot
      data ((ex(i,j),sc(i,j),j=1,10),i=1,1) /
     a 0.1796674e-01,0.1126647e-01,0.1815135e-01,0.1168918e-01,
     a 0.1820247e-01,0.1180978e-01,0.1827016e-01,0.1196792e-01,
     a 0.1842182e-01,0.1232056e-01,0.1853081e-01,0.1256952e-01,
     a 0.1881427e-01,0.1319347e-01,0.1974608e-01,0.1520712e-01,
     a 0.1910712e-01,0.1531952e-01,0.1876025e-01,0.1546761e-01/
      data ((ex(i,j),sc(i,j),j=1,10),i=2,2) /
     a 0.7653460e-06,0.7377123e-06,0.6158538e-06,0.5939413e-06,
     a 0.5793444e-06,0.5587120e-06,0.5351736e-06,0.5125148e-06,
     a 0.4480091e-06,0.4289210e-06,0.3971033e-06,0.3772760e-06,
     a 0.2900993e-06,0.2648252e-06,0.1161433e-06,0.9331806e-07,
     a 0.3975192e-07,0.3345499e-07,0.1338443e-07,0.1201109e-07/
      data ((ex(i,j),sc(i,j),j=1,10),i=3,3) /
     a 0.3499458e-02,0.3499455e-02,0.3574996e-02,0.3574993e-02,
     a 0.3596592e-02,0.3596591e-02,0.3622467e-02,0.3622465e-02,
     a 0.3676341e-02,0.3676338e-02,0.3708866e-02,0.3708858e-02,
     a 0.3770822e-02,0.3770696e-02,0.3692255e-02,0.3677038e-02,
     a 0.3267943e-02,0.3233194e-02,0.2801670e-02,0.2728013e-02/
      data ((ex(i,j),sc(i,j),j=1,10),i=4,4) /
     a 0.8609083e-06,0.2299196e-06,0.6590103e-06,0.1519321e-06,
     a 0.6145787e-06,0.1350890e-06,0.5537643e-06,0.1155423e-06,
     a 0.4503008e-06,0.8200095e-07,0.3966041e-06,0.6469735e-07,
     a 0.2965532e-06,0.3610638e-07,0.1493927e-06,0.6227224e-08,
     a 0.1017134e-06,0.1779378e-08,0.6065031e-07,0.3050002e-09/
 
       data ((ex2(i,j),sc2(i,j),j=1,10),i=1,1) /
     A 0.4383631E+02,0.4028625E+02,0.4212415E+02,0.3904473E+02,        
     A 0.4157425E+02,0.3861470E+02,0.4085399E+02,0.3803645E+02,        
     A 0.3914040E+02,0.3661054E+02,0.3789763E+02,0.3554456E+02,        
     A 0.3467506E+02,0.3269951E+02,0.2459000E+02,0.2341019E+02,        
     A 0.1796726E+02,0.1715375E+02,0.1057569E+02,0.1009731E+02/        

       data ((ex3(i,j),sc3(i,j),j=1,10),i=1,1) /
     A 0.9539786E+05,0.9297790E+05,0.7530360E+05,0.7339717E+05,
     A 0.7021064E+05,0.6842549E+05,0.6421828E+05,0.6257180E+05,
     A 0.5243056E+05,0.5104987E+05,0.4557768E+05,0.4434877E+05,
     A 0.3193777E+05,0.3100621E+05,0.9637680E+04,0.9202678E+04,
     A 0.3610691E+04,0.3344476E+04,0.8105614E+03,0.6641915E+03/
 
       data ((ex4(i,j),sc4(i,j),j=1,10),i=1,1) /
     A .5427304E+08, .5427304E+08, .6198144E+08, .6198144E+08,
     A .6302432E+08, .6302432E+08, .6348947E+08, .6348947E+08,
     A .6146760E+08, .6146760E+08, .5817972E+08, .5817972E+08,
     A .4668909E+08, .4668909E+08, .1519062E+08, .1519062E+08,
     A .5133055E+07, .5133055E+07, .8998594E+06, .8998594E+06/
 
      data ((asy(i,j),j=1,10),i=1,4) /
     a 0.896,0.885,0.880,0.877,0.867,0.860,0.845,0.836,0.905,0.871,
     a 0.642,0.633,0.631,0.628,0.621,0.616,0.610,0.572,0.562,0.495,
     a 0.795,0.790,0.788,0.781,0.783,0.782,0.778,0.783,0.797,0.750,
     a 0.397,0.359,0.348,0.337,0.311,0.294,0.253,0.154,0.103,0.055/
 
      data ((asy2(i,j),j=1,10),i=1,1)/
     A 0.718,0.712,0.710,0.708,0.704,0.702,0.696,0.680,0.668,0.649/    
 
      data ((asy3(i,j),j=1,10),i=1,1)/
     A 0.704,0.690,0.686,0.680,0.667,0.659,0.637,0.541,0.437,0.241/    
 
      data ((asy4(i,j),j=1,10),i=1,1)/
     A .705, .744, .751, .757, .762, .759, .737, .586, .372, .139/
 
      common /sixs_aer/ ext(10),ome(10),gasym(10),phase(10)
      common /sixs_aerbas/ ph(10,83)
      common /sixs_sos/phasel(10,83),cgaus(83),pdgs(83)
c
c     optical properties of aerosol model computed from sra basic comp
      pi=4.*atan(1.)
      do 1 l=1,10
       ext(l)=0.
       sca(l)=0.
       if(l.eq.4.and.iaer.eq.0) ext(l)=1.
       ome(l)=0.
       gasym(l)=0.
       phase(l)=0.
       do 1 k=1,83
        phasel(l,k)=0.
    1 continue
 
      do 2 j=1,4
       ci(j)=co(j)
    2 continue
 
      if(iaer.eq.0) return

      do 7 k=1,82
      if((xmud.ge.cgaus(k)).and.(xmud.lt.cgaus(k+1))) go to 8
    7 continue
      return
    8 j1=k
      j2=j1+1
      coef=-(xmud-cgaus(j1))/(cgaus(j2)-cgaus(j1))

      if (iaer.eq.12) then
        open(10,file=FILE)
        read(10,*)
        do l=1,10
         read(10,'(8x,4(3x,f6.4,3x))')ext(l),sca(l),ome(l),gasym(l)
        enddo    
        read(10,'(///)')
        do k=1,83
         read(10,'(8x,10(1x,e10.4))')(phasel(l,k),l=1,10)
        enddo   
        close(10)
        do l=1,10
         phase(l)=phasel(l,j1)+coef*(phasel(l,j1)-phasel(l,j2))
        enddo
        return
      endif
c
      if (iaer.eq.5) then
        do k=1,10
        asy(1,k)=asy2(iaer-4,k)
        ex(1,k)=ex2(iaer-4,k)
        sc(1,k)=sc2(iaer-4,k)
        enddo
      endif
c
      if (iaer.eq.6) then
        do k=1,10
        asy(1,k)=asy3(iaer-5,k)
        ex(1,k)=ex3(iaer-5,k)
        sc(1,k)=sc3(iaer-5,k)
        enddo
      endif
c
      if (iaer.eq.7) then
        do k=1,10
        asy(1,k)=asy4(iaer-6,k)
        ex(1,k)=ex4(iaer-6,k)
        sc(1,k)=sc4(iaer-6,k)
        enddo
      endif
c
c
      if (iaer.ge.5.and.iaer.le.11) then
c calling a special aerosol model 
C     (background desert model...)
         if (iaer.eq.5) call bdm
C     (biomass burning model...)
         if (iaer.eq.6) call bbm
C     (stratospherique aerosol model...)
         if (iaer.eq.7) call stm
C     (user defined model from size distribution)
         if (iaer.ge.8.and.iaer.le.11) call mie(iaer,wldis,ex,sc,asy)

         do l=1,10
          dd(1,l)=ph(l,j1)+coef*(ph(l,j1)-ph(l,j2))
          do k=1,83
           pha(1,l,k)=ph(l,k)
          enddo
         enddo
         icp=1
         cij(1)=1.00
c for normalization of the extinction coefficient
         nis=1.d+00/ex(1,4)
      else
c calling each sra components
         icp=4
c  -dust
         call dust
         do l=1,10
         dd(1,l)=ph(l,j1)+coef*(ph(l,j1)-ph(l,j2))
         do k=1,83
         pha(1,l,k)=ph(l,k)
         enddo
         enddo
c  -water soluble
         call wate
         do l=1,10
         dd(2,l)=ph(l,j1)+coef*(ph(l,j1)-ph(l,j2))
         do k=1,83
         pha(2,l,k)=ph(l,k)
         enddo
         enddo
c  -oceanic type
         call ocea
         do l=1,10
         dd(3,l)=ph(l,j1)+coef*(ph(l,j1)-ph(l,j2))
         do  k=1,83
         pha(3,l,k)=ph(l,k)
         enddo
         enddo
c  - soot
         call soot
         do l=1,10
         dd(4,l)=ph(l,j1)+coef*(ph(l,j1)-ph(l,j2))
         do k=1,83
         pha(4,l,k)=ph(l,k)
         enddo
         enddo
c     summ of the ci/vi calculation
         sumni=0.
         sigm=0.
         do 3 i=1,4
    3    sigm=sigm+ci(i)/vi(i)
 
c     cij coefficients calculation
         do 4 j=1,4
         cij(j)=(ci(j)/vi(j)/sigm)
    4    sumni=sumni+cij(j)/ni(j)

c nis=1/Kext(550)
         nis=1.d+00/sumni
      endif
      
c     mixing parameters calculation
      do 5 l=1,10
      do 6 j=1,icp
      ext(l)=ex(j,l)*cij(j)+ext(l)
      sca(l)=sc(j,l)*cij(j)+sca(l)
      gasym(l)=sc(j,l)*cij(j)*asy(j,l)+gasym(l)
      phase(l)=sc(j,l)*cij(j)*dd(j,l)+phase(l)
      do 77 k=1,83
      phasel(l,k)=sc(j,l)*cij(j)*pha(j,l,k)+phasel(l,k)
   77 continue
    6 continue
      ome(l)=sca(l)/ext(l)
      gasym(l)=gasym(l)/sca(l)
      phase(l)=phase(l)/sca(l)
      do 78 k=1,83
      phasel(l,k)=phasel(l,k)/sca(l)
   78 continue
      ext(l)=ext(l)*nis
      sca(l)=sca(l)*nis
    5 continue
      if (iaer.ge.8.and.iaer.le.11) then
       open(10,file=FILE)
        write(10,'(3x,A5,1x,5(1x,A10,1x),1x,A10)')'Wlgth',
     s'Nor_Ext_Co','Nor_Sca_Co','Sg_Sca_Alb',
     s'Asymm_Para','Extinct_Co','Scatter_Co'
        do 79 l=1,10
         write(10,'(2x,f6.4,4(3x,f6.4,3x),2(2x,e10.4))')
     s wldis(l),ext(l),sca(l),ome(l),gasym(l),ext(l)/nis,sca(l)/nis
 79     continue
         write(10,'(//,T20,A16,/,3x,A4,1x,10(3x,f6.4,2x))')
     s   ' Phase Function ','TETA',(wldis(l),l=1,10)
        do 76 k=1,83
         write(10,'(2x,f6.2,10(1x,e10.4))')180.*acos(cgaus(k))/pi,
     s                 (phasel(l,k),l=1,10)
 76     continue
        close(10)
      endif
      return
      end
