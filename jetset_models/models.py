from jetset.cosmo_tools import Cosmo
from astropy import units as u
import numpy as np
from jetset.base_model import Model
from jetset.model_manager import FitModel
from jetset.model_parameters import ModelParameterArray, ModelParameter
from jetset.analytical_model import AnalyticalParameter
from jetset.spectral_shapes import SED
from jetset.jet_model import Jet
from jetset.plot_sedfit import PlotSED

from astropy import units as u
from astropy import constants as const






class PlcModel(Model):
    """
    Class to handle power-law cut-off model
    """
    
    def __init__(self,nu_size=100,cosmo=None,name='plc',**keywords):
       
        
        super(PlcModel,self).__init__(  **keywords)
        self.name=name
        self.parameters = ModelParameterArray()      
       
        self.parameters.add_par(AnalyticalParameter(self,name='Ec',par_type='',val=1,val_min=0.,val_max=None,units='keV'))
        #self.parameters.add_par(AnalyticalParameter(self,name='E_ref',par_type='',val=1,val_min=0.,val_max=None,units='keV'))
        self.parameters.add_par(AnalyticalParameter(self,name='Gamma',par_type='',val=1,val_min=-10,val_max=10,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='K',par_type='',val=1,val_min=0,val_max=None,units='keV-1 cm-2'))
        self.parameters.add_par(AnalyticalParameter(self,name='alpha',par_type='',val=1,val_min=-10.,val_max=None,units=''))

        self.SED = SED(name=self.model_type)
        if cosmo is not None:
            self.cosmo=cosmo
        else:
            self.cosmo=cosmo()

        self.DL=self.cosmo.get_DL_cm()
        
        self._nu_to_keV=(1.0*u.Unit('Hz')).to(1000.0*u.eV,equivalencies=u.spectral()).value
        self._keV_to_erg=(1.0*u.Unit('keV').to('erg'))
    


    def lin_func(self,nu):
        E_keV=nu*self._nu_to_keV
        y=np.power(E_keV,-self.parameters.Gamma.val)*np.exp(-(E_keV/self.parameters.Ec.val)**self.parameters.alpha.val)
        return y*self.parameters.K.val*E_keV*E_keV*self._keV_to_erg
        
        #return (1.0/self.eta)*np.power(self.R0,((nu-self.t_0)/self.tau))
    


class CompModel(Model):
    """
    Class to handle function for comptonization model
    """
    
    def __init__(self,nu_size=100,cosmo=None,name='comptonization',**keywords):
        """
        https://arxiv.org/abs/0809.3255v2
        """
        
        super(CompModel,self).__init__(  **keywords)
        self.name=name
        self.parameters = ModelParameterArray()      
       
        self.parameters.add_par(AnalyticalParameter(self,name='Eb',par_type='',val=1,val_min=0,val_max=None,units='keV'))        
        self.parameters.add_par(AnalyticalParameter(self,name='Ec',par_type='',val=1,val_min=0.,val_max=None,units='keV'))       
        self.parameters.add_par(AnalyticalParameter(self,name='Gamma',par_type='',val=1,val_min=-10.,val_max=None,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='alpha',par_type='',val=1,val_min=-10.,val_max=None,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='K',par_type='',val=1,val_min=0,val_max=None,units='keV-1 cm-2'))  
        self.SED = SED(name=self.model_type)
        if cosmo is not None:
            self.cosmo=cosmo
        else:
            self.cosmo=cosmo()

        self.DL=self.cosmo.get_DL_cm()
        
        self._nu_to_keV=(1.0*u.Unit('Hz')).to(1000.0*u.eV,equivalencies=u.spectral()).value
        self._keV_to_erg=(1.0*u.Unit('keV').to('erg'))
    
    def lin_func(self,nu):
        E_keV=nu*self._nu_to_keV
        a=np.power((E_keV/self.parameters.Eb.val),-3)+ np.power((E_keV/self.parameters.Eb.val),self.parameters.Gamma.val-2)
        y=np.exp(-(E_keV/self.parameters.Ec.val)**self.parameters.alpha.val)*np.power(self.parameters.Eb.val,2-self.parameters.Gamma.val)/a
        y=y/(E_keV*E_keV)
        comp=self.parameters.K.val*y*E_keV*E_keV*self._keV_to_erg
        return comp



class DiskIrrComp(Model):
    """
    Class to handle function for irradiated disk and comptonization model
    """
    
    def __init__(self,nu_size=100,cosmo=None,name='DiskIrrComp',**keywords):
        """
        https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2966.2008.14166.x
        https://arxiv.org/abs/0809.3255v2
        """
        
        super(DiskIrrComp,self).__init__(  **keywords)
        self.name=name
        self.parameters = ModelParameterArray()      
       
        self.parameters.add_par(AnalyticalParameter(self,name='T_Disk',par_type='inner disk T.',val=1E5,val_min=0,val_max=None,units='k'))
        self.parameters.add_par(AnalyticalParameter(self,name='L_Disk',par_type='bolobmetric disk lum.',val=1E36,val_min=0,val_max=None,units='erg s-1'))
        self.parameters.add_par(AnalyticalParameter(self,name='theta',par_type='disk incl.',val=0,val_min=0,val_max=90,units='deg'))
        self.parameters.add_par(AnalyticalParameter(self,name='r_out',par_type='disk r_out (r_in)',val=100,val_min=1.,val_max=None,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='r_irr',par_type='disk r_irr (r_in)',val=1.1,val_min=1.,val_max=None,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='Gamma_Comp',par_type='Comp. index',val=1,val_min=-10,val_max=10,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='alpha_Comp',par_type='Comp. exp index',val=1,val_min=-10.,val_max=None,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='Eb_Comp',par_type='Comp. low-en cutoff',val=1,val_min=0,val_max=None,units='keV'))        
        self.parameters.add_par(AnalyticalParameter(self,name='Ec_Comp',par_type='Comp. high-en cutoff',val=1,val_min=0.,val_max=None,units='keV')) 
        self.parameters.add_par(AnalyticalParameter(self,name='L_Comp_ratio',par_type='L_Comp/L_Disk',val=0.1,val_min=0,val_max=None,units=''))   
        self.parameters.add_par(AnalyticalParameter(self,name='f_in',par_type='',val=0.1,val_min=0,val_max=None,units=''))     
        self.parameters.add_par(AnalyticalParameter(self,name='f_out',par_type='',val=0.1,val_min=0,val_max=None,units='')) 
        self.parameters.add_par(AnalyticalParameter(self,name='z',par_type='',val=0.1,val_min=0,val_max=None,units='')) 
        
        self.SED = SED(name=self.model_type)
        if cosmo is not None:
            self.cosmo=cosmo
        else:
            self.cosmo=cosmo()

        #self.DL=self.cosmo.get_DL_cm()
        
        self.HPLANCK=const.h.cgs.value
        self.SIGSB=const.sigma_sb.cgs.value
        self.vluce_cm=const.c.cgs.value
        self.K_boltz=const.k_B.cgs.value

        self._nu_to_keV=(1.0*u.Unit('Hz')).to(1000.0*u.eV,equivalencies=u.spectral()).value
        self._keV_to_erg=(1.0*u.Unit('keV').to('erg'))

    def _T(self,R_in,R):
       
        tin=self.parameters.T_Disk.val
        fout=self.parameters.f_out.val
        fin=self.parameters.f_in.val
        rirr=self.parameters.r_irr.val
        lcld=self.parameters.L_Comp_ratio.val
        r=R/R_in
        r_irr=self.parameters.r_irr.val
        t=np.zeros(r.shape)
        m = r<=r_irr
        t[m]=(tin*(r[m]**(-0.75)))*(  (1.0 + fout*r[m]*(1.0+(1.0+fin)*lcld)+ 2.0*(r[m]**3)*lcld*fin/(rirr**2-1.0) )**0.25)
        #t[m]=tin**4 * (r[m]**-3 + (fout*r[m]**-2) * (1.0+(1.0+fin)*lcld)+ 2.0*lcld*fin/(rirr**2-1.0) )
        m = r>r_irr
        t[m]=(tin*(r[m]**(-0.75)))*(  (1.0 + fout*r[m]*(1.0+(1.0+fin)*lcld)  )**0.25)
        #t[m]=tin**4 *(r[m]**-3 + (fout*r[m]**-2)* (1.0+(1.0+fin)*lcld) )
        #t=np.power(t,0.25)
        return t

    def _F_nu_integrand(self,nu,R_in,R):
        return  R / (np.exp((self.HPLANCK * nu) / (self.K_boltz *self._T(R_in,R))) - 1)

    def _F_nu(self,nu,R_in,R_irr,R_out):
        """
        Integration of DR(T)
        """

       
        R=np.append(np.logspace(np.log10(R_in),np.log10(R_irr),50),np.logspace(np.log10(R_irr),np.log10(R_out),1000)[1::])
        y=np.zeros(nu.size)
        for ID,n in enumerate(nu):
            f=self._F_nu_integrand(n,R_in,R)
            y[ID]= np.trapz(f,R)
        return y*nu*nu*nu

    def _R_in(self):
        """ 
        R_in is set from 
        T_Disk^4=L_D/(4 pi SIGTH R_in^2)
        """
        return np.sqrt(self.parameters.L_Disk.val/(4*np.pi*self.SIGSB*self.parameters.T_Disk.val*self.parameters.T_Disk.val*self.parameters.T_Disk.val*self.parameters.T_Disk.val))

    def _comp(self,nu):
        E_keV=nu*self._nu_to_keV
        a=np.power((E_keV/self.parameters.Eb_Comp.val),-3)+ np.power((E_keV/self.parameters.Eb_Comp.val),self.parameters.Gamma_Comp.val-2)
        y=np.exp(-(E_keV/self.parameters.Ec_Comp.val)**self.parameters.alpha_Comp.val)*np.power(self.parameters.Eb_Comp.val,2-self.parameters.Gamma_Comp.val)/a
        y=y/(E_keV*E_keV)
        comp=y*E_keV*E_keV*self._keV_to_erg
        return comp

    def lin_func(self,nu):
        
        
        R_in=self._R_in()
        R_out=R_in*self.parameters.r_out.val
        R_irr= R_in*self.parameters.r_irr.val      

        


        dl=self.cosmo.get_DL_cm(z=self.parameters.z.val)
        
        nuFnu_disk=nu*self._F_nu(nu,R_in,R_irr,R_out)*(np.pi*4*self.HPLANCK)/ (self.vluce_cm* self.vluce_cm)
        
        log_nu_2=np.log10(10*self.K_boltz*self._T(R_in,R_in)/self.HPLANCK)
        log_nu_1=np.log10(self.K_boltz*self._T(R_in,R_out)/self.HPLANCK/10)
        nu_disk=np.logspace(log_nu_1,log_nu_2,500)
        Fnu_disk_L=self._F_nu(nu_disk,R_in,R_irr,R_out)*(np.pi*4*self.HPLANCK)/ (self.vluce_cm* self.vluce_cm)
        L_D=np.trapz(Fnu_disk_L,nu_disk)
        L_bol=L_D*(1+self.parameters.L_Comp_ratio.val*(1+self.parameters.f_in.val))
        
      
        
        nuFnu_comp=self._comp(nu)       
        log_nu_1=np.log10(self.parameters.Eb_Comp.val/self._nu_to_keV)
        log_nu_2=np.log10(self.parameters.Ec_Comp.val/self._nu_to_keV*10)
        nu_comp=np.logspace(log_nu_1,log_nu_2,100)
        self.L_C=np.trapz(self._comp(nu_comp)/nu_comp,nu_comp)
        self.L_D=L_D
        self.T_in=self._T(R_in,R_in)
        comp_factor=1.0/self.L_C

        cos_factor=np.cos(np.deg2rad(self.parameters.theta.val))
        
        nuFnu_comp=nuFnu_comp*comp_factor*L_D*self.parameters.L_Comp_ratio.val
        return (nuFnu_disk*cos_factor+nuFnu_comp)/(dl*dl)




class DiskIrr(Model):
    """
    Class to handle function for irradiated disk and comptonization model
    """
    
    def __init__(self,nu_size=100,cosmo=None,name='DiskIrrComp',compth=False,hump=False,**keywords):
        """
        https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2966.2008.14166.x
        https://arxiv.org/abs/0809.3255v2
        """
        
        super(DiskIrr,self).__init__(  **keywords)
        self.name=name
        self.parameters = ModelParameterArray()      
       
        self.parameters.add_par(AnalyticalParameter(self,name='T_Disk',par_type='inner disk T.',val=1E5,val_min=0,val_max=None,units='k'))
        self.parameters.add_par(AnalyticalParameter(self,name='L_Disk',par_type='bolobmetric disk lum.',val=1E36,val_min=0,val_max=None,units='erg s-1'))
        self.parameters.add_par(AnalyticalParameter(self,name='theta',par_type='disk incl.',val=0,val_min=0,val_max=90,units='deg'))
        self.parameters.add_par(AnalyticalParameter(self,name='r_out',par_type='disk r_out (r_in)',val=100,val_min=1.,val_max=None,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='r_irr',par_type='disk r_irr (r_in)',val=1.1,val_min=1.,val_max=None,units=''))
        
        if compth is not None:
            self.parameters.add_par(AnalyticalParameter(self,name='L_Comp_ratio',par_type='L_Comp/L_Disk',val=0.1,val_min=0,val_max=None,units=''))   
            self.parameters.add_par(AnalyticalParameter(self,name='Eb_Comp',par_type='',val=1,val_min=0,val_max=None,units='keV'))        
            self.parameters.add_par(AnalyticalParameter(self,name='Ec_Comp',par_type='',val=1,val_min=0.,val_max=None,units='keV'))       
            self.parameters.add_par(AnalyticalParameter(self,name='Gamma_Comp',par_type='',val=1,val_min=-10.,val_max=None,units=''))
            self.parameters.add_par(AnalyticalParameter(self,name='alpha_Comp',par_type='',val=1,val_min=-10.,val_max=None,units=''))
            #self.parameters.add_par(AnalyticalParameter(self,name='K_Comp',par_type='',val=1,val_min=0,val_max=None,units='keV-1 cm-2'))
        
        if hump is not None:
            self.parameters.add_par(AnalyticalParameter(self,name='Ec_hump',par_type='',val=1,val_min=0.,val_max=None,units='keV'))
            self.parameters.add_par(AnalyticalParameter(self,name='Gamma_hump',par_type='',val=1,val_min=-10,val_max=10,units=''))
            self.parameters.add_par(AnalyticalParameter(self,name='K_hump',par_type='',val=1,val_min=0,val_max=None,units='keV-1 cm-2'))
            self.parameters.add_par(AnalyticalParameter(self,name='alpha_hump',par_type='',val=1,val_min=-10.,val_max=None,units=''))

        self.parameters.add_par(AnalyticalParameter(self,name='f_in',par_type='',val=0.1,val_min=0,val_max=None,units=''))     
        self.parameters.add_par(AnalyticalParameter(self,name='f_out',par_type='',val=0.1,val_min=0,val_max=None,units='')) 
        self.parameters.add_par(AnalyticalParameter(self,name='z',par_type='',val=0.1,val_min=0,val_max=None,units='')) 
        
        self.SED = SED(name=self.model_type)
        if cosmo is not None:
            self.cosmo=cosmo
        else:
            self.cosmo=cosmo()

        #self.DL=self.cosmo.get_DL_cm()

        self.compth=compth
        self.hump=hump
        if hump is not None and compth is None:
            raise RuntimeError("if you provide hump you must provide compth too")

        self.HPLANCK=const.h.cgs.value
        self.SIGSB=const.sigma_sb.cgs.value
        self.vluce_cm=const.c.cgs.value
        self.K_boltz=const.k_B.cgs.value

        self._nu_to_keV=(1.0*u.Unit('Hz')).to(1000.0*u.eV,equivalencies=u.spectral()).value
        self._keV_to_erg=(1.0*u.Unit('keV').to('erg'))

   
    def _T(self,R_in,R):
       
        tin=self.parameters.T_Disk.val
        fout=self.parameters.f_out.val
        fin=self.parameters.f_in.val
        rirr=self.parameters.r_irr.val
        lcld=self.parameters.L_Comp_ratio.val
        r=R/R_in
        r_irr=self.parameters.r_irr.val
        t=np.zeros(r.shape)
        m = r<=r_irr
        t[m]=(tin*(r[m]**(-0.75)))*(  (1.0 + fout*r[m]*(1.0+(1.0+fin)*lcld)+ 2.0*(r[m]**3)*lcld*fin/(rirr**2-1.0) )**0.25)
        #t[m]=tin**4 * (r[m]**-3 + (fout*r[m]**-2) * (1.0+(1.0+fin)*lcld)+ 2.0*lcld*fin/(rirr**2-1.0) )
        m = r>r_irr
        t[m]=(tin*(r[m]**(-0.75)))*(  (1.0 + fout*r[m]*(1.0+(1.0+fin)*lcld)  )**0.25)
        #t[m]=tin**4 *(r[m]**-3 + (fout*r[m]**-2)* (1.0+(1.0+fin)*lcld) )
        #t=np.power(t,0.25)
        return t

    def _F_nu_integrand(self,nu,R_in,R):
        return  R / (np.exp((self.HPLANCK * nu) / (self.K_boltz *self._T(R_in,R))) - 1)

    def _F_nu(self,nu,R_in,R_irr,R_out):
        """
        Integration of DR(T)
        """

       
        R=np.append(np.logspace(np.log10(R_in),np.log10(R_irr),50),np.logspace(np.log10(R_irr),np.log10(R_out),1000)[1::])
        #print(R,R.shape)
        y=np.zeros(nu.size)
        for ID,n in enumerate(nu):
            f=self._F_nu_integrand(n,R_in,R)
            y[ID]= np.trapz(f,R)
        return y*nu*nu*nu

    def _R_in(self):
        """ 
        R_in is set from 
        T_Disk^4=L_D/(4 pi SIGTH R_in^2)
        """
        return np.sqrt(self.parameters.L_Disk.val/(4*np.pi*self.SIGSB*self.parameters.T_Disk.val*self.parameters.T_Disk.val*self.parameters.T_Disk.val*self.parameters.T_Disk.val))

    def _comp(self,nu):
        E_keV=nu*self._nu_to_keV
        a=np.power((E_keV/self.parameters.Eb_Comp.val),-3)+ np.power((E_keV/self.parameters.Eb_Comp.val),self.parameters.Gamma_Comp.val-2)
        y=np.exp(-(E_keV/self.parameters.Ec_Comp.val)**self.parameters.alpha_Comp.val)*np.power(self.parameters.Eb_Comp.val,2-self.parameters.Gamma_Comp.val)/a
        y=y/(E_keV*E_keV)
        comp=y*E_keV*E_keV*self._keV_to_erg
        return comp

    def _hump(self,nu):
        E_keV=nu*self._nu_to_keV
        y=np.power(E_keV,-self.parameters.Gamma_hump.val)*np.exp(-(E_keV/self.parameters.Ec_hump.val)**self.parameters.alpha_hump.val)
        return y*self.parameters.K_hump.val*E_keV*E_keV*self._keV_to_erg

    def lin_func(self,nu):
        
        
        R_in=self._R_in()
        R_out=R_in*self.parameters.r_out.val
        R_irr= R_in*self.parameters.r_irr.val      

        


        dl=self.cosmo.get_DL_cm(z=self.parameters.z.val)
        
        nuFnu_disk=nu*self._F_nu(nu,R_in,R_irr,R_out)*(np.pi*4*self.HPLANCK)/ (self.vluce_cm* self.vluce_cm)
        
        log_nu_2=np.log10(10*self.K_boltz*self._T(R_in,R_in)/self.HPLANCK)
        log_nu_1=np.log10(self.K_boltz*self._T(R_in,R_out)/self.HPLANCK/10)
        nu_disk=np.logspace(log_nu_1,log_nu_2,500)
        Fnu_disk_L=self._F_nu(nu_disk,R_in,R_irr,R_out)*(np.pi*4*self.HPLANCK)/ (self.vluce_cm* self.vluce_cm)
        L_D=np.trapz(Fnu_disk_L,nu_disk)
        
        if self.compth is not None:
            L_Comp_ratio=self.parameters.L_Comp_ratio.val
        else:
            L_Comp_ratio=0

        L_bol=L_D*(1+L_Comp_ratio*(1+self.parameters.f_in.val))

        self.L_C=0
        self.L_H=0
        self.L_H_C=0
        nuFnu_comp=0
        nuFnu_hump=0
        if self.compth is not None:
            nuFnu_comp=self._comp(nu)       
            log_nu_1=np.log10(self.parameters.Eb_Comp.val/self._nu_to_keV)
            log_nu_2=np.log10(self.parameters.Ec_Comp.val/self._nu_to_keV*10)
            nu_comp=np.logspace(log_nu_1,log_nu_2,100)
            self.L_C=np.trapz(self._comp(nu_comp)/nu_comp,nu_comp)
            nuFnu_comp *= 1.0/self.L_C
        if self.hump is not None:
            nuFnu_hump=self._hump(nu)                   
            self.L_H=np.trapz(self._hump(nu_comp)/nu_comp,nu_comp)
            nuFnu_hump *= 1.0/self.L_H
            self.L_H_C=self.L_H/self.L_C
        self.L_D=L_D
        self.T_in=self._T(R_in,R_in)
        
        #comp_factor=1.0/self.L_C

        cos_factor=np.cos(np.deg2rad(self.parameters.theta.val))
       
        nuFnu_comp=nuFnu_comp*L_D*self.parameters.L_Comp_ratio.val/(1+self.L_H_C)
        nuFnu_hump=nuFnu_hump*L_D*self.parameters.L_Comp_ratio.val*(self.L_H_C/(1+self.L_H_C))
       
        self.nuFnu_disk=nuFnu_disk*cos_factor/(dl*dl)
        self.nuFnu_comp=nuFnu_comp*cos_factor/(dl*dl)
        self.nuFnu_hump=nuFnu_hump*cos_factor/(dl*dl)
        
        return self.nuFnu_disk+self.nuFnu_comp+self.nuFnu_hump







from scipy.interpolate import interp1d
from jetset.base_model import  Model, MultiplicativeModel

class PhabsModel(MultiplicativeModel):
    def __init__(self,nu_size=100,cosmo=None,**keywords):
        """
       http://articles.adsabs.harvard.edu/pdf/1983ApJ...270..119M
        """
        
        super(PhabsModel,self).__init__(  **keywords)
        self.name='phabs'
        self.parameters = ModelParameterArray()      
       
        self.parameters.add_par(AnalyticalParameter(self,name='N_H',par_type='',val=1,val_min=0,val_max=None,units='cm-2'))        
        
        self.nu_min=1E14
        self.nu_max=1E19
      
        if cosmo is not None:
            self.cosmo=cosmo
        else:
            self.cosmo=cosmo()
            
        self._set_coeff_table()
        self._nu_to_keV=(1.0*u.Unit('Hz')).to(1000.0*u.eV,equivalencies=u.spectral()).value
    
    def _set_coeff_table(self):
        self._egrid=np.array([0.030,0.100,0.284,0.4,0.532,0.707,0.867,1.303,1.840,2.471,3.210,4.038,7.11,8.331,10])
        self._egrid=(self._egrid[1:]+self._egrid[:-1])*0.5
        
        self._c0=np.array([17.3,
                            34.6,
                            78.1,
                            71.4,
                            95.5,
                            308.9,
                            120.6,
                            141.3,
                            202.7,
                            342.7,
                            352.2,
                            433.9,
                            629.0,
                            701.2])
        
        self._c1=np.array([608.1,
                            267.9,
                            18.8,
                            66.8,
                            145.8,
                            -380.6,
                            169.3,
                            146.8,
                            104.7,
                            18.7,
                            18.7,
                            -2.4,
                            30.9,
                            25.2])
        
        self._c2=np.array([-2150.,
                            -476.1,
                            4.3,
                            -51.4,
                            -61.1,
                            294.0,
                            -47.7,
                            -31.5,
                            -17.0,
                            0.0,
                            0.0,
                            0.75,
                            0.0,
                            0.0])
        
        self.c0= interp1d(self._egrid,self._c0,bounds_error =False,fill_value=0)
        self.c1= interp1d(self._egrid,self._c1,bounds_error =False,fill_value=0)
        self.c2= interp1d(self._egrid,self._c2,bounds_error =False,fill_value=0)
        
    
    
    def eval(self, fill_SED=True, nu=None, get_model=False, loglog=False):
        """

        Parameters
        ----------
        fill_SED
        nu
        get_model
        loglog
        z

        Returns
        -------

        """

        out_model=None

        if nu is None:
            nu = np.copy(self._egrid*2.4E17)

        else:
            if loglog is True:
                log_nu = nu
            else:
                log_nu = np.log10(nu)



        model =  self.lin_func(nu)

        if get_model == True:
            if loglog == False:
                out_model=model
            else:
                out_model=  np.log10(model)

        if fill_SED is True:
            if loglog is False:
                _nu=np.power(10.,log_nu)
            else:
                _nu=np.copy(log_nu)

            self.nu=_nu
            self.tau=model

        return out_model

        
    def lin_func(self,nu):
        E=nu*self._nu_to_keV
        sigma_E=(self.c0(E)+ self.c1(E)*E + self.c2(E)*E*E)/(E*E*E)

        #print(nu,np.exp(-sigma_E*self.parameters.N_H.val*1E-24))
        return np.exp(-sigma_E*self.parameters.N_H.val*1E-24)
    
    
    def plot_model(self, plot_obj=None,  label=None, line_style='-',color=None):
        if plot_obj is None:
            plot_obj = PlotSpectralMultipl()

        if label is None:
            label = 'N_H=%5.5e atom/cm2'%self.parameters.get_par_by_name('N_H').val

        plot_obj.plot(nu=self.nu,y=self.tau, y_label=r'$ log10(\exp^{- \tau}) $', line_style=line_style, label=label, color=color)


        return plot_ob






class RadioJet(Model):
    """
    Class to handle power-law cut-off model
    """
    
    def __init__(self,jet_radio,nu_size=100,N_comp=10,cosmo=None,log_R_H_grid=True,name='radio_jet',**keywords):
       
        
        super(RadioJet,self).__init__( **keywords)
        self.name=name
        self.parameters = ModelParameterArray()      
        self.N_comp=N_comp
        self.jet=jet_radio
        self.log_R_H_grid=log_R_H_grid
        self.parameters.add_par(AnalyticalParameter(self,name='B_0',par_type='magnetic_field',val=1,val_min=0.,val_max=None,units='Gauss'))
        self.parameters.add_par(AnalyticalParameter(self,name='z_0',par_type='',val=1,val_min=0.,val_max=None,units='cm'))
        self.parameters.add_par(AnalyticalParameter(self,name='z_inj',par_type='',val=1,val_min=0.,val_max=None,units='cm'))
        self.parameters.add_par(AnalyticalParameter(self,name='BulkFactor',par_type='',val=1,val_min=0.,val_max=None,units=''))

        self.parameters.add_par(AnalyticalParameter(self,name='N_frac',par_type='emitters_density ratio',val=1,val_min=0.,val_max=None,units=''))
        #self.parameters.add_par(AnalyticalParameter(self,name='R_0',par_type='jet_base_radius',val=1E10,val_min=0,val_max=None,units='cm'))
        #self.parameters.add_par(AnalyticalParameter(self,name='R_H_0',par_type='jet_base_position',val=1E10,val_min=0,val_max=None,units='cm'))
        self.parameters.add_par(AnalyticalParameter(self,name='R_H_start_frac',par_type='jet_start_position',val=100,val_min=0.0,val_max=None,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='R_H_stop_frac',par_type='jet_stop_position',val=100,val_min=1.0,val_max=None,units=''))
        #self.parameters.add_par(AnalyticalParameter(self,name='R_index',par_type='',val=1,val_min=-4,val_max=4.,units='None'))
        self.parameters.add_par(AnalyticalParameter(self,name='m_index',par_type='',val=1,val_min=-4,val_max=4.,units='None'))
        #self.parameters.add_par(AnalyticalParameter(self,name='N_index',par_type='',val=1,val_min=-4,val_max=4.,units='None'))
        
        self.parameters.add_par(AnalyticalParameter(self,name='gamma_cut',par_type='',val=10,val_min=1,val_max=None,units='None'))
        self.parameters.add_par(AnalyticalParameter(self,name='p',par_type='',val=1,val_min=-10,val_max=10,units='None'))
        self.parameters.add_par(AnalyticalParameter(self,name='N_0',par_type='',val=1,val_min=0,val_max=None,units='None'))
        self.parameters.add_par(AnalyticalParameter(self,name='B',par_type='magnetic_field',val=1,val_min=0.,val_max=None,units='Gauss'))
        self.parameters.add_par(AnalyticalParameter(self,name='R',par_type='R',val=1,val_min=0.,val_max=None,units='cm'))

        self.SED = SED(name=self.model_type)
        if cosmo is not None:
            self.cosmo=cosmo
        else:
            self.cosmo=cosmo()

        self.adiabatic_cooling=True

    def eval_t_comov(self,z,BulkFactor,v_jet):
        return z/(v_jet*BulkFactor)
    
    def gamma_cool(self,z,z_inj,z_0,g_inj,B0,BulkFactor,v_jet,m_R,m_B,m_N):
        t_0=self.eval_t_comov(z_0,BulkFactor,v_jet)

        t_inj=self.eval_t_comov(z_inj,BulkFactor,v_jet)
        t=self.eval_t_comov(z,BulkFactor,v_jet)

        f=1-2*m_B -2*(m_R/3)
        cooling_const=(const.sigma_T.cgs.value)/(6*const.m_e.cgs.value*const.c.to('cm s-1').value*f*np.pi)*B0**2

        a=g_inj*np.power(t,-(2*m_R/3))
        b=np.power(t_inj,-(2*m_R/3))
        c=cooling_const*np.power(t_0,2*m_B)
        d=g_inj*(t**f - t_inj**f)
        
        return a/(b+c*d)


    def gamma_cool_no_ad(self,z,z_inj,z_0,g_inj,B0,BulkFactor,v_jet,m_index):
        t_0=self.eval_t_comov(z_inj,BulkFactor,v_jet)
        
        t_inj=self.eval_t_comov(z_inj,BulkFactor,v_jet)
        t_0=t_inj
        


        t=self.eval_t_comov(z,BulkFactor,v_jet)
        #print('->B(t_inj),',B0*(t_inj/t_0)**-m_index,self.jet_acc.parameters.B.val) 
        #print(t[0],t_0)
        f=1-2*m_index
        cooling_const=(const.sigma_T.cgs.value*B0**2*t_0)/(6*-f*const.m_e.cgs.value*const.c.to('cm s-1').value*np.pi)

        a=g_inj 
    
        c=cooling_const*g_inj
        d=(1 - (t/t_0)**(f))
        #print('%e'%d[-1])
        return a/(1+c*d)

    def gamma_cool_no_ad_peer(self,z,z_inj,z_0,g_inj,B0,BulkFactor,v_jet,m_index):
        t_0=self.eval_t_comov(z_0,BulkFactor,v_jet)
        
        t_inj=self.eval_t_comov(z_inj,BulkFactor,v_jet)
        t=self.eval_t_comov(z,BulkFactor,v_jet)

        f=2*m_index-1
        cooling_const=(const.sigma_T.cgs.value*B0**2*t_inj)/(f*const.m_e.cgs.value*const.c.to('cm s-1').value*np.pi*6)

        a=g_inj 
    
        c=cooling_const*g_inj
        d=(1-(t/t_inj)**-f)
        #print('%e'%d[-1])
        return a/(1+c*d)


    def B_scaling(self,B0,z0,z,m):
        return B0*(z0/z)**m
    
    def eval_jet_radius(self,z0,z,r_0,R_index):
        return  r_0*np.power((z/z0),R_index)

    def set_Ne(self,N_0,R_0,R,N_index,):
        return N_0*np.power((R_0/R),N_index)

    def eval_vol_shell(self,r_0,z0,z1,z2,R_index):
        r_1=self.eval_jet_radius(z0,z1,r_0,R_index)
        r_2=self.eval_jet_radius(z0,z2,r_0,R_index)
        return (1/3)*np.pi*(r_1**2+r_1*r_2+r_2**2)*(z2-z1)

    def eval_R_blob(self,V_shell):
        return np.power((3*V_shell/(4*np.pi)),1/3)

    def sphere_to_shell(self, nuFu,R_sphere,R_shell,delta_R,beaming):
        S_sphere=np.pi*4*R_sphere*R_sphere
        S_shell=2*np.pi*R_shell*delta_R

        c =S_shell/(S_sphere*beaming)
        #print('R_shell=%e delta R=%e'%(R_shell,delta_R)
        return nuFu*c
        
    def set_region(self,ID,):
        R=self.R_H_array[ID]
        R1=self.R_H_array[ID+1]
        #R_0=self.jet_acc.parameters.R.val
        R_0=self.parameters.R.val
        B_0=self.B_0
        R_H_0=self.parameters.z_inj.val
        B_index=self.B_index
        N_index=self.N_index
        R_index=self.R_index
        
        #N_0=self.jet_acc.parameters.N.val*self.parameters.N_frac.val
        N_0=self.N_0
        #B=(self.B_scaling(B_0,R_H_0,R,B_index)+self.B_scaling(B_0,R_H_0,R1,B_index))*0.5
        B=self.B_scaling(B_0,R_H_0,R,B_index)
        #print('-B',(self.B_scaling(B_0,R_H_0,R,B_index),self.B_scaling(B_0,R_H_0,R1,B_index)))
        V_shell=self.eval_vol_shell(R_0,R_H_0,R,R1,R_index)
        
        R_shell=self.eval_jet_radius(R_H_0,R,R_0,R_index)
        R_shell_1=self.eval_jet_radius(R_H_0,R1,R_0,R_index)
        R_blob=self.eval_R_blob(V_shell)
        

        
        N_e_shell=(self.set_Ne(N_0,R_H_0,R,N_index)+self.set_Ne(N_0,R_H_0,R1,N_index))*0.5
        self.jet.parameters.N.val=N_e_shell
        self.jet.parameters.B.val=B
        self.jet.parameters.R.val=R_shell
        self.jet.parameters.gamma_cut.val=self.gamma_cut[ID]
        self.jet.parameters.p.val=self.p
        #self.jet.parameters.gmax.val=self.gamma_cut[ID]*2
        self.B_array=B
        self.R_shell=R_shell

    def lin_func(self,nu):

        R_H_0=self.parameters.z_inj.val
        R_0=self.parameters.R.val
        z_inj=self.parameters.z_inj.val 
        R_H_start=(R_H_0)*self.parameters.R_H_start_frac.val+2*R_0
        R_H_stop=R_H_0*self.parameters.R_H_stop_frac.val
        self.v_jet=np.sqrt(1- 1/self.parameters.BulkFactor.val**2)*const.c.to('cm s-1').value
        if self.log_R_H_grid is True:
            self.R_H_array= np.logspace(np.log10(R_H_start),np.log10(R_H_stop),self.N_comp)
        else:
            self.R_H_array= np.linspace( R_H_start, R_H_stop ,self.N_comp)
        
        #print('size',self.R_H_array.size)
        self.R_H_array=np.append(R_H_0,self.R_H_array)
        
        #B_0=self.jet_acc.parameters.B.val
        self.B_index=self.parameters.m_index.val
        #N_0=self.jet_acc.parameters.N.val*self.parameters.N_frac.val
        self.N_index=self.parameters.m_index.val*2
        self.R_index=self.parameters.m_index.val
        #self.gamma_inj= self.jet_acc.parameters.gamma_cut.val
        
        #self.gamma_inj=self.jet_acc.parameters.gamma_cut.val
        #self.p=self.jet_acc.parameters.p.val
        #self.N_0=self.jet_acc.parameters.N.val*self.parameters.N_frac.val
        #self.B_0=self.jet_acc.parameters.B.val

        self.gamma_inj=self.parameters.gamma_cut.val
        self.p=self.parameters.p.val
        self.N_0=self.parameters.N_0.val
        self.B_0=self.parameters.B.val

        #B_0=self.parameters.B_0.val
        if self.adiabatic_cooling is False:
           self.gamma_cut=self.gamma_cool_no_ad(self.R_H_array,
                                       z_inj,
                                       self.parameters.z_0.val,
                                       self.gamma_inj,
                                       self.B_0,
                                       self.parameters.BulkFactor.val,
                                       self.v_jet,
                                       self.B_index)
        else:
            self.gamma_cut=self.gamma_cool(self.R_H_array,
                                       z_inj,
                                       self.parameters.z_0.val,
                                       self.gamma_inj,
                                       self.B_0,
                                       self.parameters.BulkFactor.val,
                                       self.v_jet,
                                       self.B_index)
        #print(self.R_H_array,
        #        R_H_start,
        #        self.parameters.z_0.val,
        #        self.jet_acc.parameters.gamma_cut.val,
        #        self.parameters.B_0.val,
        #        self.parameters.BulkFactor.val,
        #        self.v_jet,
        #        self.parameters.R_index.val,
        #        self.parameters.B_index.val,
        #        self.parameters.N_index.val)
        #print(np.log10(self.R_H_array))
        #print(self.gamma_cut)
        self.gamma_cut[self.gamma_cut<1]=1
        #self.gamma_cut[::]=self.parameters.gamma_cut.val
        y=None
        #delta_R_0=R_H_array[1]-R_H_array[0]
        #V0=self.eval_vol_shell(R_0,R_H_0,R_H_array[0],R_H_array[1],R_index)
        for ID,R_H in enumerate(self.R_H_array[:self.N_comp]):
            R1=self.R_H_array[ID+1]
            self.set_region(ID)

            delta_R=R1-R_H
            #B=(self.B_scaling(B_0,R_H_0,R_H,B_index)+self.B_scaling(B_0,R_H_0,R_H_array[ID+1],B_index))*0.5
            #V_shell=self.eval_vol_shell(R_0,R_H_0,R_H,R_H_array[ID+1],R_index)
            R_shell=self.eval_jet_radius(R_H_0,R_H,R_0,self.R_index)
            #R_shell_1=self.eval_jet_radius(R_H_0,R_H_array[ID+1],R_0,R_index)
            #R_blob=self.eval_R_blob(V_shell)
            #if y is None:
            #    N_e_shell=N_0
            #else:
            
            #N_e_shell=(self.set_Ne(N_0,R_H_0,R_H,N_index)+self.set_Ne(N_0,R_H_0,R_H_array[ID+1],N_index))*0.5
            #N_e_shell=self.set_Ne(N_0,R_H_0,R_H,N_index)
            #self.jet.parameters.N.val=N_e_shell
            #self.jet.parameters.B.val=B
            #self.jet.parameters.R.val=R_shell
            #print(self.eval_jet_radius(R_H_start,R_H,R_0,R_index)/delta_R,R_index)
            #print('R_0,R_shell,N_e_shell',R_0,R_shell,N_e_shell)
            nuFnu_shell_sphere=self.jet.eval(nu=nu, fill_SED=True,get_model=True, loglog=False)
            nuFnu_shell_shell=self.sphere_to_shell(nuFnu_shell_sphere,R_shell,R_shell,delta_R,self.jet.get_beaming())
            #print(nuFnu_shell_sphere,nuFnu_shell_shell)
            #print('V_shell,V0',V_shell,V0,N_0,N_e_shell,N_0*V0,N_e_shell*V_shell)
            if y is None:
                y=nuFnu_shell_shell
            else:
                y += nuFnu_shell_shell
        #delta_R=R_H_array[1]-R_H_array[0]
        #print(self.eval_jet_radius(R_H_0,R_H_array[0],R_0,R_index)/delta_R,R_index)
        #delta_R=R_H_array[-1]-R_H_array[-2]
        #print(self.eval_jet_radius(R_H_0,R_H_array[-2],R_0,R_index)/delta_R,R_index)
        
        return y        



