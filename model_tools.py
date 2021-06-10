from jetset.data_loader import Data, ObsData
from jetset.cosmo_tools import Cosmo
from astropy import units as u
from IPython.core.display import display, HTML
import numpy as np
from jetset.base_model import Model
from jetset.model_manager import FitModel
from jetset.model_parameters import ModelParameterArray, ModelParameter
from jetset.analytical_model import AnalyticalParameter
from jetset.spectral_shapes import SED
from jetset.analytical_model import Disk
from jetset.jet_model import Jet
from jetset.plot_sedfit import PlotSED
from jetset_models.models import *
from jetset.jetkernel import jetkernel 
from jet_setup import JetSetup
from jetset_models.models import DiskIrr, RadioJet
from astropy import units as u
from astropy import constants as const
from astropy import time
import copy


def set_global_model(jet_setup,
                     sed_data,
                     corona_model_file,
                     cosmo,
                     m_index=0.9,
                     R_H_stop_frac=3500,
                     N_0=1E12,
                     gamma_cut=90,
                     p=1.0,
                     p_frozen=False,
                     z_inj_frozen=True,
                     nu_1=1E8,
                     nu_2=1E21,
                     freeze_disk_irr=True ):

    composite_model=FitModel.load_model(corona_model_file)

    if freeze_disk_irr is True:
        composite_model.disk_irr.parameters.freeze_all()
    #composite_model.hump.parameters.freeze_all()

    my_jet_radio = Jet(beaming_expr='bulk_theta',electron_distribution='plc',cosmo=cosmo,name='jet_radio')
    my_jet_radio.spectral_components.SSC.state='on'
    my_jet_radio.spectral_components.Sync.state='self-abs'
    my_jet_radio.parameters.BulkFactor.val=2.19
    my_jet_radio.parameters.BulkFactor.frozen=False
    my_jet_radio.parameters.theta.val=63
    my_jet_radio.parameters.theta.frozen=True

    my_jet_radio.parameters.z_cosm.val=0
    my_jet_radio.parameters.z_cosm.frozen=True

    my_jet_radio.parameters.gmin.val=1.0
    my_jet_radio.parameters.gmin.frozen=True

    my_jet_radio.parameters.gmax.val=1E3
    my_jet_radio.parameters.gmax.frozen=False

    #my_jet_radio.parameters.p_1.val=2.5
    my_jet_radio.set_gamma_grid_size(200)

    jet_radio=RadioJet(jet_radio=my_jet_radio,N_comp=20,cosmo=cosmo,log_R_H_grid=True)
    #new
    jet_radio.parameters.B_0.val=jet_setup.B_0
    jet_radio.parameters.B_0.frozen=True

    jet_radio.parameters.z_0.val=jet_setup.z_0
    jet_radio.parameters.z_0.frozen=True

    jet_radio.parameters.z_inj.val=jet_setup.z_acc_start
    jet_radio.parameters.z_inj.frozen=z_inj_frozen
    #jet_radio.parameters.z_inj.fit_range=[jet_setup.z_acc_start*0.5,jet_setup.z_acc_start*1.5]
    jet_radio.parameters.BulkFactor.val=jet_setup.BulkFactor
    jet_radio.parameters.BulkFactor.frozen=True


    jet_radio.parameters.p.val=p
    jet_radio.parameters.p.frozen=p_frozen

    jet_radio.parameters.gamma_cut.val=gamma_cut
    jet_radio.parameters.gamma_cut.fit_range=[1,10000]
    jet_radio.parameters.N_0.val=N_0
    jet_radio.parameters.B.val=jet_setup.B_acc

    jet_radio.parameters.B.fit_range=[jet_setup.B_acc*0.5,jet_setup.B_acc*1.5]

    jet_radio.parameters.R.val=jet_setup.R_acc
    jet_radio.parameters.R.fit_range=[jet_setup.R_acc*0.5,jet_setup.R_acc*1.5]

    #

    jet_radio.parameters.N_frac.val=1
    jet_radio.parameters.N_frac.frozen=True
    jet_radio.parameters.R_H_start_frac.val=1
    jet_radio.parameters.R_H_start_frac.frozen=True
    #jet_radio.parameters.R_H_start_frac.fit_range=[1.0,2.0]

    jet_radio.parameters.R_H_stop_frac.val=R_H_stop_frac
    jet_radio.parameters.R_H_stop_frac.frozen=True
    #jet_radio.parameters.R_H_stop_frac.fit_range=[2000,3500]



    jet_radio.parameters.m_index.val=m_index


    jet_radio.parameters.m_index.fit_range=[0.5,1.5]


    #jet_radio.parameters.N_index.val=jet_radio.parameters.B_index.val*2
    #jet_radio.parameters.N_index.frozen=True

    #jet_radio.parameters.R_index.val=0.9
    #jet_radio.parameters.R_index.fit_range=[0.8,1.0]

    #jet_radio.parameters.R_index.frozen=False

    jet_radio.jet.set_nu_grid(1E8,1E21,100)
    jet_radio.adiabatic_cooling=False
    jet_radio.show_pars()


    #print('z_inj=%e, z_inij(R_s)=%e'%(np.log10(jet_setup.z_acc*jet_radio.parameters.z_inj_frac.val),jet_setup.z_acc*jet_radio.parameters.z_inj_frac.val/jet_setup.Rs))
    composite_model.add_component(jet_radio)
    composite_model.nu_min_fit=nu_1
    composite_model.nu_max_fit=nu_2
    composite_model.set_nu_grid(nu_1,nu_2,500)
    composite_model.eval()
    p=composite_model.plot_model(density=True,sed_data=sed_data)
    p.rescale(y_max=-22,y_min=-27,x_min=6,x_max=20)
    return composite_model



def get_jet_acc(composite_model):
    composite_model.radio_jet.set_region(0)
    jet_acc=copy.deepcopy(composite_model.radio_jet.jet)
    jet_acc.name='jet acc.'
    jet_acc.spectral_components.SSC.state='on'
    jet_acc.set_gamma_grid_size(200)
    jet_acc.set_nu_grid(1E8,1E21,500)
    jet_acc.nu_min_fit=1E8
    jet_acc.nu_max_fit=1E21
    jet_acc.nu_min=1E8
    jet_acc.nu_max=1E21
    return jet_acc

def plot_Fnu(model,sed_data):
    jet_acc=get_jet_acc(model)
    jet_acc.eval()
    p=model.plot_model(sed_data=sed_data,frame='obs',skip_sub_components=True,density=True)
    p.add_model_plot(jet_acc,label='jet acc.',density=True,line_style='--',color='blue')
    sec_ax=p.sedplot.secondary_yaxis('right',functions=(lambda x:x+26, lambda x: x-26) )
    p.sedplot.set_ylabel(r'log(F$\nu$) ( (erg cm$^{-2}$  s$^{-1}$ Hz$^{-1}$))')

    sec_ax.set_ylabel(r'log(F$\nu$) (mJy)')
    p.sedplot.legend(loc='best')
    #p.sedplot.axes.axvline(np.log10(7*2.4E17))
    #p.fig.set_size_inches(12,12)
    p.rescale(y_max=-22.5,y_min=-28.5,x_min=8,x_max=21)
    
    return p

def plot_SED(model,sed_data,skip_sub_components=True):
    jet_acc=get_jet_acc(model)
    jet_acc.eval()
    p=model.plot_model(sed_data=sed_data,frame='obs',skip_sub_components=skip_sub_components)
    p.add_model_plot(jet_acc,line_style='--',label='jet acc.',density=False,color='blue')
    sec_ax=p.sedplot.secondary_yaxis('right',functions=(lambda x:x+26, lambda x: x-26) )
    #p.sedplot.set_ylabel(r'log(F$\nu$) ( (erg cm$^{-2}$  s$^{-1}$ Hz$^{-1}$))')

    sec_ax.set_ylabel(r'log(nuF$\nu$) (mJy Hz)')
    p.sedplot.legend(loc='best')
    #p.sedplot.axes.axvline(np.log10(7*2.4E17))
    #p.fig.set_size_inches(12,12)
    p.rescale(y_max=-7,y_min=-16.2,x_min=8,x_max=20.9)
    return p

def final_report(composite_model,jet_setup):
    jet_acc=get_jet_acc(composite_model)
    jet_acc.eval()
    jet_acc.energetic_report(verbose=False)
    print(jet_acc.energetic_dict['U_e'],jet_acc.energetic_dict['U_B'])
    print('(U_e/U_B)',jet_acc.energetic_dict['U_e']/jet_acc.energetic_dict['U_B'])
    print('jet_setup.L_Edd=',jet_setup.L_Edd)
    print('jet_setup.L_jet',jet_setup.L_jet)
    print('jet_setup.L_p_acc',jet_setup.L_p_acc)
    L_p_acc_jet=jet_acc.energetic_dict['jet_L_B']*jet_setup.eps_Up_UB_acc
    print('L_p_acc_jet= from L_p> jet_set.L_B',L_p_acc_jet)
    L_tot_no_p=jet_acc.energetic_dict['jet_L_tot']-jet_acc.energetic_dict['jet_L_p_cold']
    L_tot=L_tot_no_p+L_p_acc_jet
    print('L_tot_no_p =(jetset.L_tot  - jetset.Lp)=',L_tot_no_p)
    print('L_tot = (jetset.L_tot  - jetset.Lp +L_p_acc_jet)=',L_tot)
    print('jetset.L_tot/jet_setup.L_Edd',jet_acc.energetic_dict['jet_L_tot']/jet_setup.L_Edd)
    print('L_tot/jet_setup.L_Edd',L_tot/jet_setup.L_Edd)
    print('L_tot/jet_setup.L_je',L_tot/jet_setup.L_jet)
    print('L_tot_no_p/jet_setup.L_jet',L_tot_no_p/jet_setup.L_jet)
    print('L_tot_no_p/jet_setup.L_Edd',L_tot_no_p/jet_setup.L_Edd)
    
    N_p=jet_setup.eval_N_p(jet_setup.L_p_acc,jet_setup.BulkFactor,jet_setup.R_acc,jet_setup.beta)

    print('N_p=%e'%N_p)
    print('N_e=%e'%jet_acc.parameters.N.val)
    print('N_e/N_p=%e'%(jet_acc.parameters.N.val/N_p))
    print('eps_Up_UB_acc',jet_setup.L_p_acc/jet_acc.energetic_dict['jet_L_B'])
    #print('z_acc_start from jet_acc=%e'%(jet_acc.parameters.R_H.val-jet_acc.parameters.R.val))
    #print('z_inj from radio_jet=%e'%((jet_acc.parameters.R_H.val-jet_acc.parameters.R.val)*composite_model.radio_jet.parameters.z_inj_frac.val))
    print('composite_model.radio_jet.gamma_cut',composite_model.radio_jet.gamma_cut)
    jet_acc.energetic_report_table.pprint_all()


def gobal_fit(composite_model,sed_data,jet_setup,nu1,nu2,minimizer='minuit'):
    from jetset.model_manager import  FitModel
    from jetset.minimizer import fit_SED,ModelMinimizer
    model_minimizer_minuit=ModelMinimizer(minimizer)
    best_fit_minuit=model_minimizer_minuit.fit(composite_model,
                                         sed_data,
                                         nu1,
                                         nu2,
                                         fitname='composite_fit',
                                         repeat=1)

    #composite_model.hump.name='Comp. hump'
    composite_model.disk_irr.name='Disk. Irr. Comp.'
    #add_jet_acc(composite_model)
    
    composite_model.radio_jet.name='radio jet'

    composite_model.name='best fit model'
    composite_model.set_nu_grid(nu1,nu2,500)

    composite_model.nu_min_fit=nu1
    composite_model.nu_max_fit=nu2
    composite_model.eval()


    final_report(composite_model,jet_setup)
    
    return model_minimizer_minuit,best_fit_minuit

def save_fit_results(name,composite_model,model_minimizer_minuit,best_fit_minuit):
    composite_model.save_model('%s.pkl'%name)
    model_minimizer_minuit.save_model('%s_model_minimizer.pkl'%name)
    best_fit_minuit.save_report('%s.txt'%name)


def build_data(data_table_file,cosmo,obj_name='MAXI 1820+070',bin_width=None,systematcis_list=[],NH=1.3E21,deabs=False):
    data=Data.from_file(data_table_file)
    if deabs is True:
        phabs = PhabsModel(cosmo=cosmo)
        phabs.parameters.N_H.val=NH
        x=data.table['x']
        y_abs=phabs.eval(nu=x,get_model=True)
        data.table['y']=data.table['y']/y_abs
    sed_data=ObsData(data_table=data,cosmo=cosmo,obj_name='deabs jetset')
    if bin_width is not None:
        sed_data.group_data(bin_width=bin_width)
    for systematcis_l in systematcis_list:
        sed_data.add_systematics(systematcis_l[0],nu_range=systematcis_l[1])

    return sed_data

def conv_to_seddata_atel(d):
    l=d[0]
    F_nu_obs1=d[1]
    F_nu_obs2=d[2]
    nu=l.to('Hz',equivalencies=u.spectral())
    f1=F_nu_obs1.to('erg cm-2 s-1 Hz-1')*nu
    f2=F_nu_obs2.to('erg cm-2 s-1 Hz-1')*nu
    return nu.value, f1.value, f2.value, ((f1+f2)*.5).value,((f1+f2)*.5).value*0.1



def set_disk_irr_model(cosmo,sed_data):
    composite_model=FitModel(cosmo=cosmo,name='DiskIrrComp')
    F=1.5

    #plc_model=PlcModel(name='hump',cosmo=cosmo)
    #comptonization_model=CompModel(name='comptonization',cosmo=cosmo)
    disk_irr=DiskIrr (cosmo=cosmo,name='disk_irr',compth=True,hump=True)

    composite_model.nu_min_fit=1E15
    composite_model.nu_max_fit=1E21

    #composite_model.add_component(phabs)
    #composite_model.add_component(comptonization_model)


    composite_model.add_component(disk_irr)
    #composite_model.composite_expr='(comptonization + soft_excess + hump)'


    composite_model.disk_irr.parameters.set('T_Disk',val=1.55E6)
    composite_model.disk_irr.parameters.T_Disk.frozen=True
    composite_model.disk_irr.parameters.set('L_Disk',val=1.1E37)
    composite_model.disk_irr.parameters.L_Disk.fit_range=[5E36,5E37]

    composite_model.disk_irr.parameters.set('r_out',val=5E3)
    composite_model.disk_irr.parameters.set('r_irr',val=1.1)
    composite_model.disk_irr.parameters.r_irr.frozen=True
    composite_model.disk_irr.parameters.set('theta',val=0)
    composite_model.disk_irr.parameters.theta.frozen=True
    #composite_model.disk_irr.parameters.theta.fit_range=[50,70]

    composite_model.disk_irr.parameters.set('L_Comp_ratio',val=4.5)
    composite_model.disk_irr.parameters.L_Comp_ratio.fit_range=[4,5]

    composite_model.disk_irr.parameters.set('z',val=0)
    composite_model.disk_irr.parameters.set('f_in',val=0.1)
    composite_model.disk_irr.parameters.f_in.frozen=True

    composite_model.disk_irr.parameters.set('f_out',val=0.01)


    composite_model.disk_irr.parameters.L_Disk.fit_range=[1E36,1E39]
    composite_model.disk_irr.parameters.z.frozen=True




    composite_model.disk_irr.parameters.Eb_Comp.val=0.15
    composite_model.disk_irr.parameters.Eb_Comp.fit_range=[0.1,0.5]
    composite_model.disk_irr.parameters.Eb_Comp.frozen=True

    composite_model.disk_irr.parameters.Ec_Comp.val=140
    composite_model.disk_irr.parameters.Ec_Comp.fit_range=[50,200]

    composite_model.disk_irr.parameters.Gamma_Comp.val=1.65
    composite_model.disk_irr.parameters.Gamma_Comp.fit_range=[1.6,1.8]
    composite_model.disk_irr.parameters.alpha_Comp.val=1.0
    composite_model.disk_irr.parameters.alpha_Comp.frozen=True


    #composite_model.add_component(plc_model)
    composite_model.disk_irr.parameters.Gamma_hump.val=-1.2
    composite_model.disk_irr.parameters.Gamma_hump.fit_range=[-2,2.0]

    composite_model.disk_irr.parameters.K_hump.val=0.00015
    composite_model.disk_irr.parameters.K_hump.fit_range=[0.00001,0.001]

    composite_model.disk_irr.parameters.Ec_hump.val=20
    composite_model.disk_irr.parameters.Ec_hump.fit_range=[15,35]

    composite_model.disk_irr.parameters.alpha_hump.val=1.0
    composite_model.disk_irr.parameters.alpha_hump.frozen=True


    #plc_model.parameters.freeze_all()


    composite_model.set_nu_grid(1E14,1E21,500)
    composite_model.eval()
    p=composite_model.plot_model(sed_data=sed_data,frame='obs')
    p.sedplot.axes.axvline(np.log10(7*2.4E17))
    p.rescale(y_max=-7,y_min=-12,x_min=13,x_max=20.5)
    composite_model.show_pars()
    print(composite_model.disk_irr.L_D/composite_model.disk_irr.parameters.L_Disk.val)

    return composite_model