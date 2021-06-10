import numpy as np
from matplotlib import pylab as plt
from jetset.jetkernel import jetkernel 

from astropy import units as u
from astropy import constants as const
from astropy import time

class JetSetup(object):
    
    
    def __init__(self,
                M_BH,
                beta,
                q_jet,
                eps_B0,
                eps_Up_UB_acc,
                z0_frac,
                z_0_ratio,
                m_B,
                m_R,
                m_N,
                nu_t,
                Fnu,
                dl,
                z_radio_end,
                p,
                N_e_p_ratio):
        
        self.BulkFactor=1.0/np.sqrt(1-beta**2)
        self.beta=beta
        self.m_B=m_B
        self.m_R=m_R
        self.m_N=m_N
        self.q_jet=q_jet
        self.dl=dl
        self.Fnu=Fnu
        self.nu_t=nu_t
        self.N_e_p_ratio=N_e_p_ratio
        self.p=p
        self.eps_B0=eps_B0
        self.eps_Up_UB_acc=eps_Up_UB_acc
        self.M_BH=M_BH
        self.Rs=jetkernel.eval_R_Sw(M_BH)
        
        self.z0_frac=z0_frac
        self.z_0_ratio=z_0_ratio
        self.z_0=self.Rs*z0_frac
        self.R_0=self.z_0*z_0_ratio
        
        self.theta_open_angle=np.rad2deg(np.arctan(self.R_0/self.z_0))
        
        self.L_Edd=1.3E38*(M_BH)*0.5
        #self.B_0=np.power(B_acc_target,(1.0/m_B))*(z_acc_target/self.z_0)
        #self.L_B_0=self.eval_L(self.eval_U_B(self.B_0),self.R_0,self.BulkFactor,beta)
        #self.L_jet=self.L_B_0/eps_B0
        self.L_jet=self.L_Edd*q_jet
        #print(self.L_B_0)
        self.B_0=self.eval_B0(self.L_jet,eps_B0,self.R_0,beta,self.BulkFactor)
        self.L_B_0=self.eval_L(self.eval_U_B(self.B_0),self.R_0,self.BulkFactor,beta)
        
        self.z_acc_end=self.eval_z_acc(Fnu,dl,self.z_0,self.B_0,self.R_0,self.m_B,self.m_R,self.p,self.nu_t)
        
        R_acc_end=self.jet_radius(self.R_0,self.z_acc_end,self.z_0,m_R)

        self.z_acc_start=self.z_acc_end-2*R_acc_end

        self.R_acc_start=self.jet_radius(self.R_0,self.z_acc_start,self.z_0,m_R)

        self.R_acc_end=self.jet_radius(self.R_0,self.z_acc_end,self.z_0,m_R)

        self.R_acc=(self.R_acc_start+self.R_acc_end)*0.5

        self.B_acc_start=self.B_scaling(self.B_0,self.z_acc_start,self.z_0,m_B)

        self.B_acc_end=self.B_scaling(self.B_0,self.z_acc_end,self.z_0,m_B)

        self.R_acc=(self.R_acc_start+self.R_acc_end)*0.5
        
        self.B_acc=(self.B_acc_start+self.B_acc_end)*0.5

        self.z_acc=(self.z_acc_start+self.z_acc_end)*0.5


        self.L_B_acc=self.eval_L(self.eval_U_B(self.B_acc),self.R_acc,self.BulkFactor,beta)
       
        
        #self.z_acc=self.eval_R_H_acc(self.z_0,self.N_p_0,self.BulkFactor,self.B_0,m_B,m_N)
        #self.z_acc=self.eval_z_acc(self.z_0,self.R_0,self.R_acc,self.m_R)
        
        


        self.L_p_acc=self.L_B_acc*self.eps_Up_UB_acc
        self.N_p_acc=self.eval_N_p(self.L_p_acc,self.BulkFactor,self.R_acc,beta)
        self.N_p_0=self.eval_N_p_0(self.N_p_acc,self.z_acc,self.z_0,self.m_N)
        
        self.U_p_0=self.N_p_0*const.m_p.to('g').value*const.c.to('cm s-1').value**2
        self.L_p_0=self.eval_L(self.U_p_0,self.R_0,self.BulkFactor,beta)
        
        print('z_0=%e'%self.z_0)
        print('R_0=%e'%self.R_0)
        print('theta_open=%e'%self.theta_open_angle)
        print('beta=%e'%self.beta)
        print('BulkFactor=%e'%self.BulkFactor)
        #print('B_0=%e'%self.B_0_1)a_open=%e'%self.theta_open_angle)
        print('B_0=%e'%self.B_0)
        print('(1/2)L_Edd=%e'%self.L_Edd)
        print('N_p_acc=%e'%self.N_p_acc)
        print('N_p_0=%e'%self.N_p_0)
        print('L_B_0=%e'%self.L_B_0)
        print('L_B_acc=%e'%self.L_B_acc)
        print('L_p_acc=%e'%self.L_p_acc)
        print('L_p_0=%e'%self.L_p_0)
        print('L_B_0/L_p_0=%e'%(self.L_B_0/self.L_p_0))
        print('L_B_acc/L_p_acc=%e'%(self.L_B_acc/self.L_p_acc))
        #print('N_p_0=%e'%self.N_p_0)
        #print('(L_B_0+L_p_0)/L_jet=%e'%((self.L_B_0+self.L_p_0)/(self.L_jet)))
        #print('z_acc traget=%e'%z_acc_target)
        print('z_acc=%e'%self.z_acc)
        print('z_acc=%e (Rs)'%(self.z_acc/self.Rs))
        print('z_acc_start=%e'%self.z_acc_start)
        print('z_acc_end=%e'%self.z_acc_end)
        
        print('R_acc=%e'%self.R_acc)
        print('R_acc_start=%e'%self.R_acc_start)
        print('R_acc_end=%e'%self.R_acc_end)
        #print('B_acc target=%e'%B_acc_target)
        print('B(z_acc_start)=%e'%self.B_acc_start)
        print('B(z_acc_endt)=%e'%self.B_acc_end)
        print('B(z_acc)=%e'%self.B_acc)
       
        self.z_radio_end=z_radio_end
        self.z_radio_start=self.z_acc_end
        self.z_radio_frac=self.z_radio_end/self.z_radio_start 
        print('z_radio_start=%e'%self.z_radio_start)
        print('z_radio_end=%e'%self.z_radio_end)
        print('z_radio_frac=%e'%self.z_radio_frac)
        
        self.t_reach_z_acc=self.z_acc/(self.BulkFactor*self.beta*const.c.to('cm s-1').value)
        print('time to reach z_acc=%e'% self.t_reach_z_acc)
        
        self.t_reach_z_radio=self.z_radio_start/(self.BulkFactor*self.beta*const.c.to('cm s-1').value)
        print('time to reach z_radio_start=%e'% self.z_radio_start)
        
        #N_e_acc=self.eval_N(self.N_p_0,self.z_0,self.z_acc,self.m_N)*self.N_e_p_ratio
        #self.nu_syn_abs_acc=self.nu_abs_sync(B=self.B_acc,p=self.p,R=self.R_acc,N=N_e_acc)
        
        #print('nu_syn_abs_acc=%e'%self.nu_syn_abs_acc)
    
    def fp_k(self,p):
        return 3**((p+1)/2) * ((1.8/p**(0.7))+((p**2)/40))
   
    def fp_e(self,p):
        return 3**(p/2) * ((2.25/p**(2.2))+0.105)

    def nu_abs_sync(self,B,R,N,p):
        q_esu=4.803206815e-10
        mec=8.187111e-07/3E10
        nu_L=(q_esu*B)/(2*np.pi*mec)   
        a=(np.pi*np.sqrt(np.pi))/4
        b=q_esu*R*N*self.fp_k(p)/B
        c=(a*b)**(2/(p+4))

        return nu_L*c


    def eval_z_acc(self,Fnu,dl,z0,B0,R0,m_B,m_R,p,nu_t):
        q_esu=4.803206815e-10
        c=2.99792458e+10
        mec=8.187111e-07/c
        SIGTH=6.652461618e-25
        #nu_L=(q_esu*self.B_scaling(B0,))/(2*np.pi*mec)   
        a=np.power(nu_t*2*np.pi*mec/(q_esu*B0*z0**m_B),((p+4)/2))
        b=4 * B0**2 * R0**2 * np.power(z0,2*(m_B-m_R))
        c1=np.pi*np.sqrt(np.pi)*q_esu*self.fp_k(p)*Fnu*dl**2 
        c2=3*SIGTH*c/(np.pi**2 * 8 * 16* np.sqrt(np.pi))*self.fp_e(p)*2*np.pi*mec/q_esu
        gamma=2.0/(4*(m_B-m_R)-m_B*(p+4))
        x=a*b/c1*c2
        return np.power(x,gamma)

    
    def eval_N_p(self,L_p,BulkFactor,R,beta):
        return L_p/(BulkFactor**2 * np.pi * R**2 * beta * const.c.to('cm s-1').value * const.m_p.to('g').value*const.c.to('cm s-1').value**2)
    
    def eval_N_p_0(self,N_p,R,R0,m_R):
        return N_p*np.power((R0/R),2*m_R)

    def eval_L(self,U,R,BulkFactor,beta):
        return np.pi*self.BulkFactor**2 * R**2 * beta*const.c.to('cm s-1').value*U
    
    def eval_U_B(self,B):  
        return B**2/(np.pi*8)
    

    def eval_U_p_cold(self,N_p):
        return N_p*const.m_p.to('g').value*const.c.to('cm s-1').value**2

    def B_scaling(self,B_0,z,z_0,m_B):
        return B_0*(z_0/z)**m_B
    
    def eval_B0(self,L_jet,eps_B0,R_0,beta,BulkFactor):
        print(L_jet*eps_B0)
        return np.sqrt(8*np.pi*L_jet*eps_B0/(BulkFactor**2 * np.pi * R_0**2 * beta*const.c.to('cm s-1').value))

    def jet_radius(self,R_0,z,z_0,m_R):
        return R_0*np.power((z/z_0),m_R)


    #def eval_R_H_acc(self,z_0,N_p_0,BulkFactor,B0,m_B,m_N):
    #    c=(N_p_0*const.m_p.to('g').value*const.c.to('cm s-1').value**2*8*np.pi)**0.5/B0
    #    alfa=(m_B-m_N*0.5)
    #    x= np.power(c,1/alfa)
    #    return z_0/x

    #def eval_z_acc(self,z_0,R_0,R_acc,m_R):
    #    return np.power(R_acc,(1/m_R))*z_0/R_0

    def eval_N(self,N_0,z0,z,m_N):
        return N_0*(z0/z)**m_N
    
    def gamma_eq_e_MaxwellJuttner(self,T):
        return const.k_B*T/(const.m_e*const.c**2)

    def gamma_eq_p_MaxwellJuttner(self,T):
        return const.k_B*T/(const.m_p*const.c**2)


    def T_MaxwellJuttner(self,gamma):
        return const.m_e*const.c*const.c*gamma/const.k_B

    
    
    def plot_z_acc(self):

        fig = plt.figure(dpi=180)
        ax= fig.add_subplot(111)
        z=np.logspace(np.log10(self.z_0),np.log10(self.z_radio_end),100)
        B=self.B_scaling(self.B_0,z,self.z_0,m_B=self.m_B)
        R=self.jet_radius(self.R_0,z,self.z_0,self.m_R)
        U_B=(B**2/(8*np.pi))
        N_p=self.eval_N_p(self.L_p_acc,self.BulkFactor,R,self.beta)
        U_p=self.eval_U_p_cold(N_p)
        ax.loglog(z,U_B,label='$U_B$ index=%f'%self.m_B)
        ax.loglog(z,U_p,label='$U_p^{cold}$')
        ax.axvline(self.z_acc,lw=0.5,c='green')

        sec_ax=ax.secondary_xaxis('top',functions=(lambda x:x/self.Rs, lambda x: x*self.Rs) )
        sec_ax.set_xlabel('$R_{H}$ $(R_g)$')
        #ax2 = ax.twinx()

        ax.legend()
        ax.grid()
        ax.set_xlabel('$R_{H}$ (cm)')
        #ax2.set_ylabel('$B$ (G)')
        ax.set_ylabel('$U$ (erg/cm$^3$)')
        
        return fig
    
    def plot_jet_sections(self,m_B_radio=None):
        fig = plt.figure(dpi=180)
        ax= fig.add_subplot(111)
        print(self.z_0,self.z_radio_end)
        z=np.logspace(np.log10(self.z_0),np.log10(self.z_radio_end),100)
        ax.loglog(z,self.B_scaling(self.B_0,z,self.z_0,self.m_B),label='$B, m_B=%2.2f$'%self.m_B,c='blue')
        #ax.loglog(z,self.B_scaling(z,m=1.0),label='$B, m_B=%2.2f$'%1.0)


        ax.axvspan(self.z_0,self.z_acc,alpha=0.3,color='violet',label='pre-acc',lw=None)
        #ax.axvline(h_0+cylinder_height(h_0),lw=0.5,label='noozle',c='violet')

        ax.axvspan(self.z_acc,self.z_radio_start,alpha=0.3,color='blue',label='acc',lw=None)
        #ax.axvline(z_acc+cylinder_height(z_acc),lw=0.5,label='acc.',c='green')

        ax.axvspan(self.z_radio_start,self.z_radio_end,alpha=0.3,color='green',label='radio',lw=None)
        #ax.axvline(z_radio+cylinder_height(z_radio),lw=0.5,label='radio',c='cyan')



        ax.set_xlabel('$R_{H}$ (cm)')
        ax.set_ylabel('$B$ (G)')
        sec_ax=ax.secondary_xaxis('top',functions=(lambda x:x/self.Rs, lambda x: x*self.Rs) )
        sec_ax.set_xlabel('$R_{H}$ $(R_g)$')
        ax2 = ax.twinx()
        #R_0,z,z_0,m_R
        ax2.loglog(z,self.jet_radius(self.R_0,z,self.z_0,self.m_R),label='jet cross section',c='orange')
        ax2.set_ylabel('jet radius R (cm)', color='orange')
        
        #ax2.legend()
        if m_B_radio is not None :            
            z_0_radio=self.z_radio_start
            B_0_radio=self.B_scaling(self.B_0,z_0_radio,self.z_0,self.m_B)
            R_0_radio=self.jet_radius(self.R_0,z_0_radio,self.z_0,self.m_R)
            z=np.logspace(np.log10(self.z_radio_start),np.log10(self.z_radio_end),100)
            ax.loglog(z,self.B_scaling(B_0_radio,z,z_0_radio,m_B_radio),label='$ B_{radio}, m_{radio}=%2.2f$'%m_B_radio,ls='--',c='blue')
            ax2.loglog(z,self.jet_radius(R_0_radio,z,z_0_radio,m_B_radio),label='jet cross section',c='orange',ls='--')
        
        plt.tight_layout()
        ax.legend(loc='center left',fontsize=7)
        return fig
    
    
def main():
    jet_setup=JetSetup(M_BH=8,
                    beta=0.89,
                    q_jet=0.5,
                    eps_B0=1.0,
                    eps_Up_UB_acc=1.1,
                    z0_frac=50,
                    z_0_ratio=0.1,
                    m_B=1.1,
                    m_R=1.0,
                    m_N=2.0,
                    nu_t=2E12,
                    Fnu=1E-23,
                    dl=9.25E21,
                    z_radio_end=1E15,
                    p=1,
                    N_e_p_ratio=10)


    #jet_setup.plot_jet_sections()
    #jet_setup.plot_z_acc()
    #plt.show()

if __name__ == "__main__":
    main() 