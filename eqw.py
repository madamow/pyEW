#!/home/puccini/adamow/yt-x86_64/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, copy
from scipy import ndimage
import scipy.optimize as so
from scipy.signal import argrelextrema
from scipy.special import erf
from scipy.special import wofz
import ConfigParser
import sys
plt.switch_backend('qt4Agg')

#####################################################
#Print on screen and write to files functions
#####################################################
def print_and_log(logfile,list_of_inps):
    s = ' '.join(map(str, list_of_inps))
    print s
    logfile.write(s+"\n")

def moog_entry(l,ew,eew):
    moog= "%10s%10s%10s%10s%10s%10s%10.2f %6.3e \n" % \
          (l[0],l[1],l[2],l[3],'','',ew,eew)
    return moog
            
def print_line_info(rslt):
    fit_labels={'mg':'multi Gauss',
                'sg':'part of mGauss','g':'Gauss','v':'Voigt'}
    for fit in rslt:
        finfo= "%15s %s %4.2f %s %f %s %f" % \
        (fit_labels[fit],": EW =" ,rslt[fit][2],"eEW =",rslt[fit][3],"o-c:",rslt[fit][4])
        print_and_log(logfile,[finfo])

def print_mgauss_data(rslt):
    #print full data for all lines 
    #fitted with multi Gauss function
    mg_params=rslt['mg'][1]
    print_and_log(logfile,["\n",mg_params.shape[0],
                           "lines in multi gaussian fit:"])
    for gfit in  mg_params:
        ew=get_gew(gfit[1],gfit[2])
        #Info about lines in multigaussian fit
        info= "%4.2f %s%4.2f %s%4.3f %s%4.2f %s%4.2f" % \
              ( gfit[0], "depth=", gfit[1], \
               "FWHM=",gfit[2],\
               "EW=",ew,\
               "RW=", np.log10(0.001*ew/gfit[0]))
        print_and_log(logfile,[info])

def moog_output(out_file,a_line,lr):       
    #Write to output file
    moog= moog_entry(a_line,lr[3],lr[4])        
    out_file.write(moog)
    print_and_log(logfile,["\nMoog entry:\n",moog])

####################################################
#Create output files
####################################################
def create_out_files(infile_name,list_name):
    print file_name
    file_name_out=file_name.split("/")[-1].split(".")[0]
    
    logfile = open(file_name_out.split(".")[0]+"_EW.log",'w')
    print_and_log(logfile,["Line list:",line_list_file])
    
    list_name=line_list_file.split('.')[0]
    out_file=open("moog_"+list_name+"_"+file_name_out.split(".")[0]+".out",'wb')
    out_file.write(file_name_out+"\n")
    
    return logfile,out_file
######################################################
#Spectrum preparation
######################################################
def do_linear(spectrum):
    #Transforms spectrum to linear scale
    first=spectrum[0,0]
    last=spectrum[-1,0]	
    res=(last-first)/spectrum.shape[0]
    flux_lin=np.array(([]))
    x=np.arange(first, last,res)
    y=np.interp(x,spectrum[:,0],spectrum[:,1])	
    out=np.transpose(np.vstack((x,y)))
    return out,res

def correct_continuum(spec,rejt):  
    stop=False
    tab=np.copy(spec)
    p_no=0
    iter=0.0
    while stop==False:
        iter = iter+1.
        p_no_prev=p_no
        ft=np.polyfit(tab[:,0],tab[:,1],2)
        tab= spec[np.where(spec[:,1]>rejt*np.polyval(ft,spec[:,0]))]
        p_no=tab.shape[0]
    
        if (p_no==p_no_prev or iter>100.):
            stop=True
    if p_no!=0 and iter<100.:
        cont=spec[:,1]/np.polyval(ft,spec[:,0])
        spec[:,1]=cont
    else:
        print "Fail when correcting continuum"
        #This is for cases where line is located 
        #i.e. in a break of echelle spectrum    
    return spec,iter,p_no

def smooth(y, box_pts):
    #smooths function, recipe from StackOverflow
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def read_cont_sec(config):
    secs=[]
    for elem in config.items('cont_sec'):
        secs.append(elem[1].split(" "))
    return secs

def auto_rejt(tab,config):
    rejt_reg=np.array(read_cont_sec(config),dtype=float)
    rejt_tab=[]
    
    for item in rejt_reg:
        r_low,r_up=item
        reg,rres=do_linear(file[np.where((tab[:,0]>r_low) &(tab[:,0]<r_up))])
        reg_smo= ndimage.gaussian_filter1d(reg[:,1],sigma=s_factor,mode='wrap')
        rejt_tab.append(np.std(reg[:,1]-reg_smo))
    
    rejt=1. - np.median(rejt_tab)
    SN=1. / (1.-rejt)    

    print_and_log(logfile,["signal to noise:", SN])
    print_and_log(logfile,[ "rejt parameter:", rejt])
    
    return rejt, SN

######################################################
#Finding strong lines
######################################################
def find_derivatives(x,f,dx,s_factor):
    dxdxdx=dx*dx*dx 
    #First derivative
    gf = ndimage.gaussian_filter1d(f, sigma=s_factor, order=1, mode='wrap') / dx
    #Second derivative
    ggf = ndimage.gaussian_filter1d(f, sigma=s_factor, order=2, mode='wrap') / (dx*dx)
    #Third derivative
    gggf = np.array(ndimage.gaussian_filter1d(f, sigma=s_factor, order=3, mode='wrap') / dxdxdx)
    
    return gf,ggf,gggf


def find_inflection(x,y):
   #This function finds zero points for set of data
   #It uses linear interpolation between two points with
   #different signs. Then it selects points which change their sign
    #from + to -, and form - to +
   to_minus=[]
   to_plus=[]
   for i in np.arange(0,len(y)-1,1):
       a=(y[i+1]-y[i])/(x[i+1]-x[i])
       b=y[i]-a*x[i]
       if (y[i]>0.0 and y[i+1]<0.0):
           to_minus.append(-b/a)
       elif (y[i]<0.0 and y[i+1]>0):
           to_plus.append(-b/a)
   to_minus=np.array(to_minus)
   to_plus=np.array(to_plus)
   
   return to_minus,to_plus

def find_strong_lines(x,tab,xo,r_lvl,SN):
    #We need to find strong lines in spectrum
    #and ingnore all small changes in flux.
    #Here - code checks the change in signal 
    #around inflection points and compares it 
    #to noise multiplied by rejection parameter 
     
    max_ind= argrelextrema(tab, np.greater)
    min_ind= argrelextrema(tab, np.less)

    max_tab=np.array([x[max_ind],tab[max_ind]])
    min_tab=np.array([x[min_ind],tab[min_ind]])
    
    thold=np.std(tab)*r_lvl
    str_lines=[]
    if not (max_tab.size!=0 and min_tab.size!=0 and thold==0.0):
        for item in  xo:
            indx= np.abs(max_tab[0,:]-item).argmin()
            indm= np.abs(min_tab[0,:]-item).argmin()
        
            if ((np.abs(max_tab[1,indx])>thold) and (np.abs(min_tab[1,indm])>thold)):
                str_lines.append(item)
            
    return str_lines,thold

def evaluate_lines(line,strong_lines,det_level,gggf_infm):
    #check if our line was identified
    if (len(strong_lines)==0 and  (np.abs(gggf_infm-line).min()>det_level)):
        print line, "was not detected"
    elif (len(strong_lines)==0 and (np.abs(gggf_infm-line).min()<det_level)):#no strong lines detected
        print "I see no strong lines here,but weak line close to", line,"was detected"
        strong_lines.append(line) 
    elif (len(strong_lines)>0 and np.abs(gggf_infm-line).min()<det_level and np.abs(strong_lines-line).min()<det_level):
        print "line",line," was detected and it was classified as a strong line"
        pass
    elif (len(strong_lines)>0 and (np.abs(gggf_infm-line).min()<det_level) and np.abs(strong_lines-line).min()>det_level):
        print "I see this line at",line,", but it is weak"
        strong_lines.append(line)
    else:
        print line, "was not detected" 
    return strong_lines


######################################################
#Errors for parameters of the fit
######################################################
def leastsq_errors(fit_tab,p_no): #so.leastsq result+no of parameters fitted
    #use so.leastsq output to estimate error parameter
    pcov=fit_tab[1]

    if pcov is None:
        row_col=((len(fit_tab[0])/p_no)*p_no)
        print "Covariance matrix is empty"
        errs_matrix=np.ones((row_col,row_col))
        errs_matrix[:,:]=1000.0 #make errors huge 
    else:
        sq=np.sum(fit_tab[2]['fvec']**2)/(len(fit_tab[2])-len(fit_tab[0]))        
        errs_matrix= sq*pcov
        
    i=np.arange(len(fit_tab[0]))
    f_errs= np.reshape(errs_matrix[i,i]**2,((len(fit_tab[0])/p_no),p_no))
    
    return f_errs

######################################################
#Gaussian fitting
######################################################
AGAUSS = -4.0 * np.log(2.0)

def gaus(x_in,a,x0,fwhm):
    return a * np.exp(AGAUSS * ((x_in - x0) / fwhm)**2)

def multiple_gaus(x,params):
    mg=np.zeros_like(x)
    for row in params:
        x0,a,fwhm=row
        mg=mg+gaus(x,a,x0,fwhm)
    return mg

def res_g(p,data):
    x,y=data
    a,x0,fwhm=p
    sg=gaus(x,a,x0,fwhm)
    err=1./np.abs(y)
    return (y-sg)/err
    
def res_mg(p,x,y,nb):
    params=np.reshape(p,(nb,3))
    mg=multiple_gaus(x,params)
    err=1./np.abs(y) 
    return (y-mg)/err
    
def get_gew(ag,fwhmg):
    #calculate EW for gaussian profile
    gew=500.*ag*np.sqrt(np.pi/np.log(2))*fwhmg
    return gew

def fit_single_Gauss(x,f,a1,x01,fwhm1):
    gaus_p = so.leastsq(res_g,[a1,x01,fwhm1],
             args=([x,1.0-f]),
             full_output=1)
    a1s,x01s,fwhm1s=np.abs(gaus_p[0])

    eqw_gf=get_gew(a1s,fwhm1s)
    
    gf_errs=leastsq_errors(gaus_p,3)
    
    eqw_gf_err = eqw_gf*(gf_errs[0,0]/a1s+gf_errs[0,2]/fwhm1s)
    return eqw_gf,eqw_gf_err,gaus_p[0]

def fit_multi_Gauss(x,f,strong_lines):
    params=np.ones((len(strong_lines),3))
    params[:,0]=strong_lines #first columns = x0
    params[:,2]=0.05 #starting value for gauss fit
        
    new_params=np.array([])

    #First run
    while params.shape[0]!=new_params.shape[0] and params.shape[0]>0.:
        plsq=so.leastsq(res_mg, params,
             args=(x,1.0-f,params.shape[0]),full_output=1)
        new_params= np.reshape(plsq[0],(params.shape[0],3))
        #evaluate  run
        ind=np.where(np.abs(strong_lines-new_params[:,0])<det_level)
        strong_lines=np.array(strong_lines)[ind]
        params=new_params[ind]

        
    mg_errs=leastsq_errors(plsq,3) #fit quality

    return np.abs(params),mg_errs


######################################################
#Voigt fitting
######################################################
def voigt(x, y):
   # The Voigt function is also the real part of 
   # w(z) = exp(-z^2) erfc(iz), the complex probability function,
   # which is also known as the Faddeeva function. Scipy has 
   # implemented this function under the name wofz()
    z = x + 1j*y
    I = wofz(z).real
    return I

def Voigt(nu, alphaD, alphaL, nu_0, A, a, b):
   # The Voigt line shape in terms of its physical parameters
    #alphaD, alphaL half widths at half max for Doppler and Lorentz(not FWHM)
    #A - scaling factor
    f = np.sqrt(np.log(2))
    x = (nu-nu_0)/alphaD * f
    y = alphaL/alphaD * f
    backg = a + b*nu 
    V = A*f/(alphaD*np.sqrt(np.pi)) * voigt(x, y) + backg
    return V

def funcV(p, x):
    # Compose the Voigt line-shape
     a=0.
     b=0.
     alphaD,alphaL,nu_0,I=p
     return Voigt(x, alphaD, alphaL, nu_0, I, a, b)

def res_v(p, data):
   # Return weighted residuals of Voigt
    x, y, err = data
    err=1./np.abs(y)
    return (y-funcV(p,x)) / err

def fit_Voigt(x,f,x01):
    A_voigt=0.1
    alphaD=0.01
    alphaL=0.01
    nu_0=x01
    pv0=[alphaD, alphaL, nu_0, A_voigt]
                
    voigt_p = so.leastsq(res_v,pv0,
              args=([x,1.0-f,np.ones_like(x)]),
              full_output=1)
                               
    alphaD,alphaL, nu_0, A_voigt=np.abs(voigt_p[0])

    I=np.abs(voigt_p[0][3])*1000.
    v_errs=leastsq_errors(voigt_p,4)[0][3]

    return I,v_errs,voigt_p[0]

def voigt_fwhm(alphaD,alphaL):
    c1 = 1.0692
    c2 = 0.86639
    v_fwhm = c1*alphaL+np.sqrt(c2*alphaL**2+4*alphaD**2)
    return v_fwhm 
    
######################################################
#Other
######################################################
def pm_width(x,x01,s1): #s1 is fwhm
    iu=np.abs(x-x01-w_factor*np.abs(s1)).argmin()
    il=np.abs(x-x01+w_factor*np.abs(s1)).argmin()
        
    if iu==il or np.abs(iu-il)<3.:
        il=0
        iu=len(x)            
    return il,iu

def append_to_dict(tab,lbl,fc,param,eqw,eqw_err,soc):
    tab[lbl].append(fc)
    tab[lbl].append(param)
    tab[lbl].append(eqw)
    tab[lbl].append(eqw_err)
    tab[lbl].append(soc)
    return tab

def find_eqws(line,x,f,strong_lines):
    results = {'mg':[], 'sg':[],'g':[],'v':[]}    
    #Fit multiple gaussian profile
    params, mg_errs = fit_multi_Gauss(x,f,strong_lines)

    if params.shape[0]==0:
        print "Line ",line, "was not detected"
        params=np.array([[-9.9,-9.9,-9.9]])

    mgaus=multiple_gaus(x,params)
    
    ip= np.abs(params[:,0]-line).argmin()
    x01,a1,s1= params[ip,:]    
    
    eqw=get_gew(a1,s1)
    eqw_err= eqw*(mg_errs[ip,0]/a1+mg_errs[ip,2]/s1)
    
    #Calculate single gauss profile 
    #(this gauss is a part of multigaussian fit)
    sgaus=gaus(x,a1,x01,s1)
    sparams=[a1,x01,s1]    

    #Determine region close to gaussian line center#
    il, iu = pm_width(x,x01,s1)
    
    oc_mg=np.std(np.abs(f[il:iu]-1.0+mgaus[il:iu]))
    oc_sg=np.std(np.abs(f[il:iu]-1.0+sgaus[il:iu]))
    
    results=append_to_dict(results,'mg',mgaus,params,eqw,eqw_err,oc_mg)
    results=append_to_dict(results,'sg',sgaus,sparams,eqw,eqw_err,oc_sg)                                                         

    #Fit single Gauss and Voigt profile
    eqw_gf,eqw_gf_err,gparams=fit_single_Gauss(x[il:iu],f[il:iu],a1,x01,s1)
    gausf=gaus(x,gparams[0],gparams[1],gparams[2])    
    oc_g=np.std(np.abs(f[il:iu]-1.0+gausf[il:iu]))
    results=append_to_dict(results,'g',gausf,gparams,eqw_gf,eqw_gf_err,oc_g)       
    
    try:
        I, v_errs,vpar=fit_Voigt(x[il:iu],f[il:iu],x01)
    except:
        I, v_errs,vpar=fit_Voigt(x,f,x01)
    svoigt=Voigt(x,vpar[0],vpar[1], vpar[2], vpar[3],0.,0.)    
    oc_v=np.std(np.abs(f[il:iu]-1.0+svoigt[il:iu]))
    results=append_to_dict(results,'v',svoigt,vpar,I,v_errs,oc_v)       

    return results
    
def evaluate_results(line,rslt,v_lvl,l_eqw,h_eqw,det_level):
    print_line_info(rslt)
    if rslt['mg'][3]>0.5*rslt['mg'][2]:
        print_and_log(logfile,["Huge error!", rslt['mg'][2],rslt['mg'][3]])
        hu=True
    else:
        hu=False
    
    if rslt['v'][4]<rslt['mg'][4] and rslt['v'][4]<rslt['g'][4] and np.log10(rslt['mg'][2]*0.001/line)>v_lvl:
        print_and_log(logfile,[ "using Voigt profile"])
        v_fwhm = voigt_fwhm(rslt['v'][1][1],rslt['v'][1][2])
        out = [line,rslt['v'][1][2],v_fwhm,rslt['v'][2],rslt['v'][3]]
 
    elif rslt['g'][4]<rslt['mg'][4] and np.log10(rslt['mg'][2]*0.001/line)<v_lvl:
        print_and_log(logfile,[ "using single Gauss fit"])
        out = [line,rslt['g'][1][1],rslt['g'][1][2],rslt['g'][2],rslt['g'][3]]

    elif hu==True and rslt['g'][4]<rslt['sg'][4]:
        print_and_log(logfile,[ "using single Gauss fit"])
        out = [ line,rslt['g'][1][1],rslt['g'][1][2],rslt['g'][2],rslt['g'][3]]
    else:
        out1=rslt['mg'][1][ np.abs(rslt['mg'][1][:,0]-line).argmin()]
        out= [ line,out1[0],np.abs(out1[2]),rslt['mg'][2],rslt['mg'][3]]
        
    if (out[3]>h_eqw or out[3]<l_eqw):
        print_and_log(logfile,[ "Line is too strong or too weak"])
        out[2] =  -99.9
        out[3] =  -99.9
        out[4] =  99.9

    if np.abs(line-out[1])>det_level:
        print_and_log(logfile,[ line,elem_id, "line outside the det_level range"])
        out[2] = -99.9
        out[3] = -99.9
        out[4] =  99.9 
        
    return out



#EW analysis
def EW_analysis(line,x,f,strong_lines,v_lvl,l_eqw,h_eqw,det_level):
    #do all fits: multi gauss, sgauss (part of multi gauss),
    #gauss fitted in small area, voigt fitted in small area
    r_tab = find_eqws(line,x,f,strong_lines) #results tab
    print_mgauss_data(r_tab)
    #Check if EW is reasonable
    lr=evaluate_results(line,r_tab,v_lvl,l_eqw,h_eqw,det_level)
    
    return r_tab,lr
    
######################################################
#Ploting module
######################################################        
class Plot_Module(object):        

    def __init__(self,ax,line,x,f,strong_lines,v_lvl,l_eqw,h_eqw,
                 det_level,thold):
        self.ax = ax
        self.strong_lines = strong_lines
        
        x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        self.ax[0].set_title("%s %s" % (a_line[0],a_line[1]))
        self.ax[0].xaxis.set_major_formatter(x_formatter)
        self.ax[0].axhline(1.0,color='g',label='continuum')
        self.ax[0].set_xlabel("Wavelenght")
        self.ax[0].plot(x,f,'o',color= 'k' , label='spectrum')
        self.ax[0].set_ylim(min(f)-0.1,1.+0.4*(1.+0.1-min(f)))
    
        self.ax[1].axhline(0)
        self.ax[1].plot(x,gggf,'b', label='3rd derivative')
        self.ax[1].set_ylim(np.min(gggf),np.max(gggf))
        self.ax[1].set_xlim(line-off,line+off)
        self.ax[1].plot(gggf_infm, np.zeros_like(gggf_infm),'o',color='b',
                     label='flex points + -> -')
        self.ax[1].axhline(-thold,c='r')
        self.ax[1].axhline( thold,c='r')
        

        self.lstyle=np.array([['mg','c','-','multi Gauss',4],
                              ['sg','b',':','line in mGauss',3],
                              [ 'g', 'y','-','Gauss',2],
                              [ 'v', 'm','-','Voigt',1]])

          
    def make_plot(self):   
        for lbl in self.r_tab:
            fit_style=np.squeeze(self.lstyle[np.where(self.lstyle[:,0]==lbl)])
            ax[0].plot(x,1.0-self.r_tab[lbl][0],
                   color = fit_style[1],
                   ls =    fit_style[2],
                   label = fit_style[3],
                   zorder= fit_style[4])            
        
        x01 = self.r_tab['sg'][1][1]
        s1  = self.r_tab['sg'][1][2]
        self.ax[0].axvspan(x01-w_factor*s1,x01+w_factor*s1,color='g',
                           alpha=0.25,label="eval. area")   
        self.ax[0].legend(loc=2,numpoints=1,fontsize='10',ncol=4)
        
        
        self.ax[0].axvline(x01,color='r',lw=1.5,label="line")
        for oline in self.r_tab['mg'][1][:,0]:
            ax[0].axvline(oline,c='r',zorder=1,label='strong lines')
            
        self.ax[1].plot(self.strong_lines,np.zeros_like(self.strong_lines),
                   'o',color='r',label="strong lines")
        self.ax[1].legend(loc=3,numpoints=1,fontsize='10',ncol=3)
         
    def clean_plot(self):
        plt.sca(self.ax[0])
        for artist in plt.gca().get_children():
            if hasattr(artist,'get_label') and (
                 artist.get_label() in self.lstyle[:,3] or  
                 artist.get_label()=="line" or  
                 artist.get_label()=='eval. area' or 
                 artist.get_label()=='strong lines' or 
                 artist.get_label()=='line_pnt'):
                artist.remove()        
                
        plt.sca(self.ax[1]) 
        for artist in plt.gca().get_children():
            if hasattr(artist,'get_label') and (
                       artist.get_label()=='strong lines' or 
                       artist.get_label()=='line_pnt'):
                artist.remove()
    
    def onclick(self,event):
        toolbar = plt.get_current_fig_manager().toolbar
        if event.button==1 and toolbar.mode=='':
            ind= np.abs((gggf_infm-event.xdata)).argmin()
            self.ax[1].plot(gggf_infm[ind],1.0,'o',color='r',mec='b',picker=5,
            label='line_pnt')
            self.ax[0].axvline(gggf_infm[ind],c='r',ls=":",zorder=1,
                               label='line_pnt')
            self.strong_lines.append(gggf_infm[ind])
            
        elif event.button==3 and toolbar.mode=='':
            ind = np.squeeze(np.where(
                  np.abs(self.strong_lines-event.xdata)<0.05))
            
            if ind:     
                self.ax[1].plot(self.strong_lines[ind],1.0,'o',
                                color='b',mec='r',
                                picker=5,label='line_pnt')
                self.ax[0].axvline(self.strong_lines[ind],c='b',ls=":",zorder=1,
                             label='line_pnt')
                self.strong_lines.pop(ind)
            else:
                print "Click closer to chosen point"
            
        plt.draw()

    def ontype(self,event):
        if event.key=='enter':
            self.clean_plot()
            self.update_plot()
            plt.draw()
      
        elif event.key=='q':
             exit()

        elif event.key=='w':
            print "Writing to output file..."
            moog_output(out_file,a_line,self.lr)
            print "Close the plot window to move on"
        
        else:
            #if you accidentally hit any other key
            pass
    
    def update_plot(self):
        self.strong_lines = sorted(list(set(self.strong_lines)))
        self.r_tab,self.lr = EW_analysis(line,x,f,self.strong_lines,
                            v_lvl,l_eqw,h_eqw,det_level)
                            
        self.strong_lines= list(self.r_tab['mg'][1][:,0]) #fitted strong lines
        self.make_plot()
        plt.draw()
        moog = moog_entry(a_line,self.lr[3],self.lr[4])
        print_and_log(logfile,["\nCurrent moog entry:\n",moog])
                                                
        plt.gcf().canvas.mpl_connect('key_press_event',self.ontype)
        plt.gcf().canvas.mpl_connect('button_press_event',self.onclick)

    def run(self):
        self.update_plot()
        print "########################################"
        print "Click on plot to edit strong lines list:"
        print "(left button - add lines, right - remove lines)."
        print "Then hit enter to redo the plot"
        print "Type 'w' on active plot window \
               \n to write the result to output file"
        print "q - to quit"
        print "#######################################"
        plt.show()


######################################################
######################################################
#Read config file
config = ConfigParser.ConfigParser()
cfg_file = sys.argv[1]
config.read(cfg_file)

file_list = np.genfromtxt(config.get('Input files','files_list'),dtype=None)
line_list_file = config.get('Input files','line_list_file')

lines=np.loadtxt(line_list_file,usecols=[0,1,2,3])
#Format: line element extitation_potential loggf, 
#all in MOOG like format, especially element

off = config.getfloat('Spectrum','off')
s_factor = config.getfloat('Spectrum','s_factor')
rejt_auto = config.getboolean('Spectrum','rejt_auto')
rejt = config.getfloat('Spectrum','rejt')
#if rejt_auto is True, rejt from config file will be ignored

r_lvl = config.getfloat('Lines','r_lvl')
w_factor = config.getfloat('Lines','w_factor')
l_eqw = config.getfloat('Lines','l_eqw')
h_eqw = config.getfloat('Lines','h_eqw')
v_lvl = config.getfloat('Lines','v_lvl')
det_level = config.getfloat('Lines','det_level')
plot_flag = config.getboolean('Lines','plot_flag')
show_lines=np.array(config.get('Lines','show_lines').split(" "),dtype=float)

#When there is only 1 element in spectra list
if file_list.size==1:
    file_list=np.append(file_list,["end_iter"])


for file_name in file_list:
    if file_name=="end_iter":
        exit()
    #Create output files
    logfile,out_file = create_out_files(file_name,line_list_file) 
    
    mgtab = np.empty((0,5),dtype=float)    
    
    #Here calculations start    
    file=np.loadtxt(file_name)

###################################################
    #Deal with rejt parameter    
    if rejt_auto == True:
        rejt,SN = auto_rejt(file,config)
        print_and_log(logfile,["rejt parameter was found automatically"])
    else:
       print_and_log(logfile,["rejt parameter defined by user"])
       SN = 1.0/(1.0-rejt)
    print_and_log(logfile,["rejt parameter:", rejt])
    print_and_log(logfile,["signal to noise:", SN])
    
#####################################################
    #Check where spectrum starts and ends
    #Check if your line list fits to this range
    #If not, crop the linelist
    lines_in_spec = lines[np.where(( lines[:,0]<file[:,0].max()-off) & (lines[:,0]>file[:,0].min()+off))]

#####################################################
#Lets analyze every single line
    for a_line in lines_in_spec:
        line,elem_id,exc_p,loggf=a_line
        print_and_log(logfile,["\n#####\n",line,elem_id])

        d=file[np.where((file[:,0]>line-off) &(file[:,0]<line+off))]
        print d.shape
        if d.shape[0]==0 or d[:,0].max()<line or d[:,0].min()>line:
            print_and_log(logfile,[ "Nothing to do in this range, \
                             probably gap in your spectra"])
            continue
        
        #Make spectrum linear
        #(default assumption - spectrum is not linear)
        lin1,dx=do_linear(d)

        
        #Correct continuum around chosen line
        try:
             lin,iter,p_no=correct_continuum(lin1,rejt)
        except:
             print_and_log(logfile,["Unable to correct continuum"])
             continue

        x=lin[:,0]
        f=lin[:,1]
        
        #Find derivatives
        gf,ggf,gggf=find_derivatives(x,f,dx,s_factor)
        
        #Find inflection points 
        gggf_infm,gggf_infp=find_inflection(x,gggf)
        
        #If there is no inflection points, go to next line on list
        if (gggf_infm.size==0 or gggf_infp.size==0):
            continue        
        
        #Identify strong lines automatically
        strong_lines,thold=find_strong_lines(x,gggf,gggf_infm,r_lvl,SN)
        strong_lines=evaluate_lines(line,strong_lines,det_level,gggf_infm)
        
        if len(strong_lines)==0:
            continue
                
        print_and_log(logfile,[ "I see", len(strong_lines),
                      "line(s) in this range"])

################################################################################
        if line in show_lines and plot_flag==False:
            plot_line=True
        elif line not in show_lines and plot_flag==False:
            plot_line=False
        else:
            plot_line=True
        
        if not plot_line:
            r_tab,lr = EW_analysis(line,x,f,strong_lines,v_lvl,
                                l_eqw,h_eqw,det_level)
            moog_output(out_file,a_line,lr)
        
        #Ploting module - 
        else:      
            fig,ax = plt.subplots(2,sharex=True)
            p = Plot_Module(ax,line,x,f,strong_lines,
                            v_lvl,l_eqw,h_eqw,det_level,thold)
            p.run()

