import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import ndimage
import scipy.optimize as so
from scipy.signal import argrelextrema
from scipy.special import erf
from scipy.special import wofz
import ConfigParser
import sys

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
    while stop==False:
        p_no_prev=p_no
        ft=np.polyfit(tab[:,0],tab[:,1],2)
        tab= spec[np.where(spec[:,1]>rejt*np.polyval(ft,spec[:,0]))]
        
        p_no=tab.shape[0]
    
        if (p_no==p_no_prev):
            stop=True
    if p_no!=0:
        cont=spec[:,1]/np.polyval(ft,spec[:,0])
        spec[:,1]=cont
    else:
        print "Fail when correcting continuum"
        print "Let's stick to what we already have"
        #This is for cases where line is located 
        #i.e. in a break of echelle spectrum    
    return spec

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
    SN_tab=[]
    for item in rejt_reg:
        r_low,r_up=item
        reg,rres=do_linear(file[np.where((tab[:,0]>r_low) &(tab[:,0]<r_up))])
        reg_smo= ndimage.gaussian_filter1d(reg[:,1],sigma=s_factor,mode='wrap')
        rejt_tab.append(np.std(reg[:,1]-reg_smo))
        SN_tab.append(np.average(reg[:,1])/np.std(reg[:,1]-reg_smo))
    rejt=1.-np.median(rejt_tab)
    SN=np.median(SN_tab)
    
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
    
    noise=np.std(tab)*r_lvl
    str_lines=[]
    if not (max_tab.size!=0 and min_tab.size!=0 and noise==0.0):
        for item in  xo:
            indx= np.abs(max_tab[0,:]-item).argmin()
            indm= np.abs(min_tab[0,:]-item).argmin()
        
            if ((np.abs(max_tab[1,indx])>noise) and (np.abs(min_tab[1,indm])>noise)):
                str_lines.append(item)
            
    return str_lines,noise

def evaluate_lines(line,strong_lines,det_level,gggf_infm):
    #check if our line was identified
    if (len(strong_lines)==0 and  (np.abs(gggf_infm-line).min()>det_level)):
        print line, "was not detected"
    elif (len(strong_lines)==0 and (np.abs(gggf_infm-line).min()<det_level)):#no strong lines detected
        print "I see no strong lines here,but weak line close to", line,"was detected"
        strong_lines=np.append(strong_lines,line) 
    elif (len(strong_lines)>0 and np.abs(gggf_infm-line).min()<det_level and np.abs(strong_lines-line).min()<det_level):
        print "line",line," was detected and it was classified as a strong line"
        pass
    elif (len(strong_lines)>0 and (np.abs(gggf_infm-line).min()<det_level) and np.abs(strong_lines-line).min()>det_level):
        print "I see this line at",line,", but it is weak"
        strong_lines=np.append(strong_lines,line)
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
def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def multiple_gaus(x,params):
    mg=np.zeros_like(x)
    for row in params:
        x0,a,sig=row
        mg=mg+gaus(x,a,x0,sig)
    return mg

def res_g(p,data):
    x,y=data
    a,x0,sig=p
    sg=gaus(x,a,x0,sig)
    err=1./np.abs(y)
    return (y-sg)/err
    
def res_mg(p,x,y,nb):
    params=np.reshape(p,(nb,3))
    mg=multiple_gaus(x,params)
    err=1./np.abs(y)
    return (y-mg)/err

def get_gew(ag,sg):
    #calculate EW for gaussian profile
    gew=ag*np.sqrt(2*np.pi)*np.abs(sg)*1000.
    
    return gew
    
def fit_single_Gauss(x,f,a1,x01,s1):
    gaus_p = so.leastsq(res_g,[a1,x01,s1],
             args=([x,1.0-f]),
             full_output=1)
    a1s,x01s,s1s=gaus_p[0]

    eqw_gf=get_gew(a1s,s1s)

    gf_errs=leastsq_errors(gaus_p,3)
    eqw_gf_err= eqw_gf*(gf_errs[0,1]/a1s+gf_errs[0,2]/np.abs(s1s))

    return eqw_gf,eqw_gf_err,gaus_p[0]

def fit_multi_Gauss(x,f,strong_lines):
    params=np.ones((len(strong_lines),3))
    params[:,0]=strong_lines #first columns = x0
    params[:,2]=0.05 #starting value for gauss fit
        
    new_params=np.array([])

    #First run
    while params.shape[0]!=new_params.shape[0] and params.shape[0]>0.:
        plsq=so.leastsq(res_mg,params,
             args=(x,1.0-f,params.shape[0]),full_output=1)
        new_params= np.reshape(plsq[0],(params.shape[0],3))
        #evaluate  run
        ind=np.where(np.abs(strong_lines-new_params[:,0])<det_level)
        strong_lines=np.array(strong_lines)[ind]
        params=new_params[ind]
        
    mg_errs=leastsq_errors(plsq,3)
    
    return params,mg_errs

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
                               
    alphaD,alphaL, nu_0, A_voigt=voigt_p[0]

    I=voigt_p[0][3]*1000.
    v_errs=leastsq_errors(voigt_p,4)[0][3]
    
    return I,v_errs,voigt_p[0]
    

######################################################
#Other
######################################################
def pm_3sig(x,x01,s1):
    iu=np.abs(x-x01-3.0*np.abs(s1)).argmin()
    il=np.abs(x-x01+3.0*np.abs(s1)).argmin()
    
    if iu==il or np.abs(iu-il)<10.:
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
        params=np.array([[9.9,9.9,9.9]])

    mgaus=multiple_gaus(x,params)
    
    ip= np.abs(params[:,0]-line).argmin()
    x01,a1,s1= params[ip,:]    
    
    eqw=a1*np.sqrt(2*np.pi)*np.abs(s1)*1000.
    eqw_err= eqw*(mg_errs[ip,1]/a1+mg_errs[ip,2]/np.abs(s1))
    
    #Calculate single gauss profile 
    #(this gauss is a part of multigaussian fit)
    sgaus=gaus(x,a1,x01,s1)
    sparams=[a1,x01,s1]    

    #Determine region close to gaussian line center#
    #plus minus 3 sigma
    il, iu = pm_3sig(x,x01,s1)
    
    oc_mg=np.average(np.abs(f[il:iu]-1.0+mgaus[il:iu]))
    oc_sg=np.average(np.abs(f[il:iu]-1.0+sgaus[il:iu]))
    
    results=append_to_dict(results,'mg',mgaus,params,eqw,eqw_err,oc_mg)
    results=append_to_dict(results,'sg',sgaus,sparams,eqw,eqw_err,oc_sg)                                                         

    #Fit single Gauss and Voigt profile
    vparams=np.zeros((len(strong_lines),4))
    eqw_gf,eqw_gf_err,gparams=fit_single_Gauss(x[il:iu],f[il:iu],a1,x01,s1)
    gausf=gaus(x,gparams[0],gparams[1],gparams[2])    
    oc_g=np.average(np.abs(f[il:iu]-1.0+gausf[il:iu]))
    results=append_to_dict(results,'g',gausf,gparams,eqw_gf,eqw_gf_err,oc_g)       
    
    I, v_errs,vpar=fit_Voigt(x[il:iu],f[il:iu],x01)
    svoigt=Voigt(x,vpar[0],vpar[1], vpar[2], vpar[3],0.,0.)    
    oc_v=np.average(np.abs(f[il:iu]-1.0+svoigt[il:iu]))
    results=append_to_dict(results,'v',svoigt,vpar,I,v_errs,oc_v)       

    return results

def print_line_info(rslt):
    fit_labels={'mg':'multi Gauss','sg':'part of mGauss','g':'Gauss','v':'Voigt'}
    for fit in rslt:
        print "%15s %s %4.2f %s %f %s %f" % \
        (fit_labels[fit],": EW =" ,rslt[fit][2],"eEW =",rslt[fit][3],"o-c:",rslt[fit][4])

def print_mgauss_data(rslt):
    #print full data for all lines 
    #fitted with multi Gauss function
    mg_params=rslt['mg'][1]
    print "\n",mg_params.shape[0],"lines in multi gaussian fit:"
    for gfit in  mg_params:
        print "%4.2f %s%4.2f %s%4.3f %s%4.2f" % \
              ( gfit[0], "depth=", gfit[1], "sigma:",gfit[2],"EW=",get_gew(gfit[1],gfit[2]))
    print "\n"
    
    
def evaluate_results(line,rslt,v_lvl,l_eqw,h_eqw,det_level):
    print_line_info(rslt)
    if rslt['mg'][3]>0.5*rslt['mg'][2]:
        print "Huge error!", rslt['mg'][2],rslt['mg'][3]
        hu=True
    else:
        hu=False
            
    if rslt['v'][4]<rslt['mg'][4] and rslt['v'][4]<rslt['g'][4] and np.log10(rslt['mg'][2]*0.001/line)>v_lvl:
        print "using Voigt profile"
        eqw = rslt['v'][2]
        eqw_err = rslt['v'][3]
    elif rslt['g'][4]<rslt['mg'][4] and np.log10(rslt['mg'][2]*0.001/line)<v_lvl:
        print "using single Gauss fit"
        eqw=rslt['g'][2]
        eqw_err=rslt['g'][3]
    elif hu==True and rslt['g'][4]<rslt['sg'][4]:
        print "using single Gauss fit"
        eqw=rslt['g'][2]
        eqw_err=rslt['g'][3]
    else:
        eqw=rslt['mg'][2]
        eqw_err=rslt['mg'][3]
        
    if (eqw>h_eqw or eqw<l_eqw):
        print eqw
        print "Line is to strong or to weak"
        eqw = -99.9
        eqw_err = 99.9
    
    x01=rslt['sg'][1][1]
    if np.abs(line-x01)>det_level:
        print line,elem_id, "line outside the det_level range"
        eqw = -99.9
        eqw_err = 99.9
            
    eqw=round(eqw,2)
    return eqw,eqw_err

######################################################
#Ploting functions
######################################################        
        
def ontype(event):
    if event.key=='enter':
        print "\n", len(list(set(strong_lines))),"lines to fit"
        
        r_tab = find_eqws(line,x,f,sorted(list(set(strong_lines))))
        print_mgauss_data(r_tab)

        eqw,eqw_err=evaluate_results(line,r_tab,v_lvl,l_eqw,h_eqw,det_level)
        moog= "%10s%10s%10s%10s%10s%10s%10s %s \n" % \
                (line,elem_id,exc_p,loggf,'','',eqw,eqw_err)
        print "\n MOOG entry:"
        print moog
        
        fsl= r_tab['mg'][1][:,0] #fitted strong lines
        ndl_ind=[] #not detected lines index
        for sline in strong_lines:
           if not any(np.abs(sline-fsl)<det_level):
               print "Line at", round(sline,2), "was not included in mGauss\n"
               ndl_ind.append(sline)
        for item in ndl_ind:
            strong_lines.remove(item)

        #Remove old fits before ploting new ones
        plt.sca(ax1)
        for artist in plt.gca().get_children():
            if hasattr(artist,'get_label') and (artist.get_label() in lstyle[:,3] or 
                artist.get_label()=="line" or artist.get_label()=='pm3s' or
                artist.get_label()=='strong lines' or  artist.get_label()=='line_pnt'):
                artist.remove()        
        plt.sca(ax2) 
        for artist in plt.gca().get_children():
            if hasattr(artist,'get_label') and (artist.get_label()=='strong lines' 
                or  artist.get_label()=='line_pnt'):
                artist.remove()                
        
        #Plot new fits
        for lbl in r_tab:
            fit_style=np.squeeze(lstyle[np.where(lstyle[:,0]==lbl)])
            ax1.plot(x,1.0-r_tab[lbl][0],
                   color = fit_style[1],
                   ls =    fit_style[2],
                   label = fit_style[3])            
                   
        x01=r_tab['sg'][1][1]
        s1=r_tab['sg'][1][2]
        ax1.axvspan(x01-3.0*s1,x01+3.0*s1,color='g',alpha=0.25,label="pm3s")
        ax1.legend(loc=2,numpoints=1,fontsize='10',ncol=5)
        ax1.axvline(x01,color='r',lw=1.5,label="line")
        for oline in r_tab['mg'][1][:,0]:
            ax1.axvline(oline,c='r',zorder=1,label='strong lines')
        
    

        ax2.plot(strong_lines,np.zeros_like(strong_lines),'o',color='r',label="strong lines")
        ax2.legend(loc=3,numpoints=1,fontsize='10',ncol=3)        
        print "Click on plot to edit line list:"
        print "(left button - add lines, right - remove lines)."
        print "Then hit enter to redo the plot"
        print "Type 'w' to write the result to output file"
        print "q - quit"

    elif event.key=='q':
        exit()
    elif event.key=='w':
        r_tab = find_eqws(line,x,f,strong_lines)
        eqw,eqw_err=evaluate_results(line,r_tab,v_lvl,l_eqw,h_eqw,det_level)
        moog= "%10s%10s%10s%10s%10s%10s%10s %s \n" % \
             (line,elem_id,exc_p,loggf,'','',eqw,eqw_err)
        print "Writing to output file..."
        print moog
        out_file.write(moog)
        print "Close the plot window to move on"
        
    else:
        #if you accidentally hit any other key
        pass
    plt.draw()

def onclick(event):
    # when none of the toolbar buttons is activated and the user clicks in the
    # plot somewhere,
    toolbar = plt.get_current_fig_manager().toolbar
    if event.button==1 and toolbar.mode=='':
        ind= np.abs((gggf_infm-event.xdata)).argmin()
        ax2.plot(gggf_infm[ind],1.0,'o',color='r',mec='b',picker=5,label='line_pnt')
        ax1.axvline(gggf_infm[ind],c='r',ls=":",zorder=1,label='line_pnt')
        strong_lines.append(gggf_infm[ind])
    elif event.button==3 and toolbar.mode=='':
        ind= np.squeeze(np.where(np.abs(strong_lines-event.xdata)<0.01))
        try:
            ax2.plot(strong_lines[ind],1.0,'o',color='b',mec='r',picker=5,label='line_pnt')
            ax1.axvline(strong_lines[ind],c='b',ls=":",zorder=1,label='line_pnt')
            strong_lines.pop(ind)
        except TypeError:
            print "I see more than one line close to your point"
            print "I can't deal with that"
    plt.draw()   


######################################################
######################################################
#Read config file
config = ConfigParser.ConfigParser()
cfg_file = sys.argv[1]
config.read(cfg_file)

file_list = open(config.get('Input files','files_list')).readlines()
line_list_file = config.get('Input files','line_list_file')
lines=np.loadtxt(line_list_file) #Format: line element extitation_potential loggf, 
                                 #all in MOOG like format, especially element

off = config.getfloat('Spectrum','off')
s_factor = config.getfloat('Spectrum','s_factor')
rejt_auto = config.getboolean('Spectrum','rejt_auto')
rejt = config.getfloat('Spectrum','rejt')
#if rejt_auto is True, rejt from config file will be ignored

r_lvl = config.getfloat('Lines','r_lvl')
l_eqw = config.getfloat('Lines','l_eqw')
h_eqw = config.getfloat('Lines','h_eqw')
v_lvl = config.getfloat('Lines','v_lvl')
det_level = config.getfloat('Lines','det_level')
plot_flag = config.getboolean('Lines','plot_flag')

show_lines=np.array(config.get('Lines','show_lines').split(" "),dtype=float)

#if plot_flag == True or show_lines[0]!=0:


for file_name in file_list:
    file_name=file_name.strip()
    file_name_out=file_name.split("/")[-1]
    print line_list_file
    list_name=line_list_file.split('.')[0]
    out_file=open("moog_"+list_name+"_"+file_name_out,'wb')
    out_file.write(file_name_out+"\n")

    #Here calculations start    
    file=np.loadtxt(file_name)

###################################################
    #Deal with rejt parameter    
    if rejt_auto == True:
        rejt,SN = auto_rejt(file,config)
    else:
       print "I will use rejt defined by user"
       SN = 1.0/(1.0-rejt)
    print "signal to noise:", SN
    print "rejt parameter:", rejt
#####################################################
    #Check where spectrum starts and ends
    #Check if your line list fits to this range
    #If not, crop the linelist
    lines = lines[np.where(( lines[:,0]<file[:,0].max()-off) & (lines[:,0]>file[:,0].min()+off))]

#####################################################
#Lets analyze every single line
    for a_line in lines:
        line,elem_id,exc_p,loggf=a_line

        d=file[np.where((file[:,0]>line-off) &(file[:,0]<line+off))]
        if d.shape[0]==0:
            print "Nothing to do in this range, probably gap in you spectra"
            continue
        
        #Make spectrum linear
        #(default assumption - spectrum is not linear)
        lin1,dx=do_linear(d)
        
        #Correct continuum around chosen line
        lin=correct_continuum(lin1,rejt)

        x=lin[:,0]
        f=lin[:,1]
        
        #Find derivatives
        gf,ggf,gggf=find_derivatives(x,f,dx,s_factor)
        
        #Find inflection points 
        gggf_infm,gggf_infp=find_inflection(x,gggf)
        gf_infm,gf_infp=find_inflection(x,gf)
        
        #If there is no inflection points, go to next line on list
        if (gggf_infm.size==0 or gggf_infp.size==0):
            continue        
        
        #Identify strong lines automatically
        strong_lines,noise=find_strong_lines(x,gggf,gggf_infm,r_lvl,SN)
        strong_lines=evaluate_lines(line,strong_lines,det_level,gggf_infm)
        
        if len(strong_lines)==0:
            continue
                
        print "I see", len(strong_lines),"line(s) in this range"
########################################################################        
        if line in show_lines and plot_flag==False:
            plot_line=True
        elif line not in show_lines and plot_flag==False:
            plot_line=False
        else:
            plot_line=True

        if not plot_line:
        #do all fits: multi gauss, sgauss (part of multi gauss),
        # gauss fitted in small area, voigt fitted in small area
            r_tab = find_eqws(line,x,f,strong_lines) #results tab
        #Check if EW is reasonable
            eqw,eqw_err=evaluate_results(line,r_tab,v_lvl,l_eqw,h_eqw,det_level)
        #Write to output file
            moog= "%10s%10s%10s%10s%10s%10s%10s %s \n" % \
               (line,elem_id,exc_p,loggf,'','',eqw,eqw_err)        
            out_file.write(moog)
            print moog+"\n"            
        
        #Ploting module - 
        else:
            fig = plt.figure()
            interactive_mode=True
            #Things that won't be changed
            lstyle=np.array([['mg','b','-','multi Gauss'],
                    ['sg','b',':','line in mGauss'],
                    [ 'g', 'y','-','Gauss'],
                    [ 'v', 'm','-','Voigt']])
            
            x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
            ax1=fig.add_subplot(2,1,1)
            ax1.xaxis.set_major_formatter(x_formatter)
            ax1.plot(x,f,'o',color= 'k' , label='spectrum')
            ax1.axhline(1.0,color='g',label='continuum')
            ax1.set_xlabel("Wavelenght")
            ax1.legend(loc=2,numpoints=1,fontsize='10') 
            ax1.set_ylim(min(f)-0.1,1.4)       
            ax1.set_title(str(elem_id)+" "+str(line))           
           
            ax2=fig.add_subplot(2,1,2,sharex=ax1)
            ax2.plot(x,np.zeros_like(x))
            ax2.plot(x,gggf,'c', label='3rd derivative')
            ax2.axhline(noise,c='r')
            ax2.axhline(-noise,c='r')
            ax2.plot(gggf_infm, np.zeros_like(gggf_infm),'o',color='b',label='flex points + -> -')
            ax2.set_ylim(np.min(gggf),np.max(gggf))
            ax2.set_xlim(line-off,line+off)
            ax2.legend(loc=3,numpoints=1,fontsize='10',ncol=3)                            
            print "Press enter to make a fit"
            plt.gcf().canvas.mpl_connect('key_press_event',ontype)
            plt.gcf().canvas.mpl_connect('button_press_event',onclick)
                               
            plt.show()           
            print "############################\n"
