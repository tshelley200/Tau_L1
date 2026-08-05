'''
Check the distributions of input candidates between two versions
'''

from imports import *

#Set plotting style
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)
mpl.rcParams['lines.linewidth'] = 5

def read_inputs(data_path, sig = False):
    
    data = uproot.open(data_path)
    
    inputs = data['ntuplePupSingle']['tree']['m_inputs'].array()
    extra_inputs = data['ntuplePupSingle']['tree']['extra_inputs'].array()
    ecal_shape = data['ntuplePupSingle']['tree']['ecal_shape'].array()
    hcal_shape = data['ntuplePupSingle']['tree']['hcal_shape'].array()

    reco_pt = data['ntuplePupSingle']['tree']['pt'].array()
    reco_eta = data['ntuplePupSingle']['tree']['eta'].array()
    deltaR = data['ntuplePupSingle']['tree']['gendr1'].array()
    eta = data['ntuplePupSingle']['tree']['geneta1'].array()

    if(sig): selection = (reco_pt > 0.) & (abs(deltaR) < 0.4) & (abs(eta) < 2.4) & (abs(reco_eta) < 2.4)
    else: selection = (reco_pt > 0.) & (abs(reco_eta < 2.4))
    
    return inputs[selection], extra_inputs[selection], ecal_shape[selection], hcal_shape[selection], reco_pt[selection], reco_eta[selection]
    

def main():
    
    data_path1 = '../data/merged/bkg.root'
    data_path2 = '../data/merged/sig.root'

    label1 = 'Bkg'
    label2= 'Sig'
    
    inputs1, extra1, ecal1, hcal1, pt1, eta1  = read_inputs(data_path1, sig = False)
    inputs2 , extra2, ecal2, hcal2, pt2, eta2 = read_inputs(data_path2, sig = True)
    print("bkg", len(inputs1))
    print("sig", len(inputs2))


    H, bins, _ = plt.hist(pt1, density=True, label = label1, alpha = 0.5)
    H2, bins, _ = plt.hist(pt2, bins=bins, density=True, label = label2, alpha=0.5)
    
    plt.xlabel(r"Tau Candidate pT")
    plt.ylabel(r"a.u.")
    
    plt.legend()  
    plt.show()
    plt.savefig('plots/pt_compare.png', bbox_inches='tight')
    plt.close()


    H, bins, _ = plt.hist(eta1, density=True, label = label1, alpha = 0.5)
    H2, bins, _ = plt.hist(eta2, bins=bins, density=True, label = label2, alpha=0.5)
    
    plt.xlabel(r"Tau Candidate Eta")
    plt.ylabel(r"a.u.")
    
    plt.legend()  
    plt.show()
    plt.savefig('plots/eta_compare.png', bbox_inches='tight')
    plt.close()

    
    print(inputs1[:10,0])
    print(inputs2[:10,0])
    H, bins, _ = plt.hist(inputs1[:,0][inputs1[:,0] != 0], density=True, label = label1, alpha = 0.5)
    H2, bins, _ = plt.hist(inputs2[:,0][inputs2[:,0] != 0], bins=bins, density=True, label = label2, alpha=0.5)
    
    plt.xlabel(r"Leading candidate $p_T^{reco}$ [GeV]")
    plt.ylabel(r"a.u.")
    
    plt.legend()  
    plt.show()
    plt.savefig('plots/leading_pt_compare.png', bbox_inches='tight')
    plt.close()
    
    #Second candidate
    H, bins, _ = plt.hist(inputs1[:,8], density=True, label = label1, alpha = 0.5)
    H2, bins, _ = plt.hist(inputs2[:,8], bins=bins, density=True, label = label2, alpha=0.5)
    
    plt.xlabel(r"Second candidate $p_T^{reco}$ [GeV]")
    plt.ylabel(r"a.u.")
    
    plt.legend()  
    plt.show()
    plt.savefig('plots/second_pt_compare.png', bbox_inches='tight')
    plt.close()
    
    #Plot eta
    print(inputs1[:,1])
    H, bins, _ = plt.hist(inputs1[:,33][inputs1[:,33] != 0],bins=50, density=True, label = label1, alpha = 0.5, range=(-0.5,0.5))
    H2, bins, _ = plt.hist(inputs2[:,33][inputs2[:,33] != 0], bins = bins, density=True, label = label2, alpha=0.5)
    
    plt.xlabel(r"4th candidate $\eta^{reco}$ [GeV]")
    plt.ylabel(r"a.u.")
    
    plt.legend()  
    plt.show()
    plt.savefig('plots/leading_eta_compare.png', bbox_inches='tight')
    plt.close()


    #Plot extra info
    print(extra1.type)
    H, bins, _ = plt.hist(extra1[:,0,0],bins=50, density=True, label = label1, alpha = 0.5, range=(-0.5,0.5))
    H2, bins, _ = plt.hist(extra2[:,0,0],bins=bins, density=True, label = label2, alpha = 0.5, range=(-0.5,0.5))
    
    plt.xlabel(r"Extra 1")
    plt.ylabel(r"a.u.")
    
    plt.legend()  
    plt.show()
    plt.savefig('plots/extra1_compare.png', bbox_inches='tight')
    plt.close()

    H, bins, _ = plt.hist(extra1[:,0,1],bins=50, density=True, label = label1, alpha = 0.5, range=(-0.5,0.5))
    H2, bins, _ = plt.hist(extra2[:,0,1],bins=bins, density=True, label = label2, alpha = 0.5, range=(-0.5,0.5))
    
    plt.xlabel(r"Extra 2")
    plt.ylabel(r"a.u.")
    
    plt.legend()  
    plt.show()
    plt.savefig('plots/extra2_compare.png', bbox_inches='tight')
    plt.close()

    H, bins, _ = plt.hist(extra1[:,1,1],bins=50, density=True, label = label1, alpha = 0.5, range=(-0.5,0.5))
    H2, bins, _ = plt.hist(extra2[:,1,1],bins=bins, density=True, label = label2, alpha = 0.5, range=(-0.5,0.5))
    
    plt.xlabel(r"Extra 3")
    plt.ylabel(r"a.u.")
    
    plt.legend()  
    plt.show()
    plt.savefig('plots/extra3_compare.png', bbox_inches='tight')
    plt.close()
    

main()
