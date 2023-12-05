'''
RM
problems as of 10.01.2022
1. in some cases, the (ds, mu) pair with mass < 8 GeV is not the one that is force decayed FIXED
2. charge conjugation missing, ds always positive, muon always negative DONE
3. missing is signal flag DONE
4. missing treatment of Ds* DONE
5. change lumiblock at each file DONE
6. check tau mother clean from photons and self copies DONE!
7. for fakes, use the MC and use muon fake rates from Bmm NOT NEEDED
8. multithrerad DONE!
9. add vertex info   DONE !
10. add cosine   DONE!
11. try reco (w/ PU?)  DONE!
12. add trigger Mu7_IP4 DONE
13. Add IP3D significance?
14. add other mass hypotheses
15. GEN matching CAN BE DONE A POSTERIORI
18. produce exclusive Ds tau sample DONE
19. produce data DONE


BUG FIXES
- Bs mass
- save kaon p4 under the kaon mass hypothesis (energy is wrong)

- trigger now is broken


'''
from __future__ import print_function
import ROOT
import re
import argparse
import numpy as np
import pickle
from time import time
from datetime import datetime, timedelta
from array import array
from glob import glob
from collections import OrderedDict
from scipy.constants import c as speed_of_light
from scipy import stats
from DataFormats.FWLite import Events, Handle
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from itertools import product, combinations
# https://pypi.org/project/particle/
import particle
from particle import Particle
ROOT.gSystem.Load('libVtxFitFitter')
from ROOT import KVFitter # VertexDistance3D is contained here, dirt trick!!

vtxfit = KVFitter()
tofit = ROOT.std.vector('reco::Track')()

# ipython -i -- inspector_rds.py --inputFiles="/pnfs/psi.ch/cms/trivcat/store/user/manzoni/inclusive_HbToDsPhiKKPiMuNu_MT_GEN_10jan21_v8/*.root" --filename="inclusive_HbToDsPhiKKPiMuNu_MT_GEN_10jan21_v8"
# ipython -i -- inspector_rds.py --inputFiles="/pnfs/psi.ch/cms/trivcat/store/user/manzoni/inclusive_HbToDsPhiKKPiMuNu_GEN_10jan21_v6/*.root" --filename="inclusive_HbToDsPhiKKPiMuNu_GEN_10jan21_v6"



# ipython -i -- inspector_rds_analysis.py --inputFiles="33A82C95-B339-1342-A78D-8F755670C22F.root" --filename="data_test"



parser = argparse.ArgumentParser(description='')
parser.add_argument('--inputFiles'   , dest='inputFiles' , required=True, type=str)
parser.add_argument('--verbose'      , dest='verbose'    , action='store_true' )
parser.add_argument('--destination'  , dest='destination', default='./' , type=str)
parser.add_argument('--filename'     , dest='filename'   , required=True, type=str)
parser.add_argument('--maxevents'    , dest='maxevents'  , default=-1   , type=int)
parser.add_argument('--miniaod'      , dest='is_miniaod' , action='store_true')
parser.add_argument('--mc'           , dest='mc'         , action='store_true')
parser.add_argument('--logfreq'      , dest='logfreq'    , default=100   , type=int)
args = parser.parse_args()

inputFiles    = args.inputFiles
destination   = args.destination
fileName      = args.filename
maxevents     = args.maxevents
is_miniaod    = args.is_miniaod
verbose       = args.verbose
logfreq       = args.logfreq
mc = False; mc = args.mc

diquarks = [
    1103,
    2101,
    2103,
    2203,
    3101,
    3103,
    3201,
    3203,
    3303,
    4101,
    4103,
    4201,
    4203,
    4301,
    4303,
    4403,
    5101,
    5103,
    5201,
    5203,
    5301,
    5303,
    5401,
    5403,
    5503,
]

excitedBs = [
    513,
    523,
    533,
    543,
    # others?
]

def fillRecoTree(ntuple_reco, tofill_reco, mc, which_signal=np.nan):
    if mc: tofill_reco['sig'] = which_signal
    ntuple_reco.Fill(array('f', tofill_reco.values()))
    
def isAncestor(a, p):
    if a == p :
        return True
    for i in xrange(0,p.numberOfMothers()):
        if isAncestor(a,p.mother(i)):
            return True
    return False

def isMyDs(ds, minpt=0.5, maxeta=2.5):
    daus = []
    for idau in range(ds.numberOfDaughters()):
        dau = ds.daughter(idau)
        if dau.pdgId()==22: 
            continue # exclude FSR
        if abs(dau.pdgId())==211:
            if dau.pt()<minpt or abs(dau.eta())>maxeta:
                continue # only pions in the acceptance
            ds.pion = dau
        if abs(dau.pdgId())==333:
            if dau.numberOfDaughters()!=2: 
                continue
            for jdau in range(dau.numberOfDaughters()):
                if abs(dau.daughter(jdau).pdgId())!=321 or \
                   dau.daughter(jdau).pt < minpt        or \
                   abs(dau.daughter(jdau).eta()) > maxeta:
                    continue # only kaons in the acceptance
            ds.phi_meson = dau
        daus.append(dau.pdgId())
    daus.sort(key = lambda x : abs(x))
    return daus==[211, 333] or daus==[-211, 333]

class candidate():
    def __init__(self, ds, muon):
        self.ds = ds
        self.muon = muon
    def p4(self):
        return self.ds.p4() + self.muon.p4()
    def charge(self):
        return self.ds.charge() + self.muon.charge()

def printAncestors(particle, ancestors=[], verbose=True):
    for i in xrange(0, particle.numberOfMothers()):
        mum = particle.mother(i)
#         if mum is None: import pdb ; pdb.set_trace()
        if abs(mum.pdgId())<8 or \
           abs(mum.pdgId())==21 or \
           abs(mum.pdgId()) in diquarks or\
           abs(mum.pdgId()) in excitedBs or\
           abs(mum.eta()) > 1000: # beam protons
            continue
        # don't count B oscillations
        if mum.pdgId() == -particle.pdgId() and abs(particle.pdgId()) in [511, 531]:
            continue 
        if not mum.isLastCopy(): continue
        try:
            if verbose: print(' <-- ', Particle.from_pdgid(mum.pdgId()).name, end = '')
            ancestors.append(mum)
            printAncestors(mum, ancestors=ancestors, verbose=verbose)
        except:
            if verbose: print(' <-- ', 'pdgid', mum.pdgId(), end = '')
            ancestors.append(mum)
            printAncestors(mum, ancestors=ancestors, verbose=verbose)
        else:
            pass
    particle.ancestors = ancestors

handles_mc = OrderedDict()
handles_mc['genpr'  ] = ('prunedGenParticles', Handle('std::vector<reco::GenParticle>')     )
handles_mc['genpk'  ] = ('packedGenParticles', Handle('std::vector<pat::PackedGenParticle>'))
handles_mc['genInfo'] = ('generator'         , Handle('GenEventInfoProduct')                )

handles = OrderedDict()
handles['muons'  ] = ('slimmedMuons'                 , Handle('std::vector<pat::Muon>')              )
handles['trk'    ] = ('packedPFCandidates'           , Handle('std::vector<pat::PackedCandidate>')   )
handles['ltrk'   ] = ('lostTracks'                   , Handle('std::vector<pat::PackedCandidate>')   )
handles['vtx'    ] = ('offlineSlimmedPrimaryVertices', Handle('std::vector<reco::Vertex>')           )
handles['trg_res'] = (('TriggerResults', '', 'HLT' ) , Handle('edm::TriggerResults'        )         )
handles['trg_ps' ] = (('patTrigger'    , '')         , Handle('pat::PackedTriggerPrescales')         )
handles['bs'     ] = ('offlineBeamSpot'              , Handle('reco::BeamSpot')                      )

# inputFiles = '/pnfs/psi.ch/cms/trivcat/store/user/manzoni/BsToDsPhiKKPiMuNu_GEN_7jun21_v*/*.root'

if ('txt' in inputFiles):
    with open(inputFiles) as f:
        files = f.read().splitlines()
elif ',' in inputFiles:
    files = inputFiles.split(',')
else:
    files = glob(inputFiles)

# files  = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/inclusive_HbToDsPhiKKPiMuNu_MINI_26jan21_v1/*.root')
# files += glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/inclusive_HbToDsPhiKKPiMuNu_MINI_25mar21_v1/*.root')

print("files:", files)

events = Events(files)
maxevents = maxevents if maxevents>=0 else events.size() # total number of events in the files

start = time()

alldecays = dict()

branches = [
    'run'          ,
    'lumi'         ,
    'event'        ,

    'n_cands'      ,

    'ds_m_mass'    ,
    'ds_m_pt'      ,
    'ds_m_eta'     ,
    'ds_m_phi'     ,
    'max_dr'       ,
    'm2_miss'      ,
    'q2'           ,
    'e_star_mu'    ,

    'mu_pt'        ,
    'mu_eta'       ,
    'mu_phi'       ,
    'mu_e'         ,
    'mu_mass'      ,
    'mu_charge'    ,

    'tau_pt'       ,
    'tau_eta'      ,
    'tau_phi'      ,
    'tau_e'        ,
    'tau_mass'     ,
    'tau_charge'   ,

    'hb_pt'        ,
    'hb_eta'       ,
    'hb_phi'       ,
    'hb_e'         ,
    'hb_mass'      ,
    'hb_charge'    ,
    'hb_pdgid'     ,

    'ds_pt'        ,
    'ds_eta'       ,
    'ds_phi'       ,
    'ds_e'         ,
    'ds_mass'      ,
    'ds_charge'    ,

    'ds_st_pt'     ,
    'ds_st_eta'    ,
    'ds_st_phi'    ,
    'ds_st_e'      ,
    'ds_st_mass'   ,
    'ds_st_charge' ,

    'phi_pt'       ,
    'phi_eta'      ,
    'phi_phi'      ,
    'phi_e'        ,
    'phi_mass'     ,
    'phi_charge'   ,

    'kp_pt'        ,
    'kp_eta'       ,
    'kp_phi'       ,
    'kp_e'         ,
    'kp_mass'      ,
    'kp_charge'    ,

    'km_pt'        ,
    'km_eta'       ,
    'km_phi'       ,
    'km_e'         ,
    'km_mass'      ,
    'km_charge'    ,

    'pi_pt'        ,
    'pi_eta'       ,
    'pi_phi'       ,
    'pi_e'         ,
    'pi_mass'      ,
    'pi_charge'    ,

    'pv_x'         ,
    'pv_y'         ,
    'pv_z'         ,

    'ds_st_vx'     ,
    'ds_st_vy'     ,
    'ds_st_vz'     ,

    'ds_vx'        ,
    'ds_vy'        ,
    'ds_vz'        ,

    'cos'          ,
    
    'same_b'       ,
    'sig'          ,
]

branches_reco = [
    'run'          ,
    'lumi'         ,
    'event'        ,
    'npv'          ,
    
    'n_cands'      ,
    
    'ds_m_mass'    ,
    'ds_m_pt'      ,
    'ds_m_eta'     ,
    'ds_m_phi'     ,
    'b_pt'         ,
    'b_eta'        ,
    'b_phi'        ,
    'b_beta'       ,
    'b_gamma'      ,
    'b_ct'         ,
    'm2_miss'      ,
    'q2'           ,
    'e_star_mu'    ,
    'pt_miss_sca'  ,
    'pt_miss_vec'  ,
    'ptvar'        ,

    'mu_pt'        ,
    'mu_eta'       ,
    'mu_phi'       ,
    'mu_e'         ,
    'mu_mass'      ,
    'mu_charge'    ,
    'mu_id_loose'  ,
    'mu_id_soft'   ,
    'mu_id_medium' ,
    'mu_id_tight'  ,
    'mu_ch_iso'    ,
    'mu_db_n_iso'  ,
    'mu_abs_iso'   ,
    'mu_rel_iso'   ,
#     'mu_iso'       ,
    'mu_dxy'       ,
    'mu_dxy_e'     ,
    'mu_dxy_sig'   ,
    'mu_dz'        ,
    'mu_dz_e'      ,
    'mu_dz_sig'    ,
    'mu_bs_dxy'    ,
    'mu_bs_dxy_e'  ,
    'mu_bs_dxy_sig',

    'ds_pt'        ,
    'ds_eta'       ,
    'ds_phi'       ,
    'ds_e'         ,
    'ds_mass'      ,

    'phi_pt'       ,
    'phi_eta'      ,
    'phi_phi'      ,
    'phi_e'        ,
    'phi_mass'     ,

    'kp_pt'        ,
    'kp_eta'       ,
    'kp_phi'       ,
    'kp_e'         ,
    'kp_mass'      ,
    'kp_charge'    ,
    'kp_dxy'       ,
    'kp_dxy_e'     ,
    'kp_dxy_sig'   ,
    'kp_dz'        ,
    'kp_dz_e'      ,
    'kp_dz_sig'    ,
    'kp_bs_dxy'    ,
    'kp_bs_dxy_e'  ,
    'kp_bs_dxy_sig',

    'km_pt'        ,
    'km_eta'       ,
    'km_phi'       ,
    'km_e'         ,
    'km_mass'      ,
    'km_charge'    ,
    'km_dxy'       ,
    'km_dxy_e'     ,
    'km_dxy_sig'   ,
    'km_dz'        ,
    'km_dz_e'      ,
    'km_dz_sig'    ,
    'km_bs_dxy'    ,
    'km_bs_dxy_e'  ,
    'km_bs_dxy_sig',

    'pi_pt'        ,
    'pi_eta'       ,
    'pi_phi'       ,
    'pi_e'         ,
    'pi_mass'      ,
    'pi_charge'    ,
    'pi_dxy'       ,
    'pi_dz'        ,

    'dr_m_kp'      ,
    'dr_m_km'      ,
    'dr_m_pi'      ,
    'dr_m_ds'      ,

    'pv_x'         ,
    'pv_y'         ,
    'pv_z'         ,

    'phi_vx'       ,
    'phi_vy'       ,
    'phi_vz'       ,
    'phi_vtx_chi2' ,
    'phi_vtx_prob' ,

    'ds_vx'        ,
    'ds_vy'        ,
    'ds_vz'        ,
    'ds_vtx_chi2'  ,
    'ds_vtx_prob'  ,

    'ds_m_vx'      ,
    'ds_m_vy'      ,
    'ds_m_vz'      ,
    'ds_m_vtx_chi2',
    'ds_m_vtx_prob',

    'bs_x0'        ,
    'bs_y0'        ,
    'bs_z0'        ,
    
    'cos3D_ds'     ,
    'lxyz_ds'      ,
    'lxyz_ds_err'  ,
    'lxyz_ds_sig'  ,

    'cos2D_ds'     ,
    'lxy_ds'       ,
    'lxy_ds_err'   ,
    'lxy_ds_sig'   ,

    'cos3D_ds_m'   ,    
    'lxyz_ds_m'    ,
    'lxyz_ds_m_err',
    'lxyz_ds_m_sig',

    'cos2D_ds_m'   ,
    'lxy_ds_m'     ,
    'lxy_ds_m_err' ,
    'lxy_ds_m_sig' ,

    'sig'          ,
]

paths = [
    'HLT_Mu7_IP4'     ,
    'HLT_Mu8_IP3'     ,
    'HLT_Mu8_IP5'     ,
    'HLT_Mu8_IP6'     ,
    'HLT_Mu8p5_IP3p5' ,
    'HLT_Mu9_IP4'     ,
    'HLT_Mu9_IP5'     ,
    'HLT_Mu9_IP6'     ,
    'HLT_Mu10p5_IP3p5',
    'HLT_Mu12_IP6'    ,
]

branches_reco += paths
branches_reco += [path+'_ps' for path in paths]

fout = ROOT.TFile(destination + '/' + fileName + '.root', 'recreate')
ntuple = ROOT.TNtuple('tree_gen', 'tree_gen', ':'.join(branches))
tofill = OrderedDict(zip(branches, [np.nan]*len(branches)))

ntuple_reco = ROOT.TNtuple('tree', 'tree', ':'.join(branches_reco))
tofill_reco = OrderedDict(zip(branches_reco, [np.nan]*len(branches_reco)))

for i, event in enumerate(events):

    if (i+1) > maxevents:
        break
            
    if i%logfreq == 0:
        percentage = float(i) / maxevents * 100.
        speed = float(i) / (time() - start)
        eta = datetime.now() + timedelta(seconds=(maxevents-i) / max(0.1, speed))
        print('\t===> processing %d / %d event \t completed %.1f%s \t %.1f ev/s \t ETA %s s' %(i, maxevents, percentage, '%', speed, eta.strftime('%Y-%m-%d %H:%M:%S')))

    # reset trees
    for k, v in tofill.items():
        tofill[k] = np.nan
    for k, v in tofill_reco.items():
        tofill_reco[k] = np.nan

    # access the handles
    for k, v in handles.iteritems():
        event.getByLabel(v[0], v[1])
        setattr(event, k, v[1].product())
    
    if mc:
        for k, v in handles_mc.iteritems():
            event.getByLabel(v[0], v[1])
            setattr(event, k, v[1].product())
    
    lumi = event.eventAuxiliary().luminosityBlock()
    iev  = event.eventAuxiliary().event()
   
    if mc: event.qscale = event.genInfo.qScale()
   
    if verbose: print('=========>')

    which_signal = np.nan
    
    if mc:
        event.genp = [ip for ip in event.genpr] + [ip for ip in event.genpk]
    
        dss   = [ip for ip in event.genp if abs(ip.pdgId())==431 and isMyDs(ip)]
        muons = [ip for ip in event.genpr if abs(ip.pdgId())==13 and ip.status()==1 and ip.pt()>7. and abs(ip.eta())<1.5]
    
        candidates = []
        for ids, imuon in product(dss, muons):
            icand = candidate(ids, imuon)
            if icand.charge()==0 and icand.p4().mass()<8.:
                ancestors = []
                printAncestors(icand.ds, ancestors, verbose=False)
                ancestors = []
                printAncestors(icand.muon, ancestors, verbose=False)
                candidates.append(icand)    
        
        if len(candidates)==0:
            # how is this possible?!
            if verbose: print('no candidates, WEIRD!')
            continue
                    
        if len(candidates)>1:
            print('more than one GEN candidate! Total %d candidates' %len(candidates))
#             import pdb ; pdb.set_trace()
#             continue

        candidates.sort(key = lambda x : (x.charge()==0, x.p4().pt()), reverse=True)
        cand = candidates[0]
    
        the_bs    = cand.ds.ancestors[-1] if len(cand.ds.ancestors) else None
        the_ds_st = None
        the_ds    = cand.ds
        the_phi   = cand.ds.phi_meson
        the_kp    = None
        the_km    = None
        the_pi    = cand.ds.pion
        the_mu    = cand.muon
        the_tau   = None
        which_signal = np.nan
    
        # check if signal
        if len(the_ds.ancestors)>0 and \
           len(the_mu.ancestors)>0 and \
           the_ds.ancestors[-1]==the_mu.ancestors[-1] and \
           abs(the_bs.pdgId())==531:

            daughters = []
            for idx_dau in range(the_bs.numberOfDaughters()):
                idau = the_bs.daughter(idx_dau)
                if idau.pdgId()==22:
                    continue
                daughters.append(idau.pdgId())
            daughters.sort(key = lambda x : abs(x))
               
            # save which signal is this
            # 0 Ds  mu nu
            # 1 Ds* mu nu
            # 2 Ds  tau nu
            # 3 Ds* tau nu
            if daughters==[13,-14,431] or daughters==[-13,14,-431]:
                which_signal = 0
            if daughters==[13,-14,433] or daughters==[-13,14,-433]:
                which_signal = 1
            if daughters==[15,-16,431] or daughters==[-15,16,-431]:
                which_signal = 2
            if daughters==[15,-16,433] or daughters==[-15,16,-433]:
                which_signal = 3

            if which_signal in [1, 3]:
                for idx_dau in range(the_bs.numberOfDaughters()):
                    idau = the_bs.daughter(idx_dau)
                    if abs(idau.pdgId())==433:
                        the_ds_st = idau

        if the_phi:
            for idx_dau in range(the_phi.numberOfDaughters()):
                idau = the_phi.daughter(idx_dau)
                if idau.pdgId()==321:
                    the_kp = idau
                    continue
                elif idau.pdgId()==-321:
                    the_km = idau
                    continue

        if abs(the_mu.mother(0).pdgId())==15:
            the_tau = the_mu.mother(0)

        if the_ds is None or \
           the_mu is None or \
           the_phi is None or \
           the_km is None or \
           the_kp is None or \
           the_pi is None:
            continue

        b_lab_p4 = the_mu.p4() + the_ds.p4()
        b_scaled_p4 = b_lab_p4 * ((particle.literals.B_s_0.mass/1000.)/b_lab_p4.mass())
    
        b_scaled_p4_tlv = ROOT.TLorentzVector() ; b_scaled_p4_tlv.SetPtEtaPhiE(b_scaled_p4.pt(), b_scaled_p4.eta(), b_scaled_p4.phi(), b_scaled_p4.energy())
        the_mu_p4_tlv = ROOT.TLorentzVector() ; the_mu_p4_tlv.SetPtEtaPhiE(the_mu.pt(), the_mu.eta(), the_mu.phi(), the_mu.energy())
    
        b_scaled_p4_boost = b_scaled_p4_tlv.BoostVector()
    
        the_mu_p4_in_b_rf = the_mu_p4_tlv.Clone(); the_mu_p4_in_b_rf.Boost(-b_scaled_p4_boost)
    
        tofill['run'          ] = event.eventAuxiliary().run()
        tofill['lumi'         ] = event.eventAuxiliary().luminosityBlock()
        tofill['event'        ] = event.eventAuxiliary().event()
        
        tofill['n_cands'      ] = len(candidates)

        tofill['ds_m_mass'    ] = b_lab_p4.mass()
        tofill['ds_m_pt'      ] = b_lab_p4.pt()
        tofill['ds_m_eta'     ] = b_lab_p4.eta()
        tofill['ds_m_phi'     ] = b_lab_p4.phi()
        tofill['max_dr'       ] = max([deltaR(the_mu, pp) for pp in [the_kp, the_km, the_pi]])
        tofill['m2_miss'      ] = (b_scaled_p4 - the_mu.p4() - the_ds.p4()).mass2()
        tofill['q2'           ] = (b_scaled_p4 - the_ds.p4()).mass2()
        tofill['e_star_mu'    ] = the_mu_p4_in_b_rf.E()

        tofill['mu_pt'        ] = the_mu.pt()
        tofill['mu_eta'       ] = the_mu.eta()
        tofill['mu_phi'       ] = the_mu.phi()
        tofill['mu_e'         ] = the_mu.energy()
        tofill['mu_mass'      ] = the_mu.mass()
        tofill['mu_charge'    ] = the_mu.charge()

        if the_tau:
            tofill['tau_pt'       ] = the_tau.pt()
            tofill['tau_eta'      ] = the_tau.eta()
            tofill['tau_phi'      ] = the_tau.phi()
            tofill['tau_e'        ] = the_tau.energy()
            tofill['tau_mass'     ] = the_tau.mass()
            tofill['tau_charge'   ] = the_tau.charge()

        if the_bs:
            tofill['hb_pt'        ] = the_bs.pt()
            tofill['hb_eta'       ] = the_bs.eta()
            tofill['hb_phi'       ] = the_bs.phi()
            tofill['hb_e'         ] = the_bs.energy()
            tofill['hb_mass'      ] = the_bs.mass()
            tofill['hb_charge'    ] = the_bs.charge()
            tofill['hb_pdgid'     ] = the_bs.pdgId()
            tofill['pv_x'         ] = the_bs.vertex().x()
            tofill['pv_y'         ] = the_bs.vertex().y()
            tofill['pv_z'         ] = the_bs.vertex().z()
        
    #         import pdb ; pdb.set_trace()
            L = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
                                                the_ds.vertex().x() - the_bs.vertex().x(),
                                                the_ds.vertex().y() - the_bs.vertex().y(),
                                                the_ds.vertex().z() - the_bs.vertex().z() )
            if L.R() > 0.:
                tofill['cos'       ] = b_lab_p4.Vect().Dot(L) / (b_lab_p4.Vect().R() * L.R())

        tofill['ds_pt'        ] = the_ds.pt()
        tofill['ds_eta'       ] = the_ds.eta()
        tofill['ds_phi'       ] = the_ds.phi()
        tofill['ds_e'         ] = the_ds.energy()
        tofill['ds_mass'      ] = the_ds.mass()
        tofill['ds_charge'    ] = the_ds.charge()

        tofill['ds_st_vx'     ] = the_ds.vertex().x()
        tofill['ds_st_vy'     ] = the_ds.vertex().y()
        tofill['ds_st_vz'     ] = the_ds.vertex().z()

        if the_ds_st:
            tofill['ds_st_pt'     ] = the_ds_st.pt()
            tofill['ds_st_eta'    ] = the_ds_st.eta()
            tofill['ds_st_phi'    ] = the_ds_st.phi()
            tofill['ds_st_e'      ] = the_ds_st.energy()
            tofill['ds_st_mass'   ] = the_ds_st.mass()
            tofill['ds_st_charge' ] = the_ds_st.charge()

            tofill['ds_st_vx'     ] = the_ds_st.vertex().x()
            tofill['ds_st_vy'     ] = the_ds_st.vertex().y()
            tofill['ds_st_vz'     ] = the_ds_st.vertex().z()

        tofill['phi_pt'       ] = the_phi.pt()
        tofill['phi_eta'      ] = the_phi.eta()
        tofill['phi_phi'      ] = the_phi.phi()
        tofill['phi_e'        ] = the_phi.energy()
        tofill['phi_mass'     ] = the_phi.mass()
        tofill['phi_charge'   ] = the_phi.charge()

        tofill['kp_pt'        ] = the_kp.pt()
        tofill['kp_eta'       ] = the_kp.eta()
        tofill['kp_phi'       ] = the_kp.phi()
        tofill['kp_e'         ] = the_kp.energy()
        tofill['kp_mass'      ] = the_kp.mass()
        tofill['kp_charge'    ] = the_kp.charge()

        tofill['km_pt'        ] = the_km.pt()
        tofill['km_eta'       ] = the_km.eta()
        tofill['km_phi'       ] = the_km.phi()
        tofill['km_e'         ] = the_km.energy()
        tofill['km_mass'      ] = the_km.mass()
        tofill['km_charge'    ] = the_km.charge()

        tofill['pi_pt'        ] = the_pi.pt()
        tofill['pi_eta'       ] = the_pi.eta()
        tofill['pi_phi'       ] = the_pi.phi()
        tofill['pi_e'         ] = the_pi.energy()
        tofill['pi_mass'      ] = the_pi.mass()
        tofill['pi_charge'    ] = the_pi.charge()
    
        tofill['same_b'       ] = (the_ds.ancestors[-1] == the_mu.ancestors[-1]) if (len(the_ds.ancestors)>0 and len(the_mu.ancestors)>0) else 0.
        tofill['sig'          ] = which_signal
    
        ntuple.Fill(array('f', tofill.values()))
    
    ######################################################################################
    #####      RECO PART HERE
    ######################################################################################
    
    trg_names = event.object().triggerNames(event.trg_res)

    hlt_passed = False

    for iname in trg_names.triggerNames():
        #if 'part0' not in iname: continue
        if 'part' not in iname: continue
        for ipath in paths:
            idx = len(trg_names)
            if iname.startswith(ipath):
                idx = trg_names.triggerIndex(iname)
                tofill_reco[ipath        ] = ( idx < len(trg_names)) * (event.trg_res.accept(idx))
                tofill_reco[ipath + '_ps'] = event.trg_ps.getPrescaleForIndex(idx)
                
                if not mc:
                    if ipath=='HLT_Mu7_IP4' and event.trg_ps.getPrescaleForIndex(idx)==5 and ( idx < len(trg_names)) * (event.trg_res.accept(idx)):
                        hlt_passed = True

    import pdb ; pdb.set_trace()

    if not (hlt_passed or mc) :
        continue            

    tracks = [tk for tk in event.trk if tk.charge()!=0] + [tk for tk in event.ltrk if tk.charge()!=0]
    tracks = [tk for tk in tracks if tk.pt()>1. and abs(tk.eta())<2.4 and tk.hasTrackDetails()]
    tracks = [tk for tk in tracks if abs(tk.pdgId()) not in [11, 13]]
    muons = [mu for mu in event.muons if mu.pt()>7. and abs(mu.eta())<1.5 and mu.isPFMuon() and mu.isGlobalMuon()]

    if len(muons)<1 or len(tracks)<3:
        if mc: fillRecoTree(ntuple_reco, tofill_reco, mc, which_signal)
        continue
    
    m_phi = 1.020
    m_ds = 1.96835
    m_k = 0.49368
    m_pi = 0.1396
    
    reco_candidates = []
        
#     print()
#     print('=========>')
#     print('GEN  mu pt %.2f   mu eta %.2f   mu phi %.2f  mu charge %d'%(tofill['mu_pt'], tofill['mu_eta'], tofill['mu_phi'], tofill['mu_charge'])) 
#     print('     pi pt %.2f   pi eta %.2f   pi phi %.2f  pi charge %d'%(tofill['pi_pt'], tofill['pi_eta'], tofill['pi_phi'], tofill['pi_charge']))
#     print('     kp pt %.2f   kp eta %.2f   kp phi %.2f  kp charge %d'%(tofill['kp_pt'], tofill['kp_eta'], tofill['kp_phi'], tofill['kp_charge']))
#     print('     km pt %.2f   km eta %.2f   km phi %.2f  km charge %d'%(tofill['km_pt'], tofill['km_eta'], tofill['km_phi'], tofill['km_charge']))
# #     print('     dsm mass %.2f dsm pt %.2f phi mass %.2f ds mass %.2f' %(tofill['ds_m_mass'], tofill['ds_m_pt'], the_phi.mass(), the_ds.mass()))
#     print()
# 
#     mu_bm, mu_dr2 = bestMatch(the_mu, muons )
#     pi_bm, pi_dr2 = bestMatch(the_pi, tracks)
#     kp_bm, kp_dr2 = bestMatch(the_kp, tracks)
#     km_bm, km_dr2 = bestMatch(the_km, tracks)
# 
# #     print('MU BM    pt %.2f   eta %.2f   phi %.2f  charge %d   dr %.3f         dz   %.3f' %(mu_bm.pt(), mu_bm.eta(), mu_bm.phi(), mu_bm.charge(), np.sqrt(mu_dr2),     mu_bm.bestTrack().dz()               ))
# #     print('PI BM    pt %.2f   eta %.2f   phi %.2f  charge %d   dr %.3f   Delta(dz)  %.3f' %(pi_bm.pt(), pi_bm.eta(), pi_bm.phi(), pi_bm.charge(), np.sqrt(pi_dr2), abs(mu_bm.bestTrack().dz() - pi_bm.dz()) ))
# #     print('KP BM    pt %.2f   eta %.2f   phi %.2f  charge %d   dr %.3f   Delta(dz)  %.3f' %(kp_bm.pt(), kp_bm.eta(), kp_bm.phi(), kp_bm.charge(), np.sqrt(kp_dr2), abs(mu_bm.bestTrack().dz() - kp_bm.dz()) ))
# #     print('KM BM    pt %.2f   eta %.2f   phi %.2f  charge %d   dr %.3f   Delta(dz)  %.3f' %(km_bm.pt(), km_bm.eta(), km_bm.phi(), km_bm.charge(), np.sqrt(km_dr2), abs(mu_bm.bestTrack().dz() - km_bm.dz()) ))
# #     print()
# 
#     the_primary_vertex = sorted( [vtx for vtx in event.vtx], key = lambda vtx : abs( mu_bm.bestTrack().dz(vtx.position() ) ) )[0]
#         
#     print('MU BM    pt %.2f   eta %.2f   phi %.2f  charge %d   dr %.3f         dz   %.3f' %(mu_bm.pt(), mu_bm.eta(), mu_bm.phi(), mu_bm.charge(), np.sqrt(mu_dr2),     mu_bm.bestTrack().dz(the_primary_vertex.position())                                            ))
#     print('PI BM    pt %.2f   eta %.2f   phi %.2f  charge %d   dr %.3f   Delta(dz)  %.3f' %(pi_bm.pt(), pi_bm.eta(), pi_bm.phi(), pi_bm.charge(), np.sqrt(pi_dr2), abs(mu_bm.bestTrack().dz(the_primary_vertex.position()) - pi_bm.dz(the_primary_vertex.position())) ))
#     print('KP BM    pt %.2f   eta %.2f   phi %.2f  charge %d   dr %.3f   Delta(dz)  %.3f' %(kp_bm.pt(), kp_bm.eta(), kp_bm.phi(), kp_bm.charge(), np.sqrt(kp_dr2), abs(mu_bm.bestTrack().dz(the_primary_vertex.position()) - kp_bm.dz(the_primary_vertex.position())) ))
#     print('KM BM    pt %.2f   eta %.2f   phi %.2f  charge %d   dr %.3f   Delta(dz)  %.3f' %(km_bm.pt(), km_bm.eta(), km_bm.phi(), km_bm.charge(), np.sqrt(km_dr2), abs(mu_bm.bestTrack().dz(the_primary_vertex.position()) - km_bm.dz(the_primary_vertex.position())) ))
#     print()

#     import pdb; pdb.set_trace()

    for ii, imu in enumerate(muons):
#         print('RECO MU %d:  pt %.2f  eta %.2f  phi %.2f  charge %d  dz %.3f' %(ii, imu.pt(), imu.eta(), imu.phi(), imu.charge(), imu.bestTrack().dz()))
        
        # choose as PV the one that's closest to the muon in the dz parameter
        the_primary_vertex = sorted( [vtx for vtx in event.vtx], key = lambda vtx : abs( imu.bestTrack().dz(vtx.position() ) ) )[0]

        bs_point = ROOT.reco.Vertex.Point(
            event.bs.x(the_primary_vertex.position().z()),
            event.bs.y(the_primary_vertex.position().z()),
            event.bs.z0(),
        )
    
        bs_error = event.bs.covariance3D()
        chi2 = 0.
        ndof = 0.
        bsvtx = ROOT.reco.Vertex(bs_point, bs_error, chi2, ndof, 2) # size? say 3? does it matter?

        kp = None
        km = None
        pi = None
        itracks = [tk for tk in tracks if deltaR(imu, tk)<1.2 and deltaR(imu, tk)>0.005 and abs(imu.bestTrack().dz(the_primary_vertex.position()) - tk.dz(the_primary_vertex.position()))<0.5]
        if len(itracks)<3: 
            continue

        phis = []
        ds_cands = []

        for ipair in combinations(itracks, 2):
            tk1 = ipair[0] ; tk2 = ipair[1]
            
            if tk1.charge() * tk2.charge() >= 0: continue

            tk1p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(tk1.pt(), tk1.eta(), tk1.phi(), m_k)
            tk2p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(tk2.pt(), tk2.eta(), tk2.phi(), m_k)
                                
            kp = tk1 if tk1.charge() > 0 else tk2
            km = tk1 if tk1.charge() < 0 else tk2
            phip4 = tk1p4 + tk2p4

#             if (tk1 == kp_bm and tk2 == km_bm) or \
#                (tk2 == kp_bm and tk1 == km_bm):
#                 import pdb ; pdb.set_trace()

            if abs(phip4.mass()-m_phi)>0.015: continue        

            # let's find a decent phi vertex at least
            tofit.clear()
            tofit.push_back(kp.bestTrack())
            tofit.push_back(km.bestTrack())
            phi_vtx = vtxfit.Fit(tofit)
            if not phi_vtx.isValid(): continue
            if (1. - stats.chi2.cdf(phi_vtx.normalisedChiSquared(), 1)) < 1e-2: continue

            # loop over pions
            pis = [tk for tk in itracks if tk!=tk1 and tk!=tk2]
            
            for pi in pis:
                pip4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(pi.pt(), pi.eta(), pi.phi(), m_pi)
                dsp4 = phip4 + pip4
                # check if they make a good ds candidate
#                 if abs(dsp4.mass()-m_ds)>0.15: continue
                if abs(dsp4.mass()-m_ds)>0.05: continue
                # let's find a decent phi vertex at least
                tofit.clear()
                tofit.push_back(kp.bestTrack())
                tofit.push_back(km.bestTrack())
                tofit.push_back(pi.bestTrack())
                ds_vtx = vtxfit.Fit(tofit)
                if not ds_vtx.isValid(): continue
                if (1. - stats.chi2.cdf(ds_vtx.normalisedChiSquared(), 1)) < 1e-2: continue
                ds_vtx.prob = (1. - stats.chi2.cdf(ds_vtx.normalisedChiSquared(), 1))
                
                dsmp4 = dsp4 + ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(imu.pt(), imu.eta(), imu.phi(), imu.mass())
                if dsmp4.mass()>8: continue

                Lxy_ds = ROOT.VertexDistanceXY().distance(bsvtx, ds_vtx.vertexState())

                vect_Lxy_ds = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
                                                    ds_vtx.position().x() - event.bs.x(the_primary_vertex.position().z()),
                                                    ds_vtx.position().y() - event.bs.y(the_primary_vertex.position().z()),
                                                    0. )

                vect_pt_ds = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
                                                    dsp4.px(),
                                                    dsp4.py(),
                                                    0. )

                if not(vect_Lxy_ds.R() > 0.):
                    continue
                 
                ds_vtx_cosine = vect_pt_ds.Dot(vect_Lxy_ds) / (vect_pt_ds.R() * vect_Lxy_ds.R())
                if ds_vtx_cosine<0.8:
                    continue
                #                        0    1   2   3   4      5     6      7        8       9              10
                reco_candidates.append( (imu, kp, km, pi, phip4, dsp4, dsmp4, phi_vtx, ds_vtx, ds_vtx_cosine, Lxy_ds) )    

    # sort by mass and max pt
    tofill_reco['n_cands'] = len(reco_candidates)
#     reco_candidates.sort(key = lambda x : ( (x[3].charge()*x[0].charge()<0., x[6].mass()<5.37, abs(x[4].mass()-m_phi)<0.02, x[9]>0.9, abs(x[5].mass()-m_ds)<0.05, x[6].pt()) ), reverse = True)
    reco_candidates.sort(key = lambda x : ( (x[3].charge()*x[0].charge()<0., x[6].mass()<5.37, x[6].pt()) ), reverse = True)
#     reco_candidates.sort(key = lambda x : ( (x[3].charge()*x[0].charge()<0., x[6].mass()<5.37, x[8].prob) ), reverse = True)
    
    if len(reco_candidates)>1 and verbose:
      
        print()
        print('GEN  mu pt %.2f mu eta %.2f mu phi %.2f pi pt %.2f pi eta %.2f pi phi %.2f dsm mass %.2f dsm pt %.2f phi mass %.2f ds mass %.2f' %(tofill['mu_pt'], tofill['mu_eta'], tofill['mu_phi'], tofill['pi_pt'], tofill['pi_eta'], tofill['pi_phi'], tofill['ds_m_mass'], tofill['ds_m_pt'], the_phi.mass(), the_ds.mass()))

        for icand in reco_candidates:
            mu            = icand[ 0]
            kp            = icand[ 1]
            km            = icand[ 2]
            pi            = icand[ 3]
            phi           = icand[ 4]
            ds            = icand[ 5]
            dsm           = icand[ 6]
            phi_vtx       = icand[ 7]
            ds_vtx        = icand[ 8]
            ds_vtx_cosine = icand[ 9]
            Lxy_ds        = icand[10]
    
            print('RECO mu pt %.2f mu eta %.2f mu phi %.2f pi pt %.2f pi eta %.2f pi phi %.2f dsm mass %.2f dsm pt %.2f phi mass %.2f ds mass %.2f cosine %.3f' %(mu.pt(), mu.eta(), mu.phi(), pi.pt(), pi.eta(), pi.phi(), dsm.mass(), dsm.pt(), phi.mass(), ds.mass(), ds_vtx_cosine))

#         import pdb ; pdb.set_trace()

    if len(reco_candidates)==0:
        if mc: fillRecoTree(ntuple_reco, tofill_reco, mc, which_signal)
        continue

    # fill the tree with the best candidate
    icand = reco_candidates[0]
    mu            = icand[ 0]
    kp            = icand[ 1]
    km            = icand[ 2]
    pi            = icand[ 3]
    phi           = icand[ 4]
    ds            = icand[ 5]
    dsm           = icand[ 6]
    phi_vtx       = icand[ 7]
    ds_vtx        = icand[ 8]
    ds_vtx_cosine = icand[ 9]
    Lxy_ds        = icand[10]

    # choose as PV the one that's closest to the muon in the dz parameter
    the_pv = sorted( [vtx for vtx in event.vtx], key = lambda vtx : abs( mu.bestTrack().dz(vtx.position() ) ) )[0]

    tofill_reco['run'  ] = event.eventAuxiliary().run()
    tofill_reco['lumi' ] = event.eventAuxiliary().luminosityBlock()
    tofill_reco['event'] = event.eventAuxiliary().event()
    tofill_reco['npv'  ] = len(event.vtx)

    tofill_reco['ds_m_mass'    ] = dsm.mass()
    tofill_reco['ds_m_pt'      ] = dsm.pt()
    tofill_reco['ds_m_eta'     ] = dsm.eta()
    tofill_reco['ds_m_phi'     ] = dsm.phi()

    b_scaled_p4 = dsm * (particle.literals.B_0.mass/1000.) / dsm.mass()
    
    # ROOT, geez...
    b_scaled_p4_tlv = ROOT.TLorentzVector() ; b_scaled_p4_tlv.SetPtEtaPhiE(b_scaled_p4.pt(), b_scaled_p4.eta(), b_scaled_p4.phi(), b_scaled_p4.energy())
    the_mu_p4_tlv = ROOT.TLorentzVector() ; the_mu_p4_tlv.SetPtEtaPhiE(mu.pt(), mu.eta(), mu.phi(), mu.energy())
    the_ds_p4_tlv = ROOT.TLorentzVector() ; the_ds_p4_tlv.SetPtEtaPhiE(ds.pt(), ds.eta(), ds.phi(), ds.energy())
    
    b_scaled_p4_boost = b_scaled_p4_tlv.BoostVector()
    
    the_mu_p4_in_b_rf = the_mu_p4_tlv.Clone(); the_mu_p4_in_b_rf.Boost(-b_scaled_p4_boost)
    
    tofill_reco['b_pt'         ] = b_scaled_p4_tlv.Pt()
    tofill_reco['b_eta'        ] = b_scaled_p4_tlv.Eta()
    tofill_reco['b_phi'        ] = b_scaled_p4_tlv.Phi()
    beta = b_scaled_p4_tlv.Beta()
    gamma = b_scaled_p4_tlv.Gamma()
    tofill_reco['b_beta'       ] = beta
    tofill_reco['b_gamma'      ] = gamma
    
    tofill_reco['m2_miss'      ] = (b_scaled_p4_tlv - the_mu_p4_tlv - the_ds_p4_tlv).M2()
    tofill_reco['q2'           ] = (b_scaled_p4_tlv - the_ds_p4_tlv).M2()
    tofill_reco['e_star_mu'    ] = the_mu_p4_in_b_rf.E()
    tofill_reco['pt_miss_sca'  ] = b_scaled_p4_tlv.Pt() - the_mu_p4_tlv.Pt() - the_ds_p4_tlv.Pt()
    tofill_reco['pt_miss_vec'  ] = (b_scaled_p4_tlv - the_mu_p4_tlv - the_ds_p4_tlv).Pt()
    tofill_reco['ptvar'        ] = the_ds_p4_tlv.Pt() - the_mu_p4_tlv.Pt()
   
    tofill_reco['mu_pt'        ] = mu.pt()
    tofill_reco['mu_eta'       ] = mu.eta()
    tofill_reco['mu_phi'       ] = mu.phi()
    tofill_reco['mu_e'         ] = mu.energy()
    tofill_reco['mu_mass'      ] = mu.mass()
    tofill_reco['mu_charge'    ] = mu.charge()
    tofill_reco['mu_id_loose'  ] = mu.isLooseMuon()
    tofill_reco['mu_id_soft'   ] = mu.isMediumMuon()
    tofill_reco['mu_id_medium' ] = mu.isSoftMuon(the_pv)
    tofill_reco['mu_id_tight'  ] = mu.isTightMuon(the_pv)
    
    # remove the Ds->KKpi candidate from the isolation
    
#     import pdb ; pdb.set_trace()
    mu_ch_isolation = max(0., mu.pfIsolationR03().sumChargedHadronPt - sum([ip.pt() for ip in [kp, km, pi] if deltaR(ip, mu)<0.3 and abs(mu.bestTrack().dz(the_pv.position()) - ip.dz(the_pv.position()))<0.2]))
#     if deltaR(mu, kp)<0.3 and mu_ch_isolation > kp.pt(): mu_ch_isolation -= kp.pt()
#     if deltaR(mu, km)<0.3 and mu_ch_isolation > km.pt(): mu_ch_isolation -= km.pt()
#     if deltaR(mu, pi)<0.3 and mu_ch_isolation > pi.pt(): mu_ch_isolation -= pi.pt()    
    
    mu_dbeta_neutral_isolation = max(mu.pfIsolationR03().sumNeutralHadronEt + mu.pfIsolationR03().sumPhotonEt - mu.pfIsolationR03().sumPUPt/2,0.0)
    
    tofill_reco['mu_ch_iso'    ] = mu_ch_isolation
    tofill_reco['mu_db_n_iso'  ] = mu_dbeta_neutral_isolation
    tofill_reco['mu_abs_iso'   ] = (mu_ch_isolation + mu_dbeta_neutral_isolation)
    tofill_reco['mu_rel_iso'   ] = (mu_ch_isolation + mu_dbeta_neutral_isolation) / mu.pt()
#     tofill_reco['mu_iso'       ] = (mu.pfIsolationR03().sumChargedHadronPt + max(mu.pfIsolationR03().sumNeutralHadronEt + mu.pfIsolationR03().sumPhotonEt - mu.pfIsolationR03().sumPUPt/2,0.0))/mu.pt()
    tofill_reco['mu_dxy'       ] = mu.bestTrack().dxy(the_pv.position())
    tofill_reco['mu_dxy_e'     ] = mu.bestTrack().dxyError(the_pv.position(), the_pv.error())
    tofill_reco['mu_dxy_sig'   ] = mu.bestTrack().dxy(the_pv.position()) / mu.bestTrack().dxyError(the_pv.position(), the_pv.error())
    tofill_reco['mu_dz'        ] = mu.bestTrack().dz(the_pv.position())
    tofill_reco['mu_dz_e'      ] = mu.bestTrack().dzError()
    tofill_reco['mu_dz_sig'    ] = mu.bestTrack().dz(the_pv.position()) / mu.bestTrack().dzError()
    tofill_reco['mu_bs_dxy'    ] = mu.bestTrack().dxy(event.bs)
    tofill_reco['mu_bs_dxy_e'  ] = mu.bestTrack().dxyError(event.bs)
    tofill_reco['mu_bs_dxy_sig'] = mu.bestTrack().dxy(event.bs) / mu.bestTrack().dxyError(event.bs)
    
    tofill_reco['ds_pt'        ] = ds.pt()
    tofill_reco['ds_eta'       ] = ds.eta()
    tofill_reco['ds_phi'       ] = ds.phi()
    tofill_reco['ds_e'         ] = ds.energy()
    tofill_reco['ds_mass'      ] = ds.mass()

    tofill_reco['phi_pt'       ] = phi.pt()
    tofill_reco['phi_eta'      ] = phi.eta()
    tofill_reco['phi_phi'      ] = phi.phi()
    tofill_reco['phi_e'        ] = phi.energy()
    tofill_reco['phi_mass'     ] = phi.mass()

    # redefine K p4 to account for the kaon mass hypothesis
    kp_tlv = ROOT.TLorentzVector()
    kp_tlv.SetPtEtaPhiM(
        kp.pt() , 
        kp.eta(), 
        kp.phi(), 
        m_k
    )

    km_tlv = ROOT.TLorentzVector()
    km_tlv.SetPtEtaPhiM(
        km.pt() , 
        km.eta(), 
        km.phi(), 
        m_k
    )

    tofill_reco['kp_pt'        ] = kp_tlv.pt()
    tofill_reco['kp_eta'       ] = kp_tlv.eta()
    tofill_reco['kp_phi'       ] = kp_tlv.phi()
    tofill_reco['kp_e'         ] = kp_tlv.energy()
    tofill_reco['kp_mass'      ] = kp_tlv.mass()
    tofill_reco['kp_charge'    ] = kp.charge()
    tofill_reco['kp_dxy'       ] = kp.dxy(the_pv.position())
    tofill_reco['kp_dxy_e'     ] = kp.bestTrack().dxyError(the_pv.position(), the_pv.error())
    tofill_reco['kp_dxy_sig'   ] = kp.bestTrack().dxy(the_pv.position()) / kp.bestTrack().dxyError(the_pv.position(), the_pv.error())
    tofill_reco['kp_dz'        ] = kp.dz(the_pv.position())
    tofill_reco['kp_dz_e'      ] = kp.bestTrack().dzError()
    tofill_reco['kp_dz_sig'    ] = kp.bestTrack().dz(the_pv.position()) / kp.bestTrack().dzError()
    tofill_reco['kp_bs_dxy'    ] = kp.bestTrack().dxy(event.bs)
    tofill_reco['kp_bs_dxy_e'  ] = kp.bestTrack().dxyError(event.bs)
    tofill_reco['kp_bs_dxy_sig'] = kp.bestTrack().dxy(event.bs) / kp.bestTrack().dxyError(event.bs)

    tofill_reco['km_pt'        ] = km_tlv.pt()
    tofill_reco['km_eta'       ] = km_tlv.eta()
    tofill_reco['km_phi'       ] = km_tlv.phi()
    tofill_reco['km_e'         ] = km_tlv.energy()
    tofill_reco['km_mass'      ] = km_tlv.mass()
    tofill_reco['km_charge'    ] = km.charge()
    tofill_reco['km_dxy_e'     ] = km.bestTrack().dxyError(the_pv.position(), the_pv.error())
    tofill_reco['km_dxy_sig'   ] = km.bestTrack().dxy(the_pv.position()) / km.bestTrack().dxyError(the_pv.position(), the_pv.error())
    tofill_reco['km_dz'        ] = km.dz(the_pv.position())
    tofill_reco['km_dz_e'      ] = km.bestTrack().dzError()
    tofill_reco['km_dz_sig'    ] = km.bestTrack().dz(the_pv.position()) / km.bestTrack().dzError()
    tofill_reco['km_bs_dxy'    ] = km.bestTrack().dxy(event.bs)
    tofill_reco['km_bs_dxy_e'  ] = km.bestTrack().dxyError(event.bs)
    tofill_reco['km_bs_dxy_sig'] = km.bestTrack().dxy(event.bs) / km.bestTrack().dxyError(event.bs)

    tofill_reco['pi_pt'        ] = pi.pt()
    tofill_reco['pi_eta'       ] = pi.eta()
    tofill_reco['pi_phi'       ] = pi.phi()
    tofill_reco['pi_e'         ] = pi.energy()
    tofill_reco['pi_mass'      ] = pi.mass()
    tofill_reco['pi_charge'    ] = pi.charge()
    tofill_reco['pi_dxy'       ] = pi.dxy(the_pv.position())
    tofill_reco['pi_dz'        ] = pi.dz(the_pv.position())

    tofill_reco['dr_m_kp'      ] = deltaR(mu, kp)
    tofill_reco['dr_m_km'      ] = deltaR(mu, kp)
    tofill_reco['dr_m_pi'      ] = deltaR(mu, pi)
    tofill_reco['dr_m_ds'      ] = deltaR(mu, ds)

    tofill_reco['pv_x'         ] = the_pv.position().x()
    tofill_reco['pv_y'         ] = the_pv.position().y()
    tofill_reco['pv_z'         ] = the_pv.position().z()

    tofill_reco['bs_x0'        ] = event.bs.x0()
    tofill_reco['bs_y0'        ] = event.bs.y0()
    tofill_reco['bs_z0'        ] = event.bs.z0()

    # phi -> KK vertex fit
    tofill_reco['phi_vx'] = phi_vtx.position().x()
    tofill_reco['phi_vy'] = phi_vtx.position().y()
    tofill_reco['phi_vz'] = phi_vtx.position().z()
    tofill_reco['phi_vtx_chi2'] = phi_vtx.normalisedChiSquared()
    tofill_reco['phi_vtx_prob'] = 1. - stats.chi2.cdf(phi_vtx.normalisedChiSquared(), 1)

    # Ds -> KKpi vertex fit
    tofill_reco['ds_vx'] = ds_vtx.position().x()
    tofill_reco['ds_vy'] = ds_vtx.position().y()
    tofill_reco['ds_vz'] = ds_vtx.position().z()
    tofill_reco['ds_vtx_chi2'] = ds_vtx.normalisedChiSquared()
    tofill_reco['ds_vtx_prob'] = 1. - stats.chi2.cdf(ds_vtx.normalisedChiSquared(), 1)

    L_ds = ROOT.VertexDistance3D().distance(the_pv, ds_vtx.vertexState())
    vect_L_ds = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
                                        ds_vtx.position().x() - the_pv.position().x(),
                                        ds_vtx.position().y() - the_pv.position().y(),
                                        ds_vtx.position().z() - the_pv.position().z() )
    
    if vect_L_ds.R() > 0.:
        tofill_reco['cos3D_ds'   ] = ds.Vect().Dot(vect_L_ds) / (ds.Vect().R() * vect_L_ds.R())
        tofill_reco['lxyz_ds'    ] = L_ds.value()
        tofill_reco['lxyz_ds_err'] = L_ds.error()
        tofill_reco['lxyz_ds_sig'] = L_ds.significance()
    
#     if vect_Lxy_ds.R() > 0.:
    tofill_reco['cos2D_ds'  ] = ds_vtx_cosine
    tofill_reco['lxy_ds'    ] = Lxy_ds.value()
    tofill_reco['lxy_ds_err'] = Lxy_ds.error()
    tofill_reco['lxy_ds_sig'] = Lxy_ds.significance()
    
    # Ds -> KKpi + mu vertex fit
    tofit.clear()
    tofit.push_back(kp.bestTrack())
    tofit.push_back(km.bestTrack())
    tofit.push_back(pi.bestTrack())
    tofit.push_back(mu.bestTrack())
    ds_m_vtx = vtxfit.Fit(tofit)
    if ds_m_vtx.isValid():
        tofill_reco['ds_m_vx'] = ds_m_vtx.position().x()
        tofill_reco['ds_m_vy'] = ds_m_vtx.position().y()
        tofill_reco['ds_m_vz'] = ds_m_vtx.position().z()
        tofill_reco['ds_m_vtx_chi2'] = ds_m_vtx.normalisedChiSquared()
        tofill_reco['ds_m_vtx_prob'] = 1. - stats.chi2.cdf(ds_m_vtx.normalisedChiSquared(), 1)

        L_ds_m = ROOT.VertexDistance3D().distance(the_pv, ds_m_vtx.vertexState())

        vect_L_ds_m = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
                                            ds_m_vtx.position().x() - the_pv.position().x(),
                                            ds_m_vtx.position().y() - the_pv.position().y(),
                                            ds_m_vtx.position().z() - the_pv.position().z() )
        if vect_L_ds_m.R() > 0.:
            tofill_reco['cos3D_ds_m'   ] = dsm.Vect().Dot(vect_L_ds_m) / (dsm.Vect().R() * vect_L_ds_m.R())
            tofill_reco['lxyz_ds_m'    ] = L_ds_m.value()
            tofill_reco['lxyz_ds_m_err'] = L_ds_m.error()
            tofill_reco['lxyz_ds_m_sig'] = L_ds_m.significance()
            # now, lifetime L = beta * gamma * c * t ===> t = (L)/(beta*gamma*c)
            ct = L_ds_m.value() / (beta * gamma)
            tofill_reco['b_ct'         ] = ct
        
        
        Lxy_ds_m = ROOT.VertexDistanceXY().distance(bsvtx, ds_m_vtx.vertexState())

        vect_Lxy_ds_m = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
                                            ds_m_vtx.position().x() - event.bs.x(the_pv.position().z()),
                                            ds_m_vtx.position().y() - event.bs.y(the_pv.position().z()),
                                            0. )

        vect_pt_ds_m = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
                                            dsm.px(),
                                            dsm.py(),
                                            0. )

        if vect_Lxy_ds_m.R() > 0.:
            tofill_reco['cos2D_ds_m'  ] = vect_pt_ds_m.Dot(vect_Lxy_ds_m) / (vect_pt_ds_m.R() * vect_Lxy_ds_m.R())
            tofill_reco['lxy_ds_m'    ] = Lxy_ds_m.value()
            tofill_reco['lxy_ds_m_err'] = Lxy_ds_m.error()
            tofill_reco['lxy_ds_m_sig'] = Lxy_ds_m.significance()
    
    fillRecoTree(ntuple_reco, tofill_reco, mc, which_signal)
#     if mc: tofill_reco['sig'] = which_signal
#     ntuple_reco.Fill(array('f', tofill_reco.values()))
            
fout.cd()
if mc:
    ntuple_reco.AddFriend(ntuple)
    ntuple.Write()
ntuple_reco.Write()
fout.Close()


