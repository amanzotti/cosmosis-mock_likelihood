'''
I want to write a nice mock likelihood module. This would be used to go beyond Fisher. It will be very usefull if we used the right non gaussian likelihood for cmb S4

TODO: add noise. Remmeber it has to go into both.


'''

import os
import sys
from cosmosis.datablock.cosmosis_py.section_names import likelihoods, cmb_cl
from cosmosis.datablock import option_section
dirname = os.path.split(__file__)[0]
import matplotlib.pyplot as plt
import numpy as np
import sys  # TODO
#  vectorize the l thing it should be possible.

def years2sec(years):
    ''' years to sec '''
    return years * 365 * 24. * 60. * 60.


def fsky2arcmin(fsky):
    '''convert fsky in fraction of unity to arcmin^2'''
    # 41253 square degrees in all sky
    return 41253. * fsky * 60. * 60.


def setup(options):
    # From the directory the data is stored in,
    # load the various files they want.

        # load CLs
    section = option_section
    l_max = options.get_int(section, "l_max", default=4000)
    l_min = options.get_int(section, "l_min", default=4)
    Y = options.get_double(section, "Y", default=0.25)
    years = options.get_double(section, "years", default=5.)
    beam_arcmin = options.get_double(section, "beam_arcmin", default=10.)
    N_det = options.get_double(section, "N_det", default=1e5)
    fsky = options.get_double(section, "f_sky", default=1.)
    arcmin_from_fsky = fsky2arcmin(fsky)
    sec_of_obs = years2sec(years)

    s = 350. * np.sqrt(arcmin_from_fsky) / np.sqrt(N_det * Y * sec_of_obs)  # half sky in arcmin^2
    t = beam_arcmin / 60. / 180. * np.pi  # 2arcmin to rads beam
    # N = (s * np.pi / 180. / 60.) ** 2 * np.exp(ell * (ell + 1.) * t ** 2 / 8. / np.log(2))
# you would probaly pass s and t and you can compute N_l on the spot

    # TEST

    #  Set the obserrved specctra to the theoritical one pluse noise.

    mock_data_tt = np.loadtxt(
        '/home/manzotti/cosmosis/modules/likelihoods/cmb_mock_likelihood/cmb_cl/tt.txt')
    mock_data_te = np.loadtxt(
        '/home/manzotti/cosmosis/modules/likelihoods/cmb_mock_likelihood/cmb_cl/te.txt')
    mock_data_ee = np.loadtxt(
        '/home/manzotti/cosmosis/modules/likelihoods/cmb_mock_likelihood/cmb_cl/ee.txt')
    # mock_data_pp = np.loadtxt(
    #     '/home/manzotti/cosmosis/modules/likelihoods/cmb_mock_likelihood/output/cmb_cl/pp.txt')
    mock_data_ell = np.loadtxt(
        '/home/manzotti/cosmosis/modules/likelihoods/cmb_mock_likelihood/cmb_cl/ell.txt')
    print
    # mock_data = mock_data[0:np.int(np.where(mock_data[:, 0] == l_max)[0]), :]
    # print 'here',mock_data_tt[100]
    # read the cl here and pass those to the execute.
    # actually even better you should already create the 3x3 matrix structure for each ell
    # it would be a [ell,3,3] matrix
    N_ell = np.zeros_like(mock_data_ell[:])
    C_ell_cov = np.zeros((np.shape(mock_data_ell)[0], 2, 2))
    for i, ell in enumerate(mock_data_ell[:]):
        # create 3x3 matrix
        # CTT   CTE  CTphi
        # CTE   CEE  0
        # CTphi 0    Cphi
        # print ell
        # we use the _lenspotentialcls so

         # l CTT CEE CBB CTE Cdd CdT CdE
         # 0  1  2    3   4   5   6  7
        N_ell[i] = (s * np.pi / 180. / 60.) ** 2 * np.exp(ell * (ell + 1.) * t ** 2 / 8. / np.log(2))
        # C_ell_cov[i, 0, 0] = mock_data_lensed[i, 1]
        # C_ell_cov[i, 0, 1] = C_ell_cov[i, 1, 0] = mock_data_lensed[i, 4]
        # # C_ell_cov[i,0,2]=C_ell_cov[i,2,0]=mock_data[i,6]
        # C_ell_cov[i, 1, 1] = mock_data_lensed[i, 2]
        # # C_ell[i,1,2]=0.0
        # C_ell_cov[i, 2, 2] = mock_data[i, 5]
        # # print ell,C_ell_cov[i,:,:]
        # print ell,mock_data_tt[i] ,N_ell[i] * ell * (ell + 1.) / 2. / np.pi (2.75e6)**2*
        C_ell_cov[i, 0, 0] = mock_data_tt[i] +  N_ell[i] * ell * (ell + 1.) / 2. / np.pi
        C_ell_cov[i, 0, 1] = C_ell_cov[i, 1, 0] = mock_data_te[i]
        # C_ell_cov[i,0,2]=C_ell_cov[i,2,0]=mock_data_tt[i]
        C_ell_cov[i, 1, 1] = mock_data_ee[i] + 2. * N_ell[i] * ell * (ell + 1.) / 2. / np.pi
        # C_ell[i,1,2]=0.0
        # C_ell_cov[i, 2, 2] = mock_data_pp[i] * 2
        if ell==100: print ell,C_ell_cov[i,:,:] / 7.4311e12
        # A quick warning for the unwary user...
    print '========='
    print "You're using the mock likelihood code."
    print '========='
    # plt.figure()
    # print('plotting')
    # plt.loglog(mock_data_ell[:],N_ell* mock_data_ell * (mock_data_ell + 1.) / 2. / np.pi)
    # plt.loglog(mock_data_ell[:],mock_data_tt[:])
    # plt.loglog(mock_data_ell[:],mock_data_ee[:])
    # plt.loglog(mock_data_ell[:],mock_data_te[:])
    # plt.savefig("/home/manzotti/cosmosis/modules/likelihoods/cmb_mock_likelihood/test.pdf")
    # plt.close()
    return C_ell_cov, mock_data_ell, N_ell  # \ mock_data[0, :]


def execute(block, config):
    # Get back all the stuff we loaded in during setup.
    C_l_mock_data, ell_data, N_ell = config

    # Load ell and Cls
    # Get right ell range
    ell_theory = block[cmb_cl, "ELL"]
    cl_tt_theory = block[cmb_cl, "tt"]
    cl_ee_theory = block[cmb_cl, "ee"]
    cl_te_theory = block[cmb_cl, "te"]

    # cl_phiphi_theory = block[cmb_cl, "PP"]
    # for i,ell in enumerate(ell_theory):
    # print cl_phiphi_theory[i]*ell_theory[i]*(ell_theory[i]+1.)/np.pi/2. ,C_l_mock_data[i,2,2]
    # print cl_tt_theory[i],C_l_mock_data[i,0,0]*7.4311e12


    # np.savetxt('noise.txt',N_ell)
    # Make sure we go up to high enough lmax
    # if not (theory_ell_max>=data_ell_max):
    #     sys.stderr.write("You did not calculate the CMB to a high enough lmax needed lmax=%d but got %d\n" %(data_ell_max, theory_ell_max))
    #     return 1

    # Initialize to zero
    C_ell_temp = np.zeros((2, 2))
    like = 0.0
    like_ell = 0.0
    # sys.exit()
    for i, ell in enumerate(ell_theory):
        # for ell in ells:
        # create 3x3 matrix
        # CTT   CTE  CTphi
        # CTE   CEE  0
        # CTphi 0    Cphi
        # compute noise
        # N = (s * np.pi / 180. / 60.) ** 2 * np.exp(ell * (ell + 1.) * t ** 2 / 8. / np.log(2))
        C_ell_temp[0, 0] = cl_tt_theory[i] + N_ell[i] * ell * (ell + 1.) / 2. / np.pi
        # print ell, C_ell_temp[0, 0]/7.4311e12
        C_ell_temp[0, 1] = C_ell_temp[1, 0] = cl_te_theory[i]  # / 7.4311e12
        # C_ell_temp[0, 2] = 0.
        C_ell_temp[1, 1] = cl_ee_theory[i] + 2. * N_ell[i] * ell * (ell + 1.) / 2. / np.pi
        # C_ell_temp[1, 2] = 0.0
        # C_ell_temp[2, 2] = cl_phiphi_theory[i]  *2. # * ell * (ell + 1.)

        # checked that the equation is the same of arXiv:astro-ph/0606227v1
        # print C_ell_temp
        # print 'next'
        like_ell = -0.5 * (2 * ell + 1.) * (np.log(np.linalg.det(C_ell_temp)) + np.trace(np.dot(C_l_mock_data[i, :, :], np.linalg.inv(C_ell_temp))))

        # if ell==100 : print i,C_ell_temp[:, :],like_ell

        # factor = C_l_mock_data[i, 0, 0]*cl_ee_theory[i] / 7.4311e12*(cl_phiphi_theory[i] * ell * (ell + 1.) / np.pi / 2.)+cl_tt_theory[i] / 7.4311e12*C_l_mock_data[i, 1, 1]*(cl_phiphi_theory[i] * ell * (ell + 1.) / np.pi / 2.)+cl_tt_theory[i] / 7.4311e12*cl_ee_theory[i] / 7.4311e12*(C_l_mock_data[i, 2, 2])- (cl_te_theory[i] / 7.4311e12)*( cl_te_theory[i] / 7.4311e12 * C_l_mock_data[i, 2, 2] + 2.*C_l_mock_data[i, 0, 1]*(cl_phiphi_theory[i] * ell * (ell + 1.) / np.pi / 2.))
        # print np.trace(np.dot(C_l_mock_data[i, :, :], np.linalg.inv(C_ell_temp))), factor/np.linalg.det(C_ell_temp)
        like = like + like_ell
        # compute the likelihhod
    # Get the likelihood using full sky formula http://arxiv.org/pdf/0801.0554v4.pdf eq 6
    # And save it
    block[likelihoods, "forecast_LIKE"] = like + 279116405

    # Signal that all is well
    return 0


def cleanup(config):
    # nothing to do here!  We just include this
    # for completeness
    return 0
