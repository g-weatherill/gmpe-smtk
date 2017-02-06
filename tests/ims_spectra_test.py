# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2017 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.
"""
Unit Test Suite for the OpenQuake Ground Motion Toolkit

As of February 2017 the unit test suite verifies only execution -
not numerical correctness

Plotting functionalities are not verified
"""
import os
import unittest
import numpy as np
from nose.plugins.attrib import attr
import smtk.response_spectrum as rsp
import smtk.intensity_measures as ims
from smtk.smoothing.konno_ohmachi import KonnoOhmachi


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

class BasicResponseSpectrumIMS(unittest.TestCase):
    """
    Test class to run the basic response spectrum and intensity
    measure calculations
    """
    def setUp(self):
        self.periods = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1,
                                 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
                                 0.18, 0.19, 0.20, 0.22, 0.24, 0.26, 0.28,
                                 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42,
                                 0.44, 0.46, 0.48, 0.5, 0.55, 0.6, 0.65, 0.7,
                                 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2,
                                 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2,
                                 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0,
                                 4.2, 4.4, 4.6, 4.8, 5.0, 5.5, 6.0, 6.5, 7.0,
                                 7.5, 8.0, 8.5, 9.0, 9.5, 10.0], dtype=float)
        self.number_periods = len(self.periods)
        self.x_record = np.genfromtxt(os.path.join(BASE_DATA_PATH,
                                                   "sm_record_x.txt"))
        self.y_record = np.genfromtxt(os.path.join(BASE_DATA_PATH,
                                                   "sm_record_y.txt"))
        self.x_time_step = 0.002
        self.y_time_step = 0.002

    def test_nigam_jennings(self):
        """
        Tests the execution of the Nigam & Jennings algorithm
        """
        nigam_jennings = rsp.NigamJennings(self.x_record,
                                           self.x_time_step,
                                           self.periods,
                                           damping=0.05,
                                           units="cm/s/s")
        sax, time_series, acc, vel, dis = nigam_jennings.evaluate()

    def test_newmark_beta(self):
        """
        Tests the execution of the Newmark-Beta algorithm
        """
        newmark_beta = rsp.NewmarkBeta(self.x_record,
                                       self.x_time_step,
                                       self.periods,
                                       damping=0.05,
                                       units="cm/s/s")
        sax, time_series, acc, vel, dis = newmark_beta.evaluate()
    
    def test_peak_ims(self):
        """
        Tests the retreival of the peak intensity measurements
        """
        pga_x, pgv_x, pgd_x, _, _ = ims.get_peak_measures(self.x_time_step,
                                                          self.x_record,
                                                          True, True)

    def test_significant_durations(self):
        """
        Tests the calculation of duration metrics
        """
        t_br = ims.get_bracketed_duration(self.x_record, self.x_time_step, 5.0)
        t_ud = ims.get_uniform_duration(self.x_record, self.x_time_step, 5.0)
        t_sd = ims.get_significant_duration(self.x_record,
                                            self.x_time_step,
                                            0.05, 0.95)

    def test_arias_cav_arms(self):
        """
        Tests the calculation of Arias intensity, CAV, CAV5 and Arms
        """
        i_a = ims.get_arias_intensity(self.x_record, self.x_time_step)
        i_a_5_95 = ims.get_arias_intensity(self.x_record,
                                        self.x_time_step, 0.05, 0.95)
        cav = ims.get_cav(self.x_record, self.x_time_step)
        cav5 = ims.get_cav(self.x_record, self.x_time_step, threshold=5.0)
        arms = ims.get_arms(self.x_record, self.x_time_step)

    def test_spectrum_intensities(self):
        """
        Tests the velocity and acceleration intensity
        """
        sax = ims.get_response_spectrum(self.x_record,
                                        self.x_time_step,
                                        self.periods)[0]
        vsi = ims.get_response_spectrum_intensity(sax)
        asi = ims.get_acceleration_spectrum_intensity(sax)

    def test_combining_spectra(self):
        """
        Tests code for combining spectra
        """
        sax, say = ims.get_response_spectrum_pair(self.x_record,
                                                  self.x_time_step,
                                                  self.y_record,
                                                  self.y_time_step,
                                                  self.periods,
                                                  damping=0.05,
                                                  units="cm/s/s",
                                                  method="Nigam-Jennings")
        sa_gm = ims.geometric_mean_spectrum(sax, say)
        sa_env = ims.envelope_spectrum(sax, say)


    def test_gmrot_spectra(self):
        """
        Tests the retreival of the geometric mean rotational spectra
        """
        gmrotd50 = ims.gmrotdpp(self.x_record, self.x_time_step,
                                self.y_record, self.y_time_step,
                                self.periods, percentile=50.0,
                                damping=0.05, units="cm/s/s")
        gmroti50 = ims.gmrotipp(self.x_record, self.x_time_step,
                                self.y_record, self.y_time_step,
                                self.periods, percentile=50.0,
                                damping=0.05, units="cm/s/s")
    
    @attr("slow")
    def test_rot_spectra(self):
        """
        Tests the execution of the RotIpp spectra
        """
        rotd50 = ims.rotdpp(self.x_record, self.x_time_step,
                            self.y_record, self.y_time_step,
                            self.periods, percentile=50.0,
                            damping=0.05, units="cm/s/s")[0]
        roti50 = ims.rotipp(self.x_record, self.x_time_step,
                            self.y_record, self.y_time_step,
                            self.periods, percentile=50.0,
                            damping=0.05, units="cm/s/s")

    def test_get_fourier_spectrum(self):
        """
        Tests the fourier spectrum calculation
        """
        freq, amplitude = ims.get_fourier_spectrum(self.x_record,
                                                   self.x_time_step)

    def test_konno_ohmachi_smoothing(self):
        """
        Tests the smoothing using the Konno & Ohmachi method
        """
        freq, amplitude = ims.get_fourier_spectrum(self.x_record,
                                                   self.x_time_step)
        smoothing_config = {"bandwidth": 40,
                            "count": 1,
                            "normalize": True} 

        # Apply the Smoothing
        smoother = KonnoOhmachi(smoothing_config)
        smoothed_spectra = smoother.apply_smoothing(amplitude, freq)

    def test_hvsr(self):
        """
        Tests the execution of the HVSR
        """
        record_file = os.path.join(BASE_DATA_PATH, "record_3component.csv")
        record_3comp = np.genfromtxt(record_file, delimiter=",")
 
        time_vector = record_3comp[:, 0]
        x_record = record_3comp[:, 1]
        y_record = record_3comp[:, 2]
        v_record = record_3comp[:, 3]
        time_step = 0.002
        params = {"Function": "KonnoOhmachi",
                  "bandwidth": 40.0,
                  "count": 1.0,
                  "normalize": True}
        hvsr, freq, max_hv, t_0 = ims.get_hvsr(x_record, time_step,
                                               y_record, time_step,
                                               v_record, time_step, params)
