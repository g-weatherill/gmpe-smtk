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

Tests for the trellis plotting functionalities

As of February 2017 the unit test suite verifies only execution -
not numerical correctness

Plotting functionalities are not verified
"""
import os
import unittest
import numpy as np
import matplotlib
import nose
matplotlib.use("agg")
nose.main()
#from matplotlib.testing.decorators import cleanup
from nose.plugins.attrib import attr
import smtk.trellis.trellis_plots as trpl
import smtk.trellis.configure as rcfg


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
TMP_PATH = os.path.join(os.path.dirname(__file__), "tmp")


class TrellisPlotTestCase(unittest.TestCase):
    """
    Class to verify the trellis plotting functions when using the long
    (i.e. user solves the geometry themselves approach)
    """
    def setUp(self):
        self.gmpe_list = ["AkkarBommer2010", 
                          "AkkarCagnan2010", 
                          "AkkarEtAlRjb2014", 
                          "BooreAtkinson2008", 
                          "ChiouYoungs2008",
                          "ZhaoEtAl2006Asc"]

        self.imts = ["PGA", "SA(0.2)", "SA(1.0)", "SA(2.0)"]

        self.params = {"ztor": 5.0,
                       "hypo_depth": 10.0,
                       "vs30": 800.0,
                       "vs30measured": True,
                       "z1pt0": 100.0,
                       "dip": 90.0,
                       "rake": 0.0}
        os.system(TMP_PATH)
        #self.tmpdir = os.path.join(os.path.dirname(__file__), "tmp")

    def test_magnitude_imt_trellis(self):
        """
        Tests execution of Magnitude-IMT trellis
        """
        magnitudes = np.arange(4.5, 8.1, 0.1)
        distances = {"repi": 20.0,
                     "rhypo": 22.5,
                     "rjb": 15.0,
                     "rrup": 16.0,
                     "rx": 15.0}
        # Tests execution of the plotting
        mag_imt_trellis = trpl.MagnitudeIMTTrellis(
            magnitudes, distances, gmpe_list, imts, self.params,
            figure_size=(7,5),
            filename=os.path.join(TMP_PATH,
                                  "magnitude_imt_trellis_simple.pdf"),
            filetype="pdf",
            dpi=150)
        # Tests export
        mag_imt_trellis.pretty_print(
            os.path.join(TMP_PATH, "magnitude_imt_trellis1.csv"),
            sep=",")

    def test_magnitude_imt_sigma_trellis(self):
        """
        Tests execution of Magnitude-IMT sigma trellis
        """
        magnitudes = np.arange(4.5, 8.1, 0.1)
        distances = {"repi": 20.0,
                     "rhypo": 22.5,
                     "rjb": 15.0,
                     "rrup": 16.0,
                     "rx": 15.0}
        # Tests execution of the plotting
        mag_imt_trellis = trpl.MagnitudeSigmaIMTTrellis(
            magnitudes, distances, gmpe_list, imts, self.params,
            figure_size=(7,5),
            filename=os.path.join(TMP_PATH,
                                  "magnitude_sigma_imt_trellis_simple.pdf"),
            filetype="pdf",
            dpi=150)
        # Tests export
        mag_imt_trellis.pretty_print(
            os.path.join(TMP_PATH, "magnitude_sigma_imt_trellis1.csv"),
                         sep=",")

    def tearDown(self):
        """
        Removes the temporary directory
        """
        os.system("rm {:s}".format(TMP_PATH))


