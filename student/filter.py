# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dim_state = params.dim_state # process model dimension
        self.dt = params.dt # time increment
        self.q=params.q # process noise variable for Kalman


    # Implement the F() and Q() functions to calculate a system matrix 
    # for constant velocity process model in 3D and the corresponding 
    # process noise covariance depending on the current timestep dt. Note 
    # that in our case, dt is fixed and you should load it from misc/params.py. 
    # However, in general, the timestep might vary. At the end of the 
    # prediction step, save the resulting x and P by calling the functions 
    # set_x() and set_P() that are already implemented in student/trackmanagement.py.
    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        dt = self.dt
        F = np.eye(self.dim_state)
        
        # Create an array of indices for the elements to update
        # Use advanced indexing to update the elements
        dt_idx = np.arange(self.dim_state//2, self.dim_state)
        F[dt_idx - self.dim_state//2, dt_idx] = self.dt

        return np.matrix(F)
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        #
        ############
        # process noise covariance Q
        q = self.q
        dt = self.dt
        q1 = ((dt**3)/3) * q 
        q2 = ((dt**2)/2) * q 
        q3 = dt * q
        
        half_dim_state = self.dim_state // 2

        # Have to do some ugly stuff to generalize the shape of Q!
        # Half the diag values are q1, the other half are q3
        # The off-diag values 2 indices away are q2 on either side
        diag = np.diag([q1] * (half_dim_state) + [q3] * (half_dim_state))
        
        off_diag = np.diag([q2] * (half_dim_state), half_dim_state) + \
                   np.diag([q2] * (half_dim_state), -half_dim_state)
        Q = diag + off_diag

        return np.matrix(Q)
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, 
        # save x and P in track
        ############

        # Taken from exercise code.
        F = self.F()
        x = F * track.x # state prediction
        P = F * track.P * F.transpose() + self.Q() # covariance prediction
        track.set_x(x)
        track.set_P(P)
        
        ############
        # END student code
        ############ 

    # Implement the update() function as well as the gamma() and S() functions for 
    # residual and residual covariance. You should call the functions get_hx and 
    # get_H that are already implemented in students/measurements.py to get the 
    # measurement function evaluated at the current state, h(x), and the Jacobian H. 
    #  Again, at the 
    # end of the update step, save the resulting x and P by calling the functions 
    # set_x() and set_P() that are already implemented in student/trackmanagement.py.

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        x = track.x
        P = track.P

        # update state and covariance with associated measurement
        H = meas.sensor.get_H(x) # measurement matrix
        gamma = self.gamma(track, meas)
        S = self.S(track, meas, H) # covariance of residual
        K = P * H.transpose() * np.linalg.inv(S) # Kalman gain
        x = x + K * gamma # state update
        I = np.identity(self.dim_state)
        P = (I - K * H) * P # covariance update
        track.set_x(x)
        track.set_P(P)
    
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        # Note that we have a linear measurement model for lidar, so h(x)=H*x for now. 
        # You should use h(x) nevertheless for the residual to have an EKF ready for 
        # the nonlinear camera measurement model you'll need in Step 4. 
        ############

        # gamma = meas.z - self.H() * track.x # residual
        gamma = meas.z - meas.sensor.get_hx(track.x) # residual

        return gamma
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        S = H * track.P * H.transpose() + meas.R # covariance of residual

        return S
        
        ############
        # END student code
        ############ 