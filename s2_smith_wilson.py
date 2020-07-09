#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports
import numpy as np
import pandas as pd
from math import exp, log
from scipy.optimize import minimize


# In[ ]:


class s2:
    
    def __init__(self, input_rates, input_liquid, ufr, alpha=None, min_convergence_year=60, rate_type="zero"):
        """
        Assumptions
        --------------
        -CRA: subtracted before rates are inserted
        -VA: added before rates are inserted
        -n_coupons: currently only annual coupon paying instruments are supported!
        -zero_coupon rates are all assumed to be liquid up to LLP 
        -input_rates and input_liquid should match in length and 
        
        Inputs
        --------------
        input_rates: array
            Interest rates where position matches maturity in years (e.g. index position 0 is 1y, index position 1 is 2y...)
        input_liquid: array
            Boolean indicator whether respective rate is considered liquid
        ufr: float
            Ultimate forward rate for extrapolation
        alpha: float
            speed of convergence
        min_convergence_year: int
            convergence_point = max(min_convergence_year, last_liquid_point + 40)
        rate_type: string
            Either "zero" or "swap" depending on rate input
        """
        
        #Input checks
        assert len(input_rates) == len(input_liquid)
        assert rate_type in ["zero", "swap"]
        
        self.ufr = ufr
        self.lnUfr = log(1+ufr)
        self.rate_type = rate_type
        self.input_rates = input_rates
        self.input_liquid = input_liquid
        self.last_liquid_point = np.max(np.where(self.input_liquid==1)) + 1
        self.liquid_rates = self.input_rates[np.where(self.input_liquid==1)]
        self.liquid_maturities = np.where(self.input_liquid==1)[0] + 1
        self.convergence_point = max(min_convergence_year, self.last_liquid_point + 40)
        self.tau = 0.0001 #1bps
        
        #Dimensions of matrices
        self.M = int(self.last_liquid_point)
        self.N = len(self.liquid_maturities)
        
        #If no alpha is provided, it is optimized
        if alpha is not None:
            self.alpha = np.array(alpha)
        else:
            self.optimize_alpha()
            
    def X_matrix():
        """
        N x M matrix containing cash flows of N observable instruments and M payment dates.
        For zero rates where it is assumed that all rates up to the LLP are liquid this simplifies to N x N matrix
        """
        if self.rate_type == "zero":
            X = np.diag(np.repeat(1, self.N))
        else:
            X = np.zeros([self.N, self.M])
            for n in range(0, self.N):
                maturity_in_years = self.liquid_maturities[n]
                rate = self.liquid_rate[n]
                for m in range(0, maturity_in_years - 1):
                    X[n, m] = rate
                X[n, maturity_in_years - 1] = 1.0 + rate
        return X
        
    def t(self, j):
        """
        Generic function returns time
        """
        return j
        
    def t_vector(self):
        """
        M x 1 column vector containing maturities in years
        """
        return np.array([self.t(j) for j in range(1, self.M + 1)])
    
    def t_observed(self, n):
        """
        Returns maturity in years for n-th liquid instrument (1 to N)
        """
        return self.liquid_maturities[n-1]
    
    def t_vector_observed(self):
        """
        N x 1 column vector with observed maturities of liquid instruments in years
        """
        return np.array([self.t_observed(j) for j in range(1, self.N + 1)])
    
    def r_observed(self, n):
        """
        Returns interest rate for n-th liquid instrument (1 to N)
        """
        return self.liquid_rate[n-1]
        
    def wilson_function(self, i, j):
        """
        Wilson function -- essential for extrapolation
        """
        t = self.t(i)
        uj = self.t(j)
        return exp(-self.lnUfr * (t + uj)) * (self.alpha * min(t, uj) - 0.5 * exp(-self.alpha * max(t, uj)) * (exp(self.alpha * min(t, uj)) - exp(-self.alpha * min(t, uj))))
        
    def W_matrix(self):
        """
        M x M matrix
        """
        W_matrix = np.array([[self.wilson_function(i, j) for j in range(1, self.M + 1)] for i in range(1, self.M + 1)])
        
        #Optimization leads to strange shape -- need for further investigation
        return W_matrix.reshape(self.M, self.M)
    
    def mu(self, i):
        """
        Ultimate forward rate discount factors
        """
        return exp(-self.lnUfr * self.t(i))
    
    def mu_vector(self):
        """
        M x 1 column vector with ufrs discount factors
        """
        return np.array([self.mu(i) for i in range(1, self.M + 1)])
    
    def Q_matrix(self):
        """
        M x N aux matrix
        """
        return np.matmul(np.diag(self.mu_vector()), self.X_matrix().T)
    
    def m(self, n):
        """
        Observed zero coupon / swap prices
        """
        if self.rate_type == "zero":
            return (1 + self.r_observed(n) ** (-self.t_observed))
        else:
            #Par swap rates by construction have a value of 1.0 units
            return 1.0
        
    def m_vector(self):
        """
        N x 1 column vector with bond / swap prices
        """
        return np.array([self.m(i) for i in range(1, self.N + 1)])
    
    def zeta_vector(self):
        """
        N x 1 column vector with zeta
        """
        left_matrix = np.linalg.inv(np.matmul(self.X_matrix(), np.matmul(self.W_matrix(), self.X_matrix().T)))
        right_matrix = self.m_vector() - np.matmul(self.X_matrix(), self.mu_vector())
        return np.matmul(left_matrix, right_matrix)
    
    def zeta(self, i):
        """
        Zeta for asset
        """
        return self.zeta_vector()[i-1]
    
    def quasi_constant_k(self):
        Q = self.Q_matrix()
        zeta_vec = self.zeta_vector()
        t_vec = self.t_vector()
        alpha = self.alpha
        return (1+alpha * np.matmul(t_vec.T, np.matmul(Q, zeta_vec))) / (np.matmul(np.sinh(alpha*t_vec.T), np.matmul(Q, zeta_vec)))
    
    def g_alpha(self):
        return self.alpha / abs(1 - self.quasi_constant_k() * exp(self.alpha * self.convergence_point))
    
    def error_function(self, alpha):
        """
        Minimize alpha given two constraints:
        1) alpha >= 0.05
        2) g_alpha() <= tau (certain precision)
        """
        self.alpha = np.array(alpha)
        return alpha
    
    def constraint_precision(self, alpha):
        """
        g_alpha() <= tau (certain precision)
        """
        self.alpha = np.array(alpha)
        return self.tau - self.g_alpha()
    
    def optimize_alpha(self):
        """
        Minimize alpha given two constraints:
        1) alpha >= 0.05
        2) g_alpha() <= tau (certain precision)
        """
        
        bounds = (0.05, None)
        constraints = ({"type": "ineq", "fun": self.constraint_precision}) #>=0
        
        optimize_result = minimize(fun=self.error_function,
                                  x0=np.array(0.2),
                                  method="SLSQP",
                                  constraints=constraints,
                                  bounds=bounds,
                                  tol=0.0001,
                                  options={"disp":False})
        optimized_alpha = optimize_result.x
        self.alpha = optimized_alpha
        
    def cash_flow(self, i, j):
        """
        Returns cash flow of i-th asset (1 to N) and j-th payment date (1 to M) from X_matrix
        """
        X = self.X_matrix()
        return X[i-1, j-1]
    
    def pricing(self, t):
        """
        Pricing function for zero coupon bond with maturity t in years
        """
        if self.rate_type == "zero":
            return self.mu(t) + sum(self.zeta(j) * self.wilson_function(t, j) for j in range(1, self.N +1))
        else:
            total_sum = 0.0
            for i in range(1, self.N + 1):
                m_sum = 0.0
                zeta_sum = self.zeta(i)
                for j in range(1, self.M + 1):
                    m_sum = m_sum + self.cash_flow(i, j) * self.wilson_function(t, self.t(j))
                total_sum = total_sum + zeta_sum * m_sum
            return total_sum + self.mu(t)
    
    def zero_rate(self, t):
        """
        Returns extrapolated zero rate for maturity t in years
        """
        return (1 / self.pricing(t)) ** (1 / self.t(t)) - 1
    
    def zero_curve(self, start=1, end=120):
        """
        Returns extrapolated zero curve
        """
        return np.array([self.zero_rate(year) for year in range(start, end + 1)])
    
    def forward_rate(self, t):
        """
        Returns extrapolated forward rate for maturity t in years
        """
        return self.pricing(t-1) / self.pricing(t) - 1
    
    def forward_curve(self, start=1, end=120):
        """
        Returns extrapolated forward curve
        """
        return np.array([self.forward_rate(year) for year in range(start, end + 1)])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




