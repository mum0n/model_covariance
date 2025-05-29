
function linear_regression_simple(x::AbstractVector{T}, y::AbstractVector{T}; method="qr", toget="betas" ) where {T<:AbstractFloat} 
    
    res = Beta = "Method not found"

    (N = length(x)) == length(y) || throw(DimensionMismatch())
      
    if method=="normal_equations"
      X = [ones(N) x]
      # Beta = (X' * X)^(-1) * X' * y
      # Beta = (X'X)\(X'y) # direct
      Beta = (X'X)\(X'y) 
    end
  
    if method=="factorize"
      X = [ones(N) x]
      F = Hermitian(X'X) |> factorize
      Beta = F \ (X'y) # Solves a linear equation system Ax = b; A=X, b=y
    end
  
    if method=="svd"
      X = [ones(N) x]
      Beta = LinearAlgebra.pinv(X) * y
    end
  
    if method=="qr"
      X = [ones(N) x]
      Beta = X\y
    end
  
    if method=="cholesky"
      X = [ones(N) x]
      Beta = ldiv!(cholesky!(Symmetric(X'X, :U)), X'y)
    end
   
    if method =="glm"
      # using DataFrames, GLM, StatsBase
      X = [ones(N) x]
      res = lm( X, y )
      Beta = coef(res)
    end
  
    if method=="sums_of_squares"
      # old-fashioned technique with sums of squares ...

      Xmean = sum(x)/N;
      Ymean = sum(y)/N;

      SSX = sum( (x .- Xmean).^2 )
      SSY = sum( (y .- Ymean).^2 )
      SXY = sum( (x .- Xmean) .* (y .- Ymean ) )
      
      m = SXY / SSX  # == cov(x,y) / var(x)

      # intercept passes through centroid: 
      b = Ymean - m * Xmean
  
      Beta = [b, m] 
 
      SSR = SXY^2 / SSX  # of regression == cov(x,y)^2 / var(x)
      SSE = SSY - SSR
  
      r2 = ( SSY - SSE ) / SSY
      r = SXY / sqrt( SSX * SSY )
      # r == sqrt(r2) == 1/(N-1) * sum( (x .- Xmean ) ./ std(x)  .*  (y .- Ymean ) ./ std(y) )
      # m = r * std(y) / std(x)  # when x, y are standardized, m=r

      SDX = sqrt(SSX/(N-1));
      SDY = sqrt(SSY/(N-1));
      
      m_sd = SDY / sqrt( SSX )  # standard error of slope
      b_sd = SDY * sqrt( sum(x.^2 ) / (N*SSX) )
  
      r2adjusted = 1.0 - (SSE/ (N-2 ) ) / (SSY / (N-1)) 
      
      res = ( Beta=Beta, Beta_sd=[b_sd, m_sd], r=r, r2=r2, r2adjusted=r2adjusted )

    end
    
 
    if method=="MAR_york_1966"

        # copied/modified from Edward T Peltzer, MBARI, 2016 Mar 17.
        # who based it on equations in:
        #  York (1966) Canad. J. Phys. 44: 1079-1086;
        #  Kermack & Haldane (1950) Biometrika 37: 30-41;
        #  Pearson (1901) Phil. Mag. V2(6): 559-572.
        
        # X, y must be scaled and line passes through the centroid 
      
        Xmean = sum(x)/N;
        Ymean = sum(y)/N;
   
        SXY = sum( (x .- Xmean) .* (y .- Ymean));
        SSX = sum( (x .- Xmean) .^2);
        SSY = sum( (y .- Ymean) .^2);
       
        m = (SSY - SSX + sqrt(((SSY - SSX)^2) + (4 * SXY^2)))/(2 * SXY);
        b = Ymean - m * Xmean;
  
        Beta = [b, m] 

        SDX = sqrt(SSX/(N-1));
        SDY = sqrt(SSY/(N-1));
        
        r = SXY / sqrt(SSX * SSY);
          
        m_sd = (m/r) * sqrt((1 - r^2)/N);
        sb1 = (SDY - SDX * m)^2;
        sb2 = (2 * SDX * SDY) + ((Xmean^2 * m * (1 + r))/r^2);
        b_sd = sqrt((sb1 + ((1 - r) * m * sb2))/N);
        
        res = (Beta=Beta, Beta_sd=[b_sd, m_sd], r=r, r2=r^2 )
    end


    if method=="MAR_geometric_mean"
        # copied/modified from Edward T Peltzer, MBARI, 2016 Mar 17.
        #     This line is called the GEOMETRIC MEAN or the REDUCED MAJOR AXIS.
        #     See Ricker (1973) Linear regressions in Fishery Research, J. Fish.
        #       Res. Board Can. 30: 409-434, for the derivation of the geometric
        #       mean regression.
         
        #     Since no statistical treatment exists for the estimation of the
        #       asymmetrical uncertainty limits for the geometric mean slope,
        #       I have used the symmetrical limits for a model I regression
        #       following Ricker's (1973) treatment.  For ease of computation,
        #       equations from Bevington and Robinson (1992) "Data Reduction and
        #       Error Analysis for the Physical Sciences, 2nd Ed."  pp: 104, and
        #       108-109, were used to calculate the symmetrical limits: m_sd and b_sd.


        # Determine slope of Y-on-X regression
        xyreg = linear_regression_simple(x, y, method="qr")
        my = xyreg[2]

        # Determine slope of X-on-Y regression
        yxreg = linear_regression_simple(y, x, method="qr")
        mx = 1 / yxreg[2]  # inverse
 
        
        # Calculate geometric mean slope
        if sign(my) != sign(mx) 
          return "Sign not consistent"
        end
       
        m = sqrt(abs(my * mx));  # note both must be the same sign here

        if (my < 0) && (mx < 0)
            m = -m;
        end

        Xmean = sum(x)/N;
        Ymean = sum(y)/N;
        b = Ymean - m * Xmean; # intercept

        Beta = [b, m] 
         
        Sxy = sum(x .* y);
        SSX = sum(x .^ 2);
         
        den = N * SSX - sum(x)^2;     
          
        r = sqrt(my / mx);
        
        if (my < 0) && (mx < 0)
            r = -r;
        end
        
        residual = y .- b .- m .* x;
        
        s2 = sum(residual .* residual) / (N-2);
        m_sd = sqrt(N * s2 / den);
        b_sd = sqrt(SSX * s2 / den);

        res = (Beta=Beta, Beta_sd=[b_sd, m_sd], r=r, r2=r^2 )

    end


    if method=="MAR_geometric_mean_untransposed"
      # copied/modified from Edward T Peltzer, MBARI, 2016 Mar 17.
      #     This line is called the GEOMETRIC MEAN or the REDUCED MAJOR AXIS.
      #     See Ricker (1973) Linear regressions in Fishery Research, J. Fish.
      #       Res. Board Can. 30: 409-434, for the derivation of the geometric
      #       mean regression.
       
      #     Since no statistical treatment exists for the estimation of the
      #       asymmetrical uncertainty limits for the geometric mean slope,
      #       I have used the symmetrical limits for a model I regression
      #       following Ricker's (1973) treatment.  For ease of computation,
      #       equations from Bevington and Robinson (1992) "Data Reduction and
      #       Error Analysis for the Physical Sciences, 2nd Ed."  pp: 104, and
      #       108-109, were used to calculate the symmetrical limits: m_sd and b_sd.


      # Determine slope of Y-on-X regression
      xyreg = linear_regression_simple(x, y, method="qr")
      my = xyreg[2]

      # Determine slope of X-on-Y regression
      yxreg = linear_regression_simple(y, x, method="qr")
      mx = yxreg[2]  # inverse

      
      # Calculate geometric mean slope
      if sign(my) != sign(mx) 
        return "Sign not consistent"
      end
     
      m = sqrt(abs(my * mx));  # note both must be the same sign here

      if (my < 0) && (mx < 0)
          m = -m;
      end

      Xmean = sum(x)/N;
      Ymean = sum(y)/N;
      b = Ymean - m * Xmean; # intercept

      Beta = [b, m] 
       
      Sxy = sum(x .* y);
      SSX = sum(x .^ 2);
       
      den = N * SSX - sum(x)^2;     
        
      r = sqrt(my / mx);
      
      if (my < 0) && (mx < 0)
          r = -r;
      end
      
      residual = y .- b .- m .* x;
      
      s2 = sum(residual .* residual) / (N-2);
      m_sd = sqrt(N * s2 / den);
      b_sd = sqrt(SSX * s2 / den);

      res = (Beta=Beta, Beta_sd=[b_sd, m_sd], r=r, r2=r^2 )

    end


    if method=="MAR_cubic" 
      
      # this method is not complete

      # copied/modified from Edward T Peltzer, MBARI, 2016 Mar 17.

      #  m is determined by finding the roots to the cubic equation:
      #  m^3 + P * m^2 + Q * m + R = 0
      #  Eqs for P, Q and R are from York (1966) Canad. J. Phys. 44: 1079-1086.
      #  [m,b,r,m_sd,b_sd,xc,yc,ct] = lsqcubic(X,Y,sX,sY,tl)
    
      # X    =    x data (vector)
      # Y    =    y data (vector)
      # sX   =    uncertainty of x data (vector)
      # sY   =    uncertainty of y data (vector)
      #
      # tl   =    test limit for difference between slope iterations  
      #
      # m    =    slope
      # b    =    y-intercept
      # r    =    weighted correlation coefficient
      #
      # m_sd   =    standard deviation of the slope
      # b_sd   =    standard deviation of the y-intercept
      #
      # xc   =    WEIGHTED mean of x values
      # yc   =    WEIGHTED mean of y values
      #
      # ct   =    count: number of iterations
      #
      #     Notes:  1.  (xc,yc) is the WEIGHTED centroid.
      # 2.  Iteration of slope continues until successive differences
      #   are less than the user-set limit "tl".  Smaller values of
      #   tl require more iterations to find the slope.
      # 3.  Suggested values of tl = 1e-4 to 1e-6.
      # 
        # function [m,b,r,m_sd,b_sd,xc,yc,ct]=lsqcubic(X,Y,sX,sY,tl)
        tl = 1e-4

        sX = std(x)
        sY = std(y)
        wX = 1 ./ (sX .^ 2);
        wY = 1 ./ (sY .^ 2);
        
        # Set-up a few initial conditions:
        
        ct = 0;
        ML = 1;
                
        # ESTIMATE the slope by calculating the major axis according
        #   to Pearson's (1901) derivation, see: lsqfitma.m
                        # Determine slope of Y-on-X regression
        # Determine slope of Y-on-X regression
        xyreg = linear_regression_simple(x, y, method="qr")
        my = xyreg[2]

        # Determine slope of X-on-Y regression
        yxreg = linear_regression_simple(y, x, method="qr")
        mx = 1 / yxreg[2]  # inverse
 
        # Calculate geometric mean slope
        if sign(my) != sign(mx) 
          Beta = "Sign not consistent"
        end
       
        m = sqrt(abs(my * mx));  # note both must be the same sign here

        if (my < 0) && (mx < 0)
            m = -m;
        end

        test = abs((ML - m) / ML);
                
        # Calculate the least-squares-cubic
                
        # Make iterative calculations until the relative difference is
        #   less than the test conditions
                
        while test > tl
         
          # Calculate sums and other re-used expressions:
    
          MC2 = m ^ 2;
          W = (wX .* wY) ./ ((MC2 .* wY) + wX);
          W2 = W .^ 2;
    
          SW = sum(W);
          xc = (sum(W .* x)) / SW;
          yc = (sum(W .* y)) / SW;
    
          U = x .- xc;
          V = y .- yc;
    
          U2 = U .^ 2;
          V2 = V .^ 2;
    
          SW2U2wX = sum(W2 .* U2 ./ wX);
    
          # Calculate coefficients for least-squares cubic:
         
          P = -2 * sum(W2 .* U .* V ./ wX) / SW2U2wX;
          Q = (sum(W2 .* V2 ./ wX) - sum(W .* U2)) / SW2U2wX;
          R = sum(W .* U .* V) / SW2U2wX;
    
          # Find the roots to the least-squares cubic:
          
          ## Conversion incomplete from here on

          LSC = [1 P Q R];


          MR = find_zero(LSC);
    
          # Find the root closest to the slope:
    
          DIF = abs(MR - m);
          _, Index = min(DIF);
    
          ML = m;
          m = MR(Index);
          test = abs((ML - m) / ML);
          ct = ct + 1;
    
        end
         
        # Calculate m, b, r, m_sd, and b_sd
          
        b = yc - m * xc;
     
        Beta = [b, m] 
        r = sum(U .* V) / sqrt(sum(U2) * sum(V2));
         
        sm2 = (1 / (N - 2)) * (sum(W .* (((m * U) - V) .^ 2)) / sum(W .* U2));
        m_sd = sqrt(sm2);
        b_sd = sqrt(sm2 * (sum(W .* (x .^ 2)) / SW));

        res = (Beta=Beta, Beta_sd=[b_sd, m_sd], r=r, r2=r^2 )

    end


    if method=="MAR_bisection"
        # copied/modified from Edward T Peltzer, MBARI, 2016 Mar 17.

        #     The SLOPE of the line is determined by calculating the slope of the line
        #       that bisects the minor angle between the regression of Y-on-X and X-on-Y.
        #
        #     The equation of the line is:     y = mx + b.
        #
        #     This line is called the LEAST SQUARES BISECTOR.
        #
        #     See: Sprent and Dolby (1980). The Geometric Mean Functional Relationship.
        #       Biometrics 36: 547-550, for the rationale behind this regression.
        #
        #     Sprent and Dolby (1980) did not present a statistical treatment for the
        #       estimation of the uncertainty limits for the least squares bisector
        #       slope, or intercept.
        #
        #     I have used the symmetrical limits for a model I regression following
        #       Ricker's (1973) treatment.  For ease of computation, equations from
        #       Bevington and Robinson (1992) "Data Reduction and Error Analysis for
        #       the Physical Sciences, 2nd Ed."  pp: 104, and 108-109, were used to
        #       calculate the symmetrical limits: m_sd and b_sd.
        #
            
        # Determine slope of Y-on-X regression
        xyreg = linear_regression_simple(x, y, method="qr")
        my = xyreg[2]

        # Determine slope of X-on-Y regression
        yxreg = linear_regression_simple(y, x, method="qr")
        mx = 1 / yxreg[2]  # inverse
         
        # Calculate the least squares bisector slope
        theta = (atan(my) + atan(mx)) / 2;
        m = tan(theta);
           
        # Calculate sums and means
        Xmean = sum(x)/N;
        Ymean = sum(y)/N;
        
        # Calculate the least squares bisector intercept
        
        b = Ymean - m * Xmean;

        Beta = [b, m] 
        SSX = sum(x.^2);
        
        # Calculate re-used expressions
        
        den = N * SSX - sum(x)^2;          
        # Calculate r, m_sd, b_sd and s2
        
        r = sqrt(my / mx);
        
        if (my < 0) && (mx < 0)
            r = -r;
        end
        
        residual = y .- b .- m .* x;
        
        s2 = sum(residual .* residual) / (N-2);
        m_sd = sqrt(N * s2 / den);
        b_sd = sqrt(SSX * s2 / den);

        res = (Beta=Beta, Beta_sd=[b_sd, m_sd], r=r, r2=r^2 )

    end
 

    if toget=="betas"
      return Beta
    else 
      self_contained = ("sums_of_squares", "glm", "MAR_bisection", "MAR_york_1966", "MAR_cubic", "MAR_geometric_mean"  )

      if any( any( x -> occursin(method, x), self_contained) )
        return res

      else

        Xmean = sum(x)/N;
        Ymean = sum(y)/N;

        SSX = sum( (x .- Xmean).^2 )
        SSY = sum( (y .- Ymean).^2 )
        SXY = sum( (x .- Xmean) .* (y .- Ymean ) )
        # SXY = cov(x, y)
        
        SSR = SXY^2 / SSX  # of regression == cov(x,y)^2 / var(x)
        SSE = SSY - SSR
    
        r2 = ( SSY - SSE ) / SSY
        r = SXY / sqrt( SSX * SSY )
        # r == sqrt(r2) == 1/(N-1) * sum( (x .- Xmean ) ./ std(x)  .*  (y .- Ymean ) ./ std(y) )
        # m = r * std(y) / std(x)  # when x, y are standardized, m=r
  
        SDX = sqrt(SSX/(N-1));
        SDY = sqrt(SSY/(N-1));
        
        m_sd = SDY / sqrt( SSX )  # standard error of slope
        b_sd = SDY * sqrt(  sum(x.^2 ) / (N*SSX) )
 
        r2adjusted = 1.0 - (SSE/ (N-2 ) ) / (SSY / (N-1)) 
        
        res = ( Beta=Beta, Beta_sd=[b_sd, m_sd], r=r, r2=r2, r2adjusted=r2adjusted )
         
        return res

      end

    end
 
  end



  function random_correlation_matrix(d=3, eta=1)
   
    # etas = [1 10 100 1000 1e+4 1e+5];
    # d = size of matrix
  
   # EXTENDED ONION METHOD to generate random correlation matrices
   # distributed ~ det(S)^eta [or maybe det(S)^(eta-1), not sure]
   # https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices
   
   # LKJ modify this method slightly, in order to be able to sample correlation matrices C from a distribution proportional to [detC]η−1. The larger the η, the larger will be the determinant, meaning that generated correlation matrices will more and more approach the identity matrix. The value η=1 corresponds to uniform distribution. On the figure below the matrices are generated with η=1,10,100,1000,10000,100000. 
 
     beta = eta + (d-2)/2;
     u = rand( Beta(beta, beta) );
     r12 = 2*u - 1;
     S = [1 r12; r12 1];  
 
     for k = 3:d
         beta = beta - 1/2;
         y = rand( Beta((k-1)/2, beta) );  # sample from beta
         r = sqrt(y);
         theta = randn(k-1,1);
         theta = theta/norm(theta);
         w = r*theta;
         U, E = eigen(S);
         U = hcat(U)
         R = U' * sqrt(E) * U; # R is a square root of S
         q = R[].re * w;
         S = [S q; q' 1];
     end
     return S
 end
 