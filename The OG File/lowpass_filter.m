
function I = lowpassFilter(I)
        sz = 15;                  % must be odd
        f0 = 0.9;                 % cutoff freq (1 = Nyquist)
        n  = 4;                   % steepness
        f  = (0:sz)/sz;
        R  = exp(-(f ./ f0).^n);
        B  = firpm(sz+1, f, R);
        H  = ftrans2(B);
        I  = filter2(H,I);
end

