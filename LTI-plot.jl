# Plot the amplitude describing function of an LTI transfer function
# Author: Tom Chaffey
# Date: 28/06/2022
#

using Printf
using LinearAlgebra
using Optim
using Plots
using FFTW
using CSV

# returns a square wave, amplitude α, period T, phase τ (in seconds), n samples
function square(a, T, τ, n)
    s = vcat(a*ones(Int.(floor.(n/2)), 1), -a*ones(Int.(ceil.(n/2)), 1))
    return circshift(s, Int.(floor.(τ*n/T)))
end

function limit(x, l, u)
    if (round(x) < l)
        return Int(l)
    elseif (round(x) > u)
        return Int(u)
    else
        return Int(round(x))
    end
end
# gradient of the objective function sum(w - s)^2
function square_gradient!(G, a, T, τ, n, w)
    if (τ < T/2)
        G[1] = (-a - w[limit(τ*n/T, 1, length(w))])^2 - (a - w[limit(τ*n/T, 1, length(w))])^2 - (-a - w[limit((τ + T/2)*n/T, 1, length(w))])^2 + (a - w[limit((τ + T/2)*n/T, 1, length(w))])^2
        G[2] = sum(-2 .*(-a .- w[1:limit(τ*n/T, 1, length(w))])) + sum(2 .*(a .- w[limit(τ*n/T, 1, length(w)):limit((τ+ T/2)*n/T, 1, length(w))])) + sum(-2 .*(-a .- w[limit((τ+ T/2)*n/T, 1, length(w)):end])) 
    else
        G[1] = (-a - w[limit(τ*n/T, 1, length(w))])^2 - (a - w[limit(τ*n/T, 1, length(w))])^2 - (-a - w[limit((τ - T/2)*n/T, 1, length(w))])^2 + (a - w[limit((τ - T/2)*n/T, 1, length(w))])^2
        G[2] = sum(2 .*(a .- w[1:limit((τ - T/2)*n/T, 1, length(w))])) + sum(-2 .*(-a .- w[limit((τ - T/2)*n/T, 1, length(w)):limit(τ*n/T, 1, length(w))])) + sum(2 .*(a .- w[limit(τ*n/T, 1, length(w)):end])) 
    end
end

# apply G to the T-periodic signal u.  Doesn't work for improper transfer functions,
# due to unbounded Fourier coefficients.
function steady_state(u, T, G)

    z = rfft(u)
    w = im*zeros(size(z))
    N = length(z)
    freqs = rfftfreq(N, length(u)/T)
    for n = 1:length(freqs)
        # coeff = G(im*2*pi*(n-1)*T/N) 
        coeff = 2*G(im*2*π*freqs[n]) # factor of two adjusts for negative frequencies
        #if (freqs[n] ≈ 0.0) # && G(0) ≈ NaN)
        #    w[n] = 0.0 + 0.0*im
        #else
            w[n] = coeff*z[n]
        #end
        #@printf("ω: %f G(jω): %f + j*%f, w[n]: %f + j%f\n", freqs[n], real(coeff), imag(coeff), real(w[n]), imag(w[n]))
    end
    return irfft(w, 1)
end

# compute the steady state response of an improper transfer function via a resolvent iteration
function improper_steady_state(u, T, R, ϵ = 0.01)
    y_old = steady_state(u, T, R)
    while (true)
        y_new = steady_state(y_old, T, R)
        error = maximum(abs.(y_old - y_new)) 
        @printf("Error: %f\n", error)
        if (error < ϵ)
            break
        end
        y_old = y_new
    end
end

# fit a square wave to w in the variables τ and a using interior point method
function opt_square(w, T::Float64, n::Int, x0 = [T/2, T/2])
    multiplied = false
    if (maximum(w) < 0.1)
        w = 1000.0 .*w
        multiplied = true
    end
    objective(x) = norm(w - square(x[2], T, x[1], n)) 
    g!(G, x) = square_gradient!(G, x[2], T, x[1], n, w)
    lower = [0.0, (T < 0.1 ? 0.0 : 0.01)]
    upper = [T, Inf]
    inner_optimizer = BFGS()
    result = optimize(objective, g!, lower, upper, [T/2, T/2], Fminbox(inner_optimizer), Optim.Options(
                             iterations = 10000,
                             g_tol=1e-12))#, lower, upper)
    if (!Optim.converged(result))
        display(result)
    end
    τ, a = Optim.minimizer(result)
    if (multiplied)
        a = a/1000.0
    end
    return [τ, a], Optim.converged(result)
end

# plots the amplitude Nyquist diagram using adaptive period spacing
function amplitude_nyquist_adaptive(G; start = 0.1, step = 0.1, stop = 100.0)
        n = 2^10 # samples in square wave
        m = 50 # samples in Nyquist curve

        # generate a Haar wavelet input signal
        
        u = hcat(1.0 .*ones(1, Int(floor(n/2))), -1.0 .*ones(1, Int(ceil(n/2))))

        points = []
        Ts = []
        point = zeros(2, 1)
        last_point = zeros(2, 1)
        # to-do: automatically estimate bandwidth
        xold = [0.05, 0.01] # for warm start
        p = plot()
        m = 0
        T = start
        while T < stop
            m = m + 1
            # apply G to u
                y = real.(steady_state(u, T, G))
                
                vals, success = opt_square(y', T, n, xold)
                τ, a = vals
                @printf("%9s T = %f τ = %f ϕ = %f a = %f\n", success ? "converged" : "failed", T, τ, 2*π*τ/T, a)

                point[1] = a*cos(2*π*τ/T)
                point[2] = a*sin(2*π*τ/T)
                push!(Ts, T)
                if (success)
                    xold = [τ, a]
                    display(scatter!(p, [point[1]], [point[2]], legend=false, markersize=2.0, color=:blue, aspect_ratio = 1.0))
                    #display(scatter!(p, [points[m, 1]], [points[m, 2]], legend=false, markersize=2.0, color=:blue, aspect_ratio=1.0))
                        push!(points, copy(point))
                        display(points)
                        push!(Ts, T)
                end
                if (norm(point - last_point) < 0.1)
                    step = step * 1.5
                    @printf("shift of %f, increasing step\n", norm(point - last_point))
                elseif (norm(point - last_point) > 0.5)
                    step = step / 2
                    @printf("shift of %f, halving step\n", norm(point - last_point))
                end
                last_point = copy(point)
                T = T + step
        end

        scatter(points)
        # plot(points[:, 1], points[:, 2])
        CSV.write("amp_nyquist.csv", (re = [i[1] for i in points], jm = [i[2] for i in points], Ts))
        return points
        
end

# plot the amplitude nyquist diagram using the supplied range of periods
function amplitude_nyquist(G, samples = exp10.(range(-1, stop=1.5, length=50)))
        n = 2^10 # samples in square wave
        m = 50 # samples in Nyquist curve

        # generate a Haar wavelet input signal
        
        u = hcat(1.0 .*ones(1, Int(floor(n/2))), -1.0 .*ones(1, Int(ceil(n/2))))

        points = zeros(m, 2)
        # to-do: automatically estimate bandwidth
        xold = [0.05, 0.01] # for warm start
        p = plot()
        #samples = [0.1:0.1:1.0; 1.2:0.2:5.0; 5.02:0.02:6.2; 6.2:0.001:6.50; 6.502:0.02:6.80; 6.8:0.1:7.0; 8.0:1.0:12.0; 15.0:1.0:30.0]#
        samples = [0.1:0.1:1.0; 1.01:0.1:5.0; 5.02:0.02:6.2; 6.2:0.0001:6.50]#; 6.502:0.02:6.80; 6.8:0.01:7.0; 8.0:0.5:12.0; 15.0:1.0:30.0]#exp10.(range(-1, stop=1.5, length=m))

        for T in samples
            # apply G to u
                y = real.(steady_state(u, T, G))
                
                vals, success = opt_square(y', T, n)#, xold)
                τ, a = vals
                @printf("%9s T = %f τ = %f ϕ = %f a = %f\n", success ? "converged" : "failed", T, τ, 2*π*τ/T, a)

                if (success)
                    xold = [τ, a]
                    points[m, 1] = a*cos(2*π*τ/T)
                    points[m, 2] = a*sin(2*π*τ/T)
                    display(scatter!(p, [points[m, 1]], [points[m, 2]], legend=false, markersize=2.0, color=:blue, aspect_ratio=1.0))
                end
        end

#        plot(points[:, 1], points[:, 2])

end
# utility function to convert rad/s to period
function rad2T(ω)
        return 2*π/ω
end
# multiply the output of two LTI filters, G1 and G2
# Frequency of output square wave is double the frequency of the input square wave, due to product...
function product_example(T, G1, G2)
        n = 2^10
        u = circshift(hcat(1.0 .*ones(1, Int(floor(n/2))), -1.0 .*ones(1, Int(ceil(n/2))))', div(0, 2))'
        t = LinRange(0, T/2, div(n, 2))
        y1 = real.(steady_state(u, T, G1))
        y2 = real.(steady_state(u, T, G2))
        y = y1 .* y2 # multiply outputs
        half_y = y[1:div(length(y), 2)]
        vals, success = opt_square(half_y, T/2, div(n, 2), [0.0, 1.0])
        τ, a = vals
        z = square(a, T/2, τ, div(n, 2))
        @printf("%9s T = %f τ = %f ϕ = %f a = %f\n",success ? "converged" : "failed",  T, τ, 2*π*τ/T, a)

#¢        plot(t, u')
        plot(t, z)
        plot!(t, half_y)

end
# multiply the output of two LTI filters, G1 and G2. Plot Nyquist diagram
function product_example_Nyquist(G1, G2; start = 0.1, step = 0.1, stop = 100.0)
        n = 2^10
        u = hcat(1.0 .*ones(1, Int(floor(n/2))), -1.0 .*ones(1, Int(ceil(n/2))))

        xold = [0.05, 0.01] # for warm start
        p = plot()

        point = zeros(2, 1)
        last_point = zeros(2, 1)
        T = start
        while T < stop
            # apply G to u
                y1 = real.(steady_state(u, T, G1))
                y2 = real.(steady_state(u, T, G2))
                y = y1 .* y2 # multiply outputs
                half_y = y[1:div(length(y), 2)]
                vals, success = opt_square(half_y, T/2, div(n, 2), [0.0, 1.0])
                τ, a = vals
                
                @printf("%9s T = %f τ = %f ϕ = %f a = %f\n", success ? "converged" : "failed", T, τ, 2*π*τ/T, a)

                point[1] = a*cos(2*π*τ/T/2)
                point[2] = a*sin(2*π*τ/T/2)
                if (success)
                    xold = [τ, a]
                    display(scatter!(p, [point[1]], [point[2]], legend=false))
                end

                if (norm(point - last_point) < 0.5)
                    step = step * 1.5
                    @printf("shift of %f, increasing step\n", norm(point - last_point))
                elseif (norm(point - last_point) > 2)
                    step = step / 2
                    @printf("shift of %f, halving step\n", norm(point - last_point))
                end
                last_point = copy(point)
                T = T + step
        end
end

function run_test(G, T)
        n = 2^10

        # generate a Haar wavelet input signal
        
        u = circshift(hcat(1.0 .*ones(1, int(floor(n/2))), -1.0 .*ones(1, int(ceil(n/2))))', div(0, 2))'
        #u = circshift(vcat(1.0 .*ones(Int(floor(n/2)), 1), -1.0 .*ones(Int(ceil(n/2)), 1)), div(200, 2))
        t = LinRange(0, T, n)
        
        #u = cos.(2*pi*range(0, stop=1, length=n))
        
        # apply G to u
        
        y = real.(steady_state(u, T, G))
        
        vals, success = opt_square(y', T, n, [0.0, 1.0])
        τ, a = vals
        z = square(a, T, τ, n)
        @printf("%9s T = %f τ = %f ϕ = %f a = %f Re = %f Im = %f\n",success ? "converged" : "failed",  T, τ, 2*π*τ/T, a, a*cos(2*π*τ/T), a*sin(2*π*τ/T))

        plot(t, u', legend=false)
        plot!(t, z, legend= false)
        plot!(t, y', legend=false)

end

function example_1()

        T = 6.28005 # cross over period for α = 0.001, ω = 1
        α = 0.001
        ω = 1.0
        G(s) = 1/(s + 1)/(s^2 + α*s + ω)
        run_test(G, T)
end

function example_2()

        T = 2.05 # cross over period for α = 0.001, ω = 1
        G(s) = 1/(s^2 + 0.01*s + 1)/(s^2 + 0.00001*s + 9)/(s^2 + 0.0001*s + 36)
        run_test(G, T)
end
function example_3()

        T = 2.05 # cross over period for α = 0.001, ω = 1
        G(s) = 1/(s+1)/(s^2 + 0.001*s + 1)
        amplitude_nyquist(G)
end

function example_4()

        T = 2.05 # cross over period for α = 0.001, ω = 1
        G(s) = 1/s^2
        #amplitude_nyquist(G)
        run_test(G, T)
end

function FHN_example()
        a = 0.7
        b = 0.8
        τ = 12.5
        Iext = 0.5
        τ_f = 1.0
        τ_s = τ

        G(s) = -1/(τ_f*s + 1/(τ_s * s + b))

        #run_test(G, 2.0)
        amplitude_nyquist_adaptive(G, start = 0.1, stop = 1000.0, step = 0.1)
end

function FHN_example_2()
        a = 0.0
        b = 0.1
        τ = 0.100
        Iext = 1.5
        τ_f = 1.0
        τ_s = τ

        G(s) = -1/(τ_f*s + 1/(τ_s * s + b))

        run_test(G, 2.13)
        #amplitude_nyquist_adaptive(G, start = 0.1, stop = 5000.0, step = 0.1)
end

# not working yet.  Possibly need to use FPI in time domain, see monotone paper scripts.
function VDP_example_high_pass()
        μ = 0.100
        α = 0.05
        T = 2.13
        n = 100

        #R(s) = 1/(1 + α*(s^2 - μ*s + 1)/s)
        R(s) = 1/(1 + α*(s + 1))

        u = hcat(1.0 .*ones(1, Int(floor(n/2))), -1.0 .*ones(1, Int(ceil(n/2))))
        improper_steady_state(u, T, R) 
        #run_test(R, 2.13)
        #amplitude_nyquist_adaptive(G, start = 0.1, stop = 5000.0, step = 0.1)
end

function plot_comparison()
        adf = [0.4015, 0.807, 1.221, 1.648, 2.093]
        df  = [0.497, 1.013, 1.573, 2.220, 3.048]
        tsy = [0.4013, 0.811, 1.238, 1.695, 2.197]
        adf2= [0.807, 1.648, 2.563, 3.612, 4.89]
        ϵ   = [0.1, 0.2, 0.3, 0.4, 0.5]
        plot(ϵ, [adf, df, tsy, adf2], 
             label = ["adf" "df" "Tsypkin" "adf2"], 
             line=(1, [:solid :dash :dot :dashdot]), 
             legend = :topleft, 
             xlabel = "ϵ", 
             ylabel = "period")
end

function hysteresis_example()
        G(s) = 1/(s+1)
        run_test(G, 0.4015) # a = 0.050005
        run_test(G, 0.807) # a = 0.100003
        run_test(G, 1.221) # a = 0.15009
        run_test(G, 1.648) # a = 0.200032
        run_test(G, 2.093) #  a = 0.250001
        run_test(G, 5.0)
        example_4()
        
        
        run_test(G, 0.807) # a = 0.100003
        run_test(G, 1.648) # a = 0.200032
        run_test(G, 2.563) #  a = 0.300001
        run_test(G, 3.612) #  a = 0.400084
        run_test(G, 4.890) #  a = 0.500063
        
        plot_comparison()
end

#VDP_example_high_pass()
G_1(s) = 1/(s^2 + 1.0*s + 1)
run_test(G_1, 1.0)
#G_2(s) = 1/(s^2 + 0.1*s + 9)
#ω = 1.0
#T = rad2T(ω)
#product_example_Nyquist( G_1, G_2)

#G(s, d, ϵ) = 1/(s^4 + 1 + 1) - ϵ/d
#amplitude_nyquist_adaptive(G, start = 0.001, stop = 10.0, step = 0.001)
#run_test(s -> G(s, 1.0, 0.1), 1.877) # Re is zero

#run_test(s -> G(s, 1.0, 0.1), 0.4015) # Im is zero.  Agrees with Hysteresis example.  Except factor of 2 is gone????????
#run_test(s -> G(s, 1.0, 0.2), 0.811) # a = 0.100003
#run_test(s -> G(s, 1.0, 0.3), 1.238) # a = 0.15009
#run_test(s -> G(s, 1.0, 0.4), 1.695) # a = 0.200032
#run_test(s -> G(s, 1.0, 0.5), 2.197) #  a = 0.250001


#G(s) = (s^2 - s + 1)/s
#α = 0.01
#G(s) = 1/(s + 1)/(s^2 + α*s + 1)
#G(s) = 1/(s^2 + 0.01*s + 1)/(s^2 + 0.0001*s + 9)/(s^2 + 0.0001*s + 36)
#amplitude_nyquist(G)
