using CSV, Plots, LinearAlgebra, FFTW, Meshes


# returns a square wave, amplitude α, period T, phase τ (in seconds), n samples
function square(a, T, τ, n)
    s = vcat(a*ones(Int.(floor.(n/2)), 1), -a*ones(Int.(ceil.(n/2)), 1))
    return circshift(s, Int.(floor.(τ*n/T)))
end

function sat(x)
    if abs(x) > 1
        return abs(x)/x
    else
        return x
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
    return vec(irfft(w, 1))
end

# Plots the input, output and approximated output used to calculate the adf, for the transfer function 
# G at period T.
function plot_approx(G, T)
    n = 1024
    u = square(1.0, T, 0.0, n)
    y = real.(steady_state(u', T, G))'
    shifts = 0:T/50:T/2
    c = Vector{Float64}(undef, length(shifts))
    j = 0
    for τ in shifts
        j = j + 1
        c[j] = hwt(y, 1, τ, n)
    end
    ν = argmax(abs.(c))
    α = c[ν]
    ν = shifts[ν]
    z = square(α, T, ν, n)
    t = LinRange(0, T, n)
    plot(t, u, legend=false)
    plot!(t, z, legend= false)
    plot!(t, y, legend=false)
end

# Solves for the value of the adf at period T, using n time samples.
function adfT(G, T, n = 1024)
    # check h_crossings first
    dt = T/n
    u = square(1.0/T, T, 0.0, n)
    y = real.(steady_state(u', T, G))
    h = h_crossings(y)
    max_b = 0.0
    max_h = 0.0
    max_t = 0.0
    for x in h
        τ = x*T/n
        sq = square(1.0/T, T, τ, n)
        b = sq⋅y*dt
        if abs(b) > max_b
            max_b = b
            max_h = x
            max_t = τ
        end
    end
    if max_b == 0.0 # check integral condition
        println("checking integral condition")
        h2 = Int(floor(n/2))
        yy = dt/2*sum(y)
        yt = 0.0
        eps = 1e-3
        h = 0
        for h = 1:h2
            yt = yt + dt*(y[h + h2] - y[h])
            if abs(yt - yy) < eps
                break
            end
        end
        τ = h*T/n
        sq = square(1.0/T, T, τ, n)
        b = sq⋅y*dt
        max_h = h
        max_t = τ
        max_b = b
    end
    return max_b, max_t, max_h, u, y
end

# Plot the nyqa for the saturated, amplitude-dependent delay from the final example.
function delay_nyqa(T, a)
    nyqa = []
    αs = 0.01:0.01:20

    for α in αs
        nyqa = [nyqa; ((sat(α)/α)*exp(-im*2*π*α/T))]
    end
    plot(real.(nyqa), imag.(nyqa), aspect_ratio=1)
    G(s) = a/(s+1)
    Gr, Gi, T = adf(G, 2048)

    # now find the intersections
    z1 = [(Gr[i], Gi[i]) for i = 1:length(Gr)]
    z2 = [(real(n), imag(n)) for n in nyqa]
    coords, coords2 = intersections(z1, z2)
    #plot!(Gr, Gi)
    display(T[coords])
    display(αs[coords2])
    
    #CSV.write("sat_delay_nyqa_2.csv", (x = real.(nyqa), y = imag.(nyqa)))
    scatter!(Gr[coords], Gi[coords], series_annotations=T[coords])
    #CSV.write("lag_nyqa.csv", (x = Gr, y = Gi))

end

# Finds the intersection points of two curves, specified by sequences of points z1, s2 in the plane.
function intersections(z1, z2)
    coords = []
    coords2 = []
    for i = 1:length(z1)-1
        s1 = Segment(z1[i], z1[i+1])
        for j = 1:length(z2)-1
            s2 = Segment(z2[j], z2[j+1])
            if s1 ∩ s2 != nothing
                display(s1 ∩ s2)
                push!(coords, i)
                push!(coords2, j)
            end
        end
    end
    return unique(coords), unique(coords2)
end

# Returns the adf of the transfer function G.  The vector Ts specifies the periods to sample, and n is the resolution of the steady state simulation in adfT.
function adf(G, n = 1024)
    nyqa = []
    #for T = 0.1:0.1:3
    Ts = [0.01:0.2:50; 50:2:1000]
    for T in Ts
        β, τ = adfT(G, T, n)
        ang = τ/T*2*π
        β = β*T
        nyqa = [nyqa; β*exp(-im*ang)]
    end
    #scatter(real.(nyqa), imag.(nyqa), aspect_ratio=:equal, markershape=:cross)
    return real.(nyqa), imag.(nyqa), Ts
end


# determine the nearest indices to (T/2)-periodic crossings in a discrete signal: 
# y(t) = y(t + T/2) (intersections with a horizontal line, T/2 apart).
function h_crossings(u::Vector{Float64}, eps=1e-1)
    zc = []
    t2 = Int(floor(length(u)/2))
    for i = 1:Int(ceil(length(u)/2))
        if abs(u[i] - u[i + t2]) < eps
                zc = [zc; i]
        end
    end
    if length(zc) > 10
        return h_crossings(u, eps/10)
    else
        return zc
    end
end

T = 10
a = 11
delay_nyqa(T, a)

