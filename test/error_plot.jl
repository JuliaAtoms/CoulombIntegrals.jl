if VERSION < v"1.1-DEV"
    isnothing(::Nothing) = true
    isnothing(::Any) = false
end

function error_plot(fun::Function,
                    Rf, rₘₐₓ, Ns::AbstractVector{<:Integer},
                    ρcol::Int,
                    errors::Vector{<:Tuple{Int,String,<:Real,<:Real}},
                    elapsed_col::Int,
                    headers::Vector{String},
                    other_vectors::AbstractVector...;
                    sa = subplot(311),
                    sb = subplot(312),
                    sc = subplot(313),
                    kwargs...)
    plot_fun = (i,data) -> begin
        ρ = data[:,ρcol]

        line = nothing
        for ((ecol,elabel,_,_),style) in zip(errors,Iterators.cycle(["-", ":", "--"]))
            sca(sa)
            line = loglog(ρ,abs.(data[:,ecol]), "s$style",
                          color = !isnothing(line) ? line.get_color() : nothing)[1]
            sca(sc)
            loglog(abs.(data[:,ecol]),data[:,elapsed_col], "s$style",
                   color = !isnothing(line) ? line.get_color() : nothing)
        end

        sca(sb)
        loglog(ρ,data[:,elapsed_col],"s-")
    end

    result = test_convergence_rates(fun, Rf, rₘₐₓ, Ns, ρcol, errors, headers, other_vectors...;
                                    plot_fun=plot_fun, kwargs...)

    sca(sa)
    axes_labels_opposite(:x)
    ylabel("Error")
    xlabel(L"\rho")

    sca(sb)
    no_tick_labels()
    ylabel("Elapsed time [s]")

    sca(sc)
    xlabel("Error")
    ylabel("Elapsed time [s]")
    tight_layout()

    result
end
