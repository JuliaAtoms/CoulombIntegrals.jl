if VERSION < v"1.1-DEV"
    isnothing(::Nothing) = true
    isnothing(::Any) = false
end

function error_plot(fun::Function, Ns::AbstractVector{<:Integer},
                    ρcol::Int,
                    errors::Vector{<:Tuple{Int,String,<:Real,<:Real}},
                    elapsed_col::Int,
                    headers::Vector{String},
                    other_vectors::AbstractVector...;
                    kwargs...)
    sa = subplot(211)
    sb = subplot(212)

    numchoices = prod(length.(other_vectors))

    plot_fun = (i,data) -> begin
        ρ = data[:,ρcol]
        
        sca(sa)
        line = nothing
        for ((ecol,elabel,_,_),style) in zip(errors,Iterators.cycle(["-", ":", "--"]))
            line = loglog(ρ,abs.(data[:,ecol]), "s$style",
                          color = !isnothing(line) ? line[:get_color]() : nothing)[1]
        end

        sca(sb)
        loglog(ρ,data[:,elapsed_col],"s-")

        if i == numchoices
            sca(sa)
            for p in 1:4
                yp = ρ.^p
                loglog(ρ, yp  * data[end,errors[1][1]]/yp[end]
                       , ":",
                       label=latexstring("\$p=$p\$"))
            end
        end        
    end

    result = test_convergence_rates(fun, Ns, ρcol, errors, headers, other_vectors...;
                                    plot_fun=plot_fun, kwargs...)

    sca(sa)
    no_tick_labels()
    ylabel("Error")

    sca(sb)
    xlabel(L"\rho")
    ylabel("Elapsed time [s]")
    tight_layout()

    result
end
