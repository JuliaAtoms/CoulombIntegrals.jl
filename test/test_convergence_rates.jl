using PrettyTables

if VERSION < v"1.1-DEV"
    isnothing(::Nothing) = true
    isnothing(::Any) = false
end

function estimate_convergence_rate(ρ, ϵ)
    i = argmin(abs.(ϵ)) - 1

    ([log10.(ρ[1:i]) ones(i)] \ log10.(abs.(ϵ[1:i])))[1]
end

function test_convergence_rates(fun::Function, Ns::AbstractVector{<:Integer},
                                ρcol::Int,
                                errors::Vector{<:Tuple{Int,String,<:Real,<:Real}},
                                headers::Vector{String},
                                other_vectors::AbstractVector...;
                                convergence_rate_atol=1e-1,
                                plot_fun::Function = (i,data) -> nothing,
                                verbosity=1)
    ovs = Iterators.product(other_vectors...)

    convergence_rates = map(enumerate(ovs)) do (i,o)
        data = map(N -> fun(N, o...), Ns) |> d -> vcat(filter(!isnothing,d)...)
        isempty(data) && return nothing
        verbosity > 1 && pretty_table(data, headers)

        ρ = data[:,ρcol]
        label = join(string.(filter(c -> !(c isa Function), [o...])), ", ")
        
        plot_fun(i, data)

        map(errors) do (ecol,elabel,eexpected,rateexpected)
            curerr = data[:,ecol]
            minerr = minimum(abs, curerr)
            γ = estimate_convergence_rate(ρ, curerr)
            # We deem that the integration has passed if the minimal
            # error is below the requested threshold and the
            # convergence rate is approximately equal to or larger
            # than the requested order.
            pass = minerr < eexpected &&
                (isapprox(γ, rateexpected, atol=convergence_rate_atol) || γ > rateexpected)
            Any[minerr γ pass] # Without Any, pass will be coerced to Float64
        end |> e -> hcat(label, e...)
    end |> e -> vcat(filter(!isnothing, e)...)

    passjs = 1 .+ 3*(1:length(errors))
    
    if verbosity > 0
        pass = Highlighter((data,i,j) -> j>1 && mod(j-1,3)==0 && data[i,j],
                           bold = true, foreground = :green)
        fail = Highlighter((data,i,j) -> j>1 && mod(j-1,3)==0 && !data[i,j],
                           bold = true, foreground = :red)
        formatter = merge(ft_printf("%2.3e", passjs .- 2),
                          ft_printf("%2.3f", passjs .- 1),
                          Dict(j => (v,i) -> v ? "✓" : "⨯" for j in passjs))

        pretty_table(convergence_rates,
                     vcat("Case", [["$elabel, minδ [$eexpected]", "$elabel, rate [$rateexpected]", "Pass"]
                                   for (_,elabel,eexpected,rateexpected) in errors]...),
                     highlighters=(pass,fail),
                     formatter=formatter)
        fails = sum([count(.!convergence_rates[:,j]) for j in passjs])
        num_cases = length(errors)*size(convergence_rates,1)
        println("Number of fails: $fails ≡ $(100.0fails/num_cases)% fail rate")
    end
    
    all([all(convergence_rates[:,j]) for j in passjs])
end
