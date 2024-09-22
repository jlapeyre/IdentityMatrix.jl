@setup_workload begin
    # Putting some things in `setup` can reduce the size of the
    # precompile file and potentially make loading faster.
    nothing
    using IdentityMatrix
    @compile_workload begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to your package or not (on Julia 1.8 and higher)
        # These don't take very long to compile anyway.
        # Especially for larger values of N, the run time is much larger than compile time.
        mrand = rand(4, 4)
        nrange = [1:20..., 32, 64, 128, 256]
        for N in nrange
            kron(Id(ComplexF64, N), mrand)
            kron(Id(N), mrand)
            kron(Id(Int, N), mrand)
            kron(Id(N), Id(N))
        end
    end
end
