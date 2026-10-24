# Tests of the homology over ℤ (homology.jl): small complexes written by hand, the three
# tiny reference snapshots, and the file of expected values. The homology of all the
# snapshots of GAUSS_SNAPSHOT_DIR is checked in test_snapshots.jl.

isdefined(Main, :read_snapshot) || include("snapshot_io.jl")

using Gauss
using Gauss: MonoidAlgebra    # public, not exported (Oscar exports a MonoidAlgebra)
using Gauss: rank
using Test
import TOML

const _H_SA = Gauss.Complexes.SparseArrays
const EXPECTED_HOMOLOGY = joinpath(@__DIR__, "data", "expected", "complex_braid_groups.toml")

struct HomologyTestMonoid <: Gauss.Complexes.AbstractMonoid
    simples::Vector{String}
end

@testset "Homology" begin
    G = FinitelyGeneratedAbelianGroup
    GG = FinitelyGeneratedGradedAbelianGroup

    @testset "FinitelyGeneratedGradedModule" begin
        M = GG([G(1), G(0, [2]), G(0)])
        @test M isa FinitelyGeneratedGradedModule{BigInt}
        @test range(M) == 0:2 && firstindex(M) == 0 && lastindex(M) == 2
        @test M[1] == G(0, [2]) && M[end] == G(0)
        @test_throws BoundsError M[3]
        @test M == GG([G(1), G(0, [2]), G(0)])
        @test hash(M) == hash(GG([G(1), G(0, [2]), G(0)]))
        @test M != GG([G(1), G(0, [2]), G(0)]; lo = 1)
        @test range(GG([G(1)]; lo = 3)) == 3:3
        @test sprint(show, M) == "FinitelyGeneratedGradedAbelianGroup in degrees 0:2: ℤ; ℤ/2ℤ; 0"
        @test sprint(show, MIME("text/plain"), M) ==
              "FinitelyGeneratedGradedAbelianGroup in degrees 0:2\n  0: ℤ\n  1: ℤ/2ℤ\n  2: 0"
    end

    @testset "Complexes written by hand" begin
        circle = ChainComplex([_H_SA.spzeros(Int, Int, 1, 1)])
        @test homology(circle) == GG([G(1), G(1)])
        @test homology(circle, 1) == G(1)
        @test_throws BoundsError homology(circle, 2)

        rp2 = ChainComplex([_H_SA.spzeros(Int, Int, 1, 1), _H_SA.sparse([1], [1], [2], 1, 1)])
        @test homology(rp2) == GG([G(1), G(0, [2]), G(0)])
        @test [homology(rp2, n) for n in 0:2] == [G(1), G(0, [2]), G(0)]

        torus = ChainComplex([_H_SA.spzeros(Int, Int, 1, 2), _H_SA.spzeros(Int, Int, 2, 1)]; lo = 1)
        @test homology(torus) == GG([G(1), G(2), G(1)]; lo = 1)

        # 2-simplex: contractible.
        d0 = _H_SA.sparse([1, 2, 1, 3, 2, 3], [1, 1, 2, 2, 3, 3], [-1, 1, -1, 1, -1, 1], 3, 3)
        d1 = _H_SA.sparse([1, 2, 3], [1, 1, 1], [1, -1, 1], 3, 1)
        @test homology(ChainComplex([d0, d1])) == GG([G(1), G(0), G(0)])

        # A matrix with torsion and units: diag(1, 2, 6) on a complex 0 → ℤ³ → ℤ³ → 0.
        D = _H_SA.sparse([1, 2, 3], [1, 2, 3], [1, 2, 6], 3, 3)
        @test homology(ChainComplex([D])) == GG([G(0, [2, 6]), G(0)])

        # Keyword arguments go to the solver.
        @test homology(rp2; value_type = Int64, nworkers = 1) == homology(rp2)
    end

    @testset "Complexes over a monoid algebra" begin
        # Free commutative monoid on a, b (simples e = 0, a = 1, b = 2, ab = 3), Koszul
        # resolution.
        L2 = NTuple{2, Int32}
        B = MonoidAlgebra(Gauss.Complexes.AbstractAlgebra.ZZ, HomologyTestMonoid(["e", "a", "b", "ab"]))
        d0 = labelled_sparse([1, 1, 1, 1], [1, 1, 2, 2], L2[(1, 0), (0, 1), (2, 0), (0, 2)],
                             [1, -1, 1, -1], 1, 2)
        d1 = labelled_sparse([1, 1, 2, 2], [1, 1, 1, 1], L2[(2, 0), (0, 2), (1, 0), (0, 1)],
                             [-1, 1, 1, -1], 2, 1)
        T = ChainComplex([BasedFreeModule(B, ["[]"]), BasedFreeModule(B, ["[a]", "[b]"]),
                          BasedFreeModule(B, ["[a|b]"])], [d0, d1])
        @test_throws ArgumentError homology(T)
        @test_throws ArgumentError homology(T, 0)
        # Trivial coefficients: the torus; sign on the right: ℤ/2ℤ, ℤ/2ℤ, 0.
        @test homology(specialise(T)) == GG([G(1), G(2), G(1)])
        lengths = [0, 1, 1, 2]                    # lengths[l + 1] = ℓ(label l)
        @test homology(specialise(T, l -> isodd(lengths[l[2] + 1]) ? -1 : 1)) ==
              GG([G(0, [2]), G(0, [2]), G(0)])
    end

    @testset "Tiny reference snapshots" begin
        tiny = joinpath(@__DIR__, "data", "snapshots")
        # Braid group on three strands; Artin group of type B₃ (Salvetti complex by hand).
        braid3 = GG([G(1), G(1), G(0)])
        artinB3 = GG([G(1), G(2), G(2), G(1)])
        for (name, expected) in (("classic_A2_a", braid3), ("dual_A2_a", braid3),
                                 ("dual_B3_a", artinB3))
            s = read_snapshot(joinpath(tiny, name * ".txt.gz"))
            @test homology(specialise(s.complex)) == expected
        end
    end

    @testset "Expected values" begin
        # Tables 1, 3 and 4 of the article named in [source], with the errata (f1 of J1.f).
        expected = TOML.parsefile(EXPECTED_HOMOLOGY)
        @test expected["source"]["arxiv"] == "2302.10033v2"
        @test expected["source"]["tables"] == [1, 3, 4]
        groups = filter(!=("source"), collect(keys(expected)))
        @test sort!(parse.(Int, chop.(groups; head = 1, tail = 0))) == 4:37
        @test count(g -> expected[g]["well_generated"], groups) == 26
        keys_known = Set(["well_generated", "real", "isodiscriminantal", "trivial", "sign",
                          "cells_mar17", "cells_optimised", "two_cell_bounds", "errata"])
        classes = Dict{String, Vector{String}}()
        n_errata = 0
        for group in groups
            entry = expected[group]
            @test issubset(keys(entry), keys_known)
            # Written in the form displayed by FinitelyGeneratedAbelianGroup.
            @test all(string(parse(G, s)) == s for s in [entry["trivial"]; entry["sign"]])
            trivial = GG([parse(G, s) for s in entry["trivial"]])
            sign = GG([parse(G, s) for s in entry["sign"]])
            @test range(sign) == range(trivial)
            @test trivial[0] == G(1)
            @test sign[0] == G(0, [2])
            # Braid groups of irreducible reflection groups: Euler characteristic 0, the
            # alternating sum of the free ranks for any module of rank one.
            @test sum((-1)^n * rank(trivial[n]) for n in range(trivial)) == 0
            @test sum((-1)^n * rank(sign[n]) for n in range(sign)) == 0
            for key in ("cells_mar17", "cells_optimised")
                haskey(entry, key) || continue
                cells = entry[key]
                @test length(cells) == length(range(trivial))
                @test sum((-1)^(n - 1) * cells[n] for n in eachindex(cells)) == 0
            end
            for erratum in get(entry, "errata", Dict{String, Any}[])
                @test erratum["key"] in ("trivial", "sign", "cells_mar17", "cells_optimised")
                @test string(entry[erratum["key"]][erratum["degree"] + 1]) != erratum["printed"]
                @test erratum["listed"] isa Bool && !isempty(erratum["reason"])
                n_errata += 1
            end
            haskey(entry, "isodiscriminantal") &&
                push!(get!(classes, entry["isodiscriminantal"], String[]), group)
        end
        @test n_errata == 3
        @test length(classes) == 10
        # The groups of an isodiscriminantality class have the same homology.
        for members in values(classes), group in members
            @test expected[group]["trivial"] == expected[first(members)]["trivial"]
            @test expected[group]["sign"] == expected[first(members)]["sign"]
        end
        @test expected["G37"]["real"] == "E8"
    end
end
