using RingMatrices
using Test


@testset "Basic" begin
    Q = RingMatrices.RingMatrix(0.9, 4)
    @test Q.index == [CartesianIndex(1,1), CartesianIndex(2,1),
                     CartesianIndex(3,2), CartesianIndex(4,3),
                     CartesianIndex(1,4)]
    @test Q.entries ≈ [0.1, 0.9,1.0,1.0,1.0]
    @test Q[1,1] ≈ 0.1
    @test Q[2,1] ≈ 0.9
    @test Q[3,2] ≈ Q[4,3] ≈ 1.0
    @test Q[1,4] ≈ 1.0
    @test Q[1,2] ≈ Q[1,3] ≈ 0.0
    @test Q[2,2] ≈ Q[2,3] ≈ Q[2,4] ≈ 0.0
    @test Q[3,1] ≈ Q[3,3] ≈ Q[3,4] ≈ 0.0
    @test Q[4,1] ≈ Q[4,2] ≈ Q[4,4] ≈ 0.0

    @test_throws BoundsError Q[5,1]

    Q2 = RingMatrices.RingMatrix(0.9, 4)

    QQ = RingMatrices.RingProductMatrix(Q,Q2)
    @test size(QQ) == (16,16)
    @test QQ[1:4,1:4] ≈ Q[1,1].*Q2
    @test QQ[5:8, 1:4] ≈ Q[2,1].*Q2
    @test QQ[9:12, 1:4] ≈ Q[3,1].*Q2
    @test QQ[13:16,1:4] ≈ Q[4,1].*Q2

    @test QQ[1:4, 5:8] ≈ Q[1,2].*Q2
    @test QQ[1:4, 9:12] ≈ Q[1,3].*Q2
    @test QQ[1:4, 13:16] ≈ Q[1,4].*Q2

    @test QQ[5:8, 5:8] ≈ Q[2,2].*Q2
    @test QQ[5:8, 9:12] ≈ Q[2,3].*Q2
    @test QQ[5:8, 13:16] ≈ Q[2,4].*Q2

    @test QQ[9:12, 5:8] ≈ Q[3,2].*Q2
    @test QQ[9:12, 9:12] ≈ Q[3,3].*Q2
    @test QQ[9:12, 13:16] ≈ Q[3,4].*Q2

    @test QQ[13:16, 5:8] ≈ Q[4,2].*Q2
    @test QQ[13:16, 9:12] ≈ Q[4,3].*Q2
    @test QQ[13:16, 13:16] ≈ Q[4,4].*Q2
end

@testset "Combinations" begin
    Q1 = RingMatrices.RingMatrix(0.9, 4)
    Q2 = RingMatrices.RingMatrix(0.9, 4)
    Q3 = RingMatrices.RingMatrix(0.9, 4)
    Qp = RingMatrices.PairwiseCombinations([Q1,Q2,Q3])
    @test Qp.nstates == 37
    @test size(Qp) == (64,64)
    @test length(Qp.entries) == length(Qp.index) == 61
    @test maximum(Qp.index) == CartesianIndex(61,61)
    @test maximum(Qp.sindex) == CartesianIndex(37,37)
    @test Qp[1,1] ≈ Qp.entries[1] ≈ 0.1^3
    @test Qp[3,5] ≈ 0.0

    @test_throws BoundsError Qp[0,1]
    @test_throws BoundsError Qp[65,1]

    Qm = RingMatrices.decompose(Qp)
    @test Qm == [Q1,Q2,Q3] 
end