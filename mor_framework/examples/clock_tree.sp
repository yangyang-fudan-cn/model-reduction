* RC Clock Tree — simple RC-tree benchmark for model order reduction
* 4 fan-out stages
R1 1 2 10
C1 2 0 1e-12
R2 2 3 10
C2 3 0 1e-12
R3 2 4 10
C3 4 0 1e-12
R4 3 5 10
C4 5 0 1e-12
R5 3 6 10
C5 6 0 1e-12
R6 4 7 10
C6 7 0 1e-12
R7 4 8 10
C7 8 0 1e-12
V1 1 0 DC 0 AC 1 PULSE(0 1 0 1e-10 1e-10 5e-9 1e-8)
.TRAN 1e-11 2e-8
.AC DEC 50 1e3 1e11
.PRINT V(5) V(6) V(7) V(8)
.END
