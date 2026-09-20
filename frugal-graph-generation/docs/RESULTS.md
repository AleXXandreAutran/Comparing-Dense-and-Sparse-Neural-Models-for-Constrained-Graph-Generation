# Recorded research experiments

32 nodes, budget 64, seeds [11, 23, 37]. Values below are means across seeds.

| Family | Method | Degree MMD² | Clustering MMD² | Spectral MMD² | Valid | Planar |
|---|---|---:|---:|---:|---:|---:|
| sbm | random | 0.0244 | 0.2412 | 0.0340 | 1.0000 | 0.0000 |
| sbm | fitted_sbm | 0.0363 | 0.1010 | 0.0270 | 1.0000 | 0.0000 |
| sbm | degree_prior | 0.2687 | 0.1485 | 0.0453 | 1.0000 | 0.0000 |
| sbm | dense_s1 | 0.0282 | 0.2365 | 0.0372 | 1.0000 | 0.0000 |
| sbm | sparse_s1 | 0.1253 | 0.6802 | 0.0578 | 1.0000 | 0.0000 |
| sbm | dense_s4 | 0.0236 | 0.1392 | 0.0369 | 1.0000 | 0.0000 |
| sbm | sparse_s4 | 0.2417 | 0.7992 | 0.0697 | 1.0000 | 0.0000 |
| sbm | dense_s8 | 0.0475 | 0.0650 | 0.0326 | 1.0000 | 0.0000 |
| sbm | sparse_s8 | 0.3470 | 1.2003 | 0.0808 | 1.0000 | 0.0000 |
| sbm | sparse_projected | 0.0870 | 0.2702 | 0.0428 | 1.0000 | 0.0000 |
| sbm | sparse_fixed_candidates | 0.3351 | 1.2315 | 0.0828 | 1.0000 | 0.0000 |
| sbm | sparse_no_mp | 0.4085 | 1.2230 | 0.0799 | 1.0000 | 0.0000 |
| sbm | sparse_dense_candidates | 0.2794 | 1.2722 | 0.0877 | 1.0000 | 0.0000 |
| sbm | sparse_degree6 | 0.3894 | 1.2458 | 0.0842 | 1.0000 | 0.0000 |
| sbm | sparse_int8 | 0.3140 | 1.2602 | 0.0827 | 1.0000 | 0.0000 |
| sbm | reference_sample | 0.0422 | 0.0374 | 0.0235 | 1.0000 | 0.0000 |
| planar | random | 0.2782 | 1.0148 | 0.1304 | 1.0000 | 0.0000 |
| planar | fitted_sbm | 0.1231 | 0.3198 | 0.0619 | 1.0000 | 0.0000 |
| planar | degree_prior | 0.5139 | 0.9608 | 0.1392 | 1.0000 | 0.0000 |
| planar | dense_s1 | 0.2766 | 1.0373 | 0.1311 | 1.0000 | 0.0000 |
| planar | sparse_s1 | 0.0249 | 1.1977 | 0.1646 | 1.0000 | 0.0000 |
| planar | dense_s4 | 0.0806 | 0.9421 | 0.1159 | 1.0000 | 0.0000 |
| planar | sparse_s4 | 0.0395 | 1.2886 | 0.1742 | 1.0000 | 0.0000 |
| planar | dense_s8 | 0.0185 | 0.7951 | 0.1005 | 1.0000 | 0.0000 |
| planar | sparse_s8 | 0.0799 | 1.4002 | 0.1905 | 1.0000 | 0.0000 |
| planar | sparse_projected | 0.0564 | 1.0024 | 0.1305 | 1.0000 | 0.0000 |
| planar | sparse_fixed_candidates | 0.0766 | 1.4212 | 0.1938 | 1.0000 | 0.0000 |
| planar | sparse_no_mp | 0.1008 | 1.4143 | 0.1982 | 1.0000 | 0.0000 |
| planar | sparse_dense_candidates | 0.0879 | 1.4134 | 0.1974 | 1.0000 | 0.0000 |
| planar | sparse_degree6 | 0.1225 | 1.4189 | 0.1981 | 1.0000 | 0.0000 |
| planar | sparse_int8 | 0.0637 | 1.3753 | 0.2033 | 1.0000 | 0.0000 |
| planar | reference_sample | 0.0262 | 0.0222 | 0.0213 | 1.0000 | 1.0000 |

## Scaling

Each setting runs in a fresh CPU process. Score-only and complete-generation scopes are separate.

| Method | Nodes | ms median | RSS MiB | RSS delta MiB | Candidates |
|---|---:|---:|---:|---:|---:|
| sparse_score | 32 | 0.200 | 270.94 | 0.03 | 141 |
| dense_candidate_score | 32 | 0.234 | 270.80 | 0.03 | 496 |
| sparse_generation | 32 | 3.422 | 270.81 | 0.03 | 172 |
| dense_generation | 32 | 0.907 | 270.86 | 0.05 | 496 |
| sparse_int8 | 32 | 4.113 | 272.16 | 0.03 | 174 |
| sparse_score | 64 | 0.205 | 271.47 | 0.03 | 303 |
| dense_candidate_score | 64 | 0.406 | 272.28 | 0.03 | 2016 |
| sparse_generation | 64 | 7.787 | 271.84 | 0.03 | 364 |
| dense_generation | 64 | 1.776 | 272.89 | 0.05 | 2016 |
| sparse_int8 | 64 | 8.447 | 272.78 | 0.03 | 368 |
| sparse_score | 128 | 0.261 | 271.12 | 0.03 | 628 |
| dense_candidate_score | 128 | 1.175 | 279.52 | 0.03 | 8128 |
| sparse_generation | 128 | 22.526 | 272.33 | 0.05 | 751 |
| dense_generation | 128 | 5.965 | 282.72 | 2.38 | 8128 |
| sparse_int8 | 128 | 21.945 | 272.28 | 0.03 | 751 |
| sparse_score | 256 | 0.381 | 271.27 | 0.03 | 1263 |
| dense_candidate_score | 256 | 3.996 | 297.78 | 0.03 | 32640 |
| sparse_generation | 256 | 71.531 | 273.23 | 0.05 | 1517 |
| dense_generation | 256 | 20.359 | 314.12 | 0.05 | 32640 |
| sparse_int8 | 256 | 73.353 | 273.50 | 0.05 | 1517 |
| sparse_score | 512 | 0.601 | 273.06 | 0.03 | 2545 |
| dense_candidate_score | 512 | 15.285 | 382.80 | 0.03 | 130816 |
| sparse_generation | 512 | 288.886 | 274.47 | 0.03 | 3054 |
| dense_generation | 512 | 88.221 | 443.64 | 0.03 | 130816 |
| sparse_int8 | 512 | 311.324 | 275.34 | 0.03 | 3053 |

## Moving agents

| Method | Feasible / attempted | Mean edges | Mean churn after frame 0 | Mean λ₂ |
|---|---:|---:|---:|---:|
| rebuild | 90/90 | 64.00 | 13.816 | 0.1465 |
| retain | 90/90 | 64.00 | 1.678 | 0.2311 |
| learned_retain | 90/90 | 64.00 | 1.678 | 0.2311 |
