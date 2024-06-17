#!/bin/zsh

runs=1

graph="barabasi_albert"
log2hinv=6
for N (100 500 1000 2000 5000) {
  echo "$graph $N $log2hinv $runs"
  ../build/solve_graph $graph $N $log2hinv $runs
}

N=1000
for log2hinv (4 6 8 10 12) {
  echo "$graph $N $log2hinv $runs"
  ../build/solve_graph $graph $N $log2hinv $runs
}

graph="dorogovtsev_goltsev_mendes"
log2hinv=6
for N (5 6 7 8 9) {
  echo "$graph $N $log2hinv $runs"
  ../build/solve_graph $graph $N $log2hinv $runs
}

N=7
for log2hinv (4 6 8 10 12) {
  echo "$graph $N $log2hinv $runs"
  ../build/solve_graph $graph $N $log2hinv $runs
}
