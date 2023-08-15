set terminal png size 600,400 enhanced
set output 'output2.png'

set yrange [0:160]
set style line 2 lc rgb 'black' lt 1 lw 1
set style data histogram
set style histogram cluster gap 1
set style fill pattern border -1
set boxwidth 0.9
set xtics format ""
set grid ytics

set title ""
plot "data.dat" using 2:xtic(1) title "Real" ls 2,    \
     "data.dat" using 3 title "Result" ls 2,     \