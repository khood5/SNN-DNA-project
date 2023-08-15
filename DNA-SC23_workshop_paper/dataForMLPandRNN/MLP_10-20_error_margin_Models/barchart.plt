#  example data
#  Title, box_y_value, box_y_error, box_y2_value, box_y2_error
#  "", 6, 2, 10, 2.5 

reset
set terminal png size 600,400 enhanced font ',18'
set style fill solid 2.00 border 0
set style histogram errorbars gap 2 lw 2
set style data histogram
set style fill pattern border -1
set linetype 1 lc rgb 'black'
set linetype 2 lc rgb '#555555'
set linetype 3 lc rgb '#999999'
set grid ytics
set ylabel "Number of Events"
set yrange [0:160]
set datafile separator ","

set xtics font "16" 
set ytics font "16"
set key outside below

# plot 'data.dat' using 2:3:xtic(1) title "Predicted" fs pattern 1, \
# '' using 4:5 title "Actual" fs pattern 7

set output './datFiles/50_nM_AR_600.png'
plot './datFiles/50_nM_AR_600.dat' using 2:3:xtic(1) title "Predicted" fs pattern 1 , \
'' using 4:5 title "Actual" fs pattern 7

set output './datFiles/100_nM_AR_600.png'
plot './datFiles/100_nM_AR_600.dat' using 2:3:xtic(1) title "Predicted" fs pattern 1, \
'' using 4:5 title "Actual" fs pattern 7

set output './datFiles/400_nM_AR_600.png'
plot './datFiles/400_nM_AR_600.dat' using 2:3:xtic(1) title "Predicted" fs pattern 1, \
'' using 4:5 title "Actual" fs pattern 7

set output './datFiles/800_nM_AR_600.png'
plot './datFiles/800_nM_AR_600.dat' using 2:3:xtic(1) title "Predicted" fs pattern 1, \
'' using 4:5 title "Actual" fs pattern 7

set output './datFiles/1200_nM_AR_600.png'
plot './datFiles/1200_nM_AR_600.dat' using 2:3:xtic(1) title "Predicted" fs pattern 1, \
'' using 4:5 title "Actual" fs pattern 7

set output './datFiles/1800_nM_AR_600.png'
plot './datFiles/1800_nM_AR_600.dat' using 2:3:xtic(1) title "Predicted" fs pattern 1, \
'' using 4:5 title "Actual" fs pattern 7