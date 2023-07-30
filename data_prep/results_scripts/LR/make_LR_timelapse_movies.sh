

for k in $(seq 0 1 4); do
    convert -delay 100 -loop 0 "Wood_inst_movie_"$k"_*.png" "LR_"$k"_inst_wood_100ms.gif"
done

for k in $(seq 0 1 4); do
    convert -delay 200 -loop 0 "Wood_inst_movie_"$k"_*.png" "LR_"$k"_inst_wood_200ms.gif"
done

for k in $(seq 0 1 4); do
    convert -delay 500 -loop 0 "Wood_inst_movie_"$k"_*.png" "LR_"$k"_inst_wood_500ms.gif"
done

for k in $(seq 0 1 4); do
    convert -delay 1000 -loop 0 "Wood_inst_movie_"$k"_*.png" "LR_"$k"_inst_wood_1000ms.gif"
done


for k in $(seq 0 1 4); do
    convert -delay 100 -loop 0 "All_inst_movie_"$k"_*.png" "LR_"$k"_inst_all_100ms.gif"
done

for k in $(seq 0 1 4); do
    convert -delay 200 -loop 0 "All_inst_movie_"$k"_*.png" "LR_"$k"_inst_all_200ms.gif"
done

for k in $(seq 0 1 4); do
    convert -delay 500 -loop 0 "All_inst_movie_"$k"_*.png" "LR_"$k"_inst_all_500ms.gif"
done

for k in $(seq 0 1 4); do
    convert -delay 1000 -loop 0 "All_inst_movie_"$k"_*.png" "LR_"$k"_inst_all_1000ms.gif"
done
